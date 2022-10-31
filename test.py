# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
from tqdm import tqdm
import pandas as pd

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import Wav2Vec2Processor, Wav2Vec2Model



# Self-Written Modules
sys.path.append(os.getcwd())
import utils
import net

def main(args):
    utils.set_deterministic(args.seed)
    
    config_dict = utils.load_env(args.conf_path)
    assert config_dict.get("config_root", None) != None, "No config_root in config/conf.json"
    assert config_dict.get(args.corpus_type, None) != None, "Change config/conf.json"
    config_path = os.path.join(config_dict["config_root"], config_dict[args.corpus_type])
    utils.print_config_description(config_path)
    
    DataManager=utils.DataManager(config_path)
    lab_type = args.label_type

    if args.label_type == "dimensional":
        assert args.output_num == 6
    elif args.label_type == "categorical":
        emo_num = DataManager.get_categorical_emo_num()
        assert args.output_num == emo_num

    audio_path, video_path, label_path = utils.load_audio_and_label_file_paths(args) 

    fnames_aud, fnames_vid = [], []
    v_fnames = os.listdir(video_path)
    for fname_aud in os.listdir(audio_path):
        if fname_aud.replace('.wav','') in v_fnames:
            fnames_aud.append(fname_aud)
            fnames_vid.append(fname_aud.replace('.wav',''))
    fnames_aud.sort()
    fnames_vid.sort()

    

    test_wav_path = DataManager.get_wav_path(split_type="test",wav_loc=audio_path, fnames=fnames_aud, lbl_loc=label_path)
    test_vid_path = DataManager.get_vid_path(split_type="test",vid_loc=video_path, fnames=fnames_vid, lbl_loc=label_path)

    test_wav_path.sort()
    test_utts = [fname.split('/')[-1] for fname in test_wav_path]
    test_utts.sort()
    test_labs = DataManager.get_msp_labels(test_utts, lab_type=lab_type,lbl_loc=label_path)
    test_wavs = utils.WavExtractor(test_wav_path).extract()
    test_vids = utils.VidExtractor(test_vid_path).extract()
    ###################################################################################################
    with open(args.model_path+"/train_norm_stat.pkl", 'rb') as f:
        wav_mean, wav_std, vid_mean, vid_std = pk.load(f)


    test_set = utils.AudVidSet(test_wavs, test_vids, test_labs, test_utts, 
        print_dur=True, lab_type=lab_type,
        wav_mean = wav_mean, wav_std = wav_std,
        vid_mean = vid_mean, vid_std = vid_std, 
        label_config = DataManager.get_label_config(lab_type)
    )
 

    lm = utils.LogManager()
    if args.label_type == "dimensional":
        lm.alloc_stat_type_list(["test_aro", "test_dom", "test_val"])
    elif args.label_type == "categorical":
        lm.alloc_stat_type_list(["test_loss", "test_acc"])

    batch_size=args.batch_size
    test_loader = DataLoader(test_set, batch_size=batch_size, collate_fn=utils.collate_fn_padd, shuffle=False)
    
    model_path = args.model_path
    modelWrapper = net.ModelWrapper(args) # Change this to use custom model
    modelWrapper.init_model()
    modelWrapper.load_model(model_path, 'test')
    modelWrapper.set_eval()
 

    with torch.no_grad():
        total_pred = [] 
        total_y = []
        for xy_pair in tqdm(test_loader):
            xa = xy_pair[0]
            xv = xy_pair[1]
            y = xy_pair[2]
            mask = xy_pair[3]
        
            xa=xa.cuda(non_blocking=True).float()
            xv=xv.cuda(non_blocking=True).float()
            y=y.cuda(non_blocking=True).float()
            mask=mask.cuda(non_blocking=True).float()

            preds_va, preds_a, preds_v = modelWrapper.feed_forward(xa, xv, attention_mask=mask)
            pred = (preds_va + preds_a + preds_v) / 3


            total_pred.append(pred)
            total_y.append(y)
            
        total_pred = torch.cat(total_pred, 0)
        total_y = torch.cat(total_y, 0)
    

    if args.label_type == "categorical":
        if args.label_learning == "hard-label":
            loss = utils.CE_category(total_pred, total_y)
            

        acc = utils.calc_acc(total_pred, total_y)
        lm.add_torch_stat("test_loss", loss)
        lm.add_torch_stat("test_acc", acc)
    
    total_pred_np = total_pred.detach().cpu().numpy()
    total_y_np = total_y.detach().cpu().numpy()

    total_pred_pd = pd.DataFrame(total_pred_np)
    total_y_pd = pd.DataFrame(total_y_np)

    info_list = model_path.split("/")
    
    save_path = model_path + '/predictions/test/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    total_pred_pd.to_csv(save_path+'y_pred.csv')
    total_y_pd.to_csv(save_path+'y_true.csv')    

    utils.scores(model_path + '/predictions/test/')

    lm.print_stat()

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == "__main__":
    # Inputs for the main function
    parser = argparse.ArgumentParser()

    # Experiment Arguments
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        default='cuda',
        type=str)
    parser.add_argument(
        '--seed',
        default=0,
        type=int)
    parser.add_argument(
        '--conf_path',
        default="config/conf.json",
        type=str)

    # Data Arguments
    parser.add_argument(
        '--corpus_type',
        default="podcast_v1.7",
        type=str)
    parser.add_argument(
        '--model_type',
        default="wav2vec2",
        type=str)
    parser.add_argument(
        '--label_type',
        choices=['dimensional', 'categorical'],
        default='categorical',
        type=str)

    # Chunk Arguments
    parser.add_argument(
        '--use_chunk',
        default=False,
        type=str2bool)
    parser.add_argument(
        '--chunk_hidden_dim',
        default=256,
        type=int)
    parser.add_argument(
        '--chunk_window',
        default=50,
        type=int)
    parser.add_argument(
        '--chunk_num',
        default=11,
        type=int)
    
    # Model Arguments
    parser.add_argument(
        '--model_path',
        default=None,
        type=str)
    parser.add_argument(
        '--output_num',
        default=4,
        type=int)
    parser.add_argument(
        '--batch_size',
        default=128,
        type=int)
    parser.add_argument(
        '--hidden_dim',
        default=256,
        type=int)
    parser.add_argument(
        '--num_layers',
        default=3,
        type=int)
    parser.add_argument(
        '--epochs',
        default=100,
        type=int)
    parser.add_argument(
        '--lr',
        default=1e-5,
        type=float)
    
     # Label Learning Arguments
    parser.add_argument(
        '--label_learning',
        default="multi-label",
        type=str)

    parser.add_argument(
        '--corpus',
        default="USC-IEMOCAP",
        type=str)
    parser.add_argument(
        '--num_classes',
        default="four",
        type=str)
    parser.add_argument(
        '--label_rule',
        default="M",
        type=str)
    parser.add_argument(
        '--partition_number',
        default="1",
        type=str)
    parser.add_argument(
        '--data_mode',
        default="primary",
        type=str)
    
    parser.add_argument(
        '--output_dim',
        default=6,
        type=int)
    # Transformers Arguments
    parser.add_argument(
        '--attn_dropout', type=float, default=0.1,
        help='attention dropout')
    parser.add_argument(
        '--relu_dropout', type=float, default=0.1,
        help='relu dropout')
    parser.add_argument(
        '--embed_dropout', type=float, default=0.25,
        help='embedding dropout')
    parser.add_argument(
        '--res_dropout', type=float, default=0.1,
        help='residual block dropout')
    parser.add_argument(
        '--out_dropout', type=float, default=0.2,
        help='output layer dropout (default: 0.2')
    parser.add_argument(
        '--layers', type=int, default = 5,
        help='number of layers in the network (default: 5)')
    parser.add_argument(
        '--num_heads', type=int, default = 10,
        help='number of heads for multi-head attention layers(default: 10)')
    parser.add_argument(
        '--attn_mask', action='store_false',
        help='use attention mask for transformer (default: true)')
    parser.add_argument(
        '--clip', type = float, default = 0.8,
        help='gradient clip value (default: 0.8)')
    parser.add_argument(
        '--optim', type = str, default = 'Adam',
        help='optimizer to use (default: Adam)')
    parser.add_argument(
        '--decay', type = int, default = 6,
        help='When to decay learning rate (default: 5)')


    args = parser.parse_args()

    # Call main function
    main(args)
