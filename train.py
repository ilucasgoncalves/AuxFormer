# -*- coding: UTF-8 -*-
# Local modules
import os
import sys
import argparse
# 3rd-Party Modules
import numpy as np
import pickle as pk
from tqdm import tqdm

# PyTorch Modules
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import torch.optim as optim
from transformers import Wav2Vec2Processor, Wav2Vec2Model, WavLMModel
# import shutil

# Self-Written Modules
sys.path.append(os.getcwd())
import utils
import net
import random
# from net import ser, chunk


def main(args):
    utils.set_deterministic(args.seed)
    utils.print_config_description(args.conf_path)

    config_dict = utils.load_env(args.conf_path)
    assert config_dict.get("config_root", None) != None, "No config_root in config/conf.json"
    # assert config_dict.get(args.corpus_type, None) != None, "Change config/conf.json"
    config_path = os.path.join(config_dict["config_root"], config_dict[args.corpus_type])
    utils.print_config_description(config_path)

    # Make model directory
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)


    # Initialize dataset
    DataManager=utils.DataManager(config_path)
    lab_type = args.label_type
    # print(lab_type)
    if args.label_type == "dimensional":
        assert args.output_num == 6

    if args.label_type == "categorical":
        emo_num = DataManager.get_categorical_emo_num()
        # print(emo_num)
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


    snum=10000000000000000
    train_wav_path = DataManager.get_wav_path(split_type="train",wav_loc=audio_path, fnames=fnames_aud, lbl_loc=label_path)[:snum]
    train_vid_path = DataManager.get_vid_path(split_type="train",vid_loc=video_path, fnames=fnames_vid, lbl_loc=label_path)[:snum]

    train_utts = [fname.split('/')[-1] for fname in train_wav_path]

    train_labs = DataManager.get_msp_labels(train_utts, lab_type=lab_type,lbl_loc=label_path)
    
    train_wavs = utils.WavExtractor(train_wav_path).extract()
    train_vids = utils.VidExtractor(train_vid_path).extract()
    
    
    dev_wav_path = DataManager.get_wav_path(split_type="dev", wav_loc=audio_path, fnames=fnames_aud, lbl_loc=label_path)[:snum]
    dev_vid_path = DataManager.get_vid_path(split_type="dev", vid_loc=video_path, fnames=fnames_vid, lbl_loc=label_path)[:snum]

    dev_utts = [fname.split('/')[-1] for fname in dev_wav_path]
    dev_labs = DataManager.get_msp_labels(dev_utts, lab_type=lab_type,lbl_loc=label_path)
    dev_wavs = utils.WavExtractor(dev_wav_path).extract()
    dev_vids = utils.VidExtractor(dev_vid_path).extract()
    ###################################################################################################

    train_set = utils.AudVidSet(train_wavs, train_vids, train_labs, train_utts, 
        print_dur=True, lab_type=lab_type,
        label_config = DataManager.get_label_config(lab_type)
    )
    
    dev_set = utils.AudVidSet(dev_wavs, dev_vids, dev_labs, dev_utts, 
        print_dur=True, lab_type=lab_type,
        wav_mean = train_set.wav_mean, wav_std = train_set.wav_std,
        vid_mean = train_set.vid_mean, vid_std = train_set.vid_std,
        label_config = DataManager.get_label_config(lab_type)
    )

    # print(train_set.wav_mean, train_set.wav_std, train_set.vid_mean, train_set.vid_std)
    
    train_set.save_norm_stat(model_path+"/train_norm_stat.pkl")
    
    total_dataloader={
        "train": DataLoader(train_set, batch_size=args.batch_size, collate_fn=utils.collate_fn_padd, shuffle=True),
        "dev": DataLoader(dev_set, batch_size=args.batch_size, collate_fn=utils.collate_fn_padd, shuffle=False)
    }

    # Initialize model
    modelWrapper = net.ModelWrapper(args) # Change this to use custom model
    modelWrapper.init_model()
    modelWrapper.init_optimizer()
    modelWrapper.load_model("wav2vec2-large-robust-finetunned/model/wav2vec2", 'train')

    
    # Initialize loss function
    lm = utils.LogManager()
    if args.label_type == "dimensional":
        lm.alloc_stat_type_list(["train_aro", "train_dom", "train_val",
            "dev_aro", "dev_dom", "dev_val"])
    elif args.label_type == "categorical":
        lm.alloc_stat_type_list(["train_loss", "train_acc", "dev_loss", "dev_acc"])

    epochs=args.epochs
    scaler = GradScaler()
    min_epoch = 0
    min_loss = 99999999999
    temp_dev = 99999999999
    losses_train, losses_dev = [], []
    for epoch in range(epochs):
        print("Epoch:",epoch)
        lm.init_stat()
        modelWrapper.set_train()
        for xy_pair in tqdm(total_dataloader["train"]):
            xa = xy_pair[0]
            xv = xy_pair[1]
            y = xy_pair[2]
            mask = xy_pair[3]

            #randomly shutting off modalities----------------------
            p1 = 0.2 
            p2 = 0.2 

            randn = torch.rand(1)

            if randn < p1:
                xa *= 0
                
            elif randn > p1 and randn < p1+p2:
                xv *= 0

            #randomly shutting off modalities----------------------  
            
            xa=xa.cuda(non_blocking=True).float()
            xv=xv.cuda(non_blocking=True).float()
            y=y.cuda(non_blocking=True).float()
            mask=mask.cuda(non_blocking=True).float()

            with autocast():

                #AuxFormer
                preds_va, preds_a, preds_v = modelWrapper.feed_forward(xa, xv, attention_mask=mask)
                if args.label_type == "categorical":
                    if args.label_learning == "hard-label":
                        # loss = utils.CE_category(pred, y)

                        #AuxFORMER
                        lossva = utils.CE_category(preds_va, y)
                        lossa = utils.CE_category(preds_a, y)
                        lossv = utils.CE_category(preds_v, y)
                        
                        wva, wa, wv = .3333, .3333, .3333
                        
                        loss = wva * lossva + wa * lossa + wv * lossv

                    pred = (preds_va + preds_a + preds_v) / 3
                    acc = utils.calc_acc(pred, y)
                    

            ## Backpropagation
            modelWrapper.backprop(loss)

            # Logging
            if args.label_type == "dimensional":
                lm.add_torch_stat("train_aro", ccc[0])
                lm.add_torch_stat("train_dom", ccc[1])
                lm.add_torch_stat("train_val", ccc[2])
            elif args.label_type == "categorical":
                lm.add_torch_stat("train_loss", loss)
                lm.add_torch_stat("train_acc", acc)

        modelWrapper.set_eval()

        with torch.no_grad():
            total_pred = [] 
            total_y = []
            for xy_pair in tqdm(total_dataloader["dev"]):
                xa = xy_pair[0]
                xv = xy_pair[1]
                y = xy_pair[2]
                mask = xy_pair[3]

                #randomly shutting off modalities----------------------
                p1 = 0.2 
                p2 = 0.2 

                randn = torch.rand(1)

                if randn < p1:
                    xa *= 0
                    
                elif randn > p1 and randn < p1+p2:
                    xv *= 0

                #randomly shutting off modalities---------------------- 
            
                xa=xa.cuda(non_blocking=True).float()
                xv=xv.cuda(non_blocking=True).float()
                y=y.cuda(non_blocking=True).float()
                mask=mask.cuda(non_blocking=True).float()

                #AuxFormer
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
            lm.add_torch_stat("dev_loss", loss)
            lm.add_torch_stat("dev_acc", acc)


        lm.print_stat()
        if args.label_type == "dimensional":
            dev_loss = 3.0 - lm.get_stat("dev_aro") - lm.get_stat("dev_dom") - lm.get_stat("dev_val")
        elif args.label_type == "categorical":
            dev_loss = lm.get_stat("dev_loss")
            tr_loss = lm.get_stat("train_loss")
            losses_dev.append(dev_loss)
            losses_train.append(tr_loss)
        if min_loss > dev_loss:
            min_epoch = epoch
            min_loss = dev_loss
        
        if float(dev_loss) < float(temp_dev):
            temp_dev = float(dev_loss)
            print('better dev loss found:' + str(float(dev_loss)) + ' saving model')
            modelWrapper.save_model(epoch)
    print("Save",end=" ")
    print(min_epoch, end=" ")
    print("")

    with open(model_path+'/train_loss.txt', 'w') as f:
        for item in losses_train:
            f.write("%s\n" % item)
    
    with open(model_path+'/dev_loss.txt', 'w') as f:
        for item in losses_dev:
            f.write("%s\n" % item)

    
    print("Loss",end=" ")
    if args.label_type == "dimensional":
        print(3.0-min_loss, end=" ")
    elif args.label_type == "categorical":
        print(min_loss, end=" ")
    print("")
    # modelWrapper.save_final_model(min_epoch, remove_param=False)

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
