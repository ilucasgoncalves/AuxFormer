import os
import librosa

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

class Wav2VecExtractor:
    def __init__(self):
        self.preprocessor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to('cuda')
        self.model.eval()

    def extract_all(self, raw_wav_list):
        wav2vec_list = []
        for raw_wav in tqdm(raw_wav_list):
            wav_input_16khz = torch.Tensor(raw_wav)
            wav_input_16khz = wav_input_16khz.cuda()
            x = self.preprocessor(wav_input_16khz, return_tensors="pt", sampling_rate=16000, padding="longest").input_values
            x = x.cuda()
            with torch.no_grad():
                z = self.model(x).last_hidden_state
            z = z.squeeze(0).cpu().numpy()
            wav2vec_list.append(z)
        return wav2vec_list


def extract_wav(wav_path):
    raw_wav, _ = librosa.load(wav_path, sr=16000)
    return raw_wav


class WavExtractor:
    def __init__(self, *args, **kwargs):
        self.wav_path_list = kwargs.get("wav_paths", args[0])
        self.nj = kwargs.get("nj", 24)
    def extract(self):
        print("Extracting wav files")
        with Pool(self.nj) as p:
            wav_list = list(tqdm(p.imap(extract_wav, self.wav_path_list), total=len(self.wav_path_list)))
        return wav_list

class VidExtractor:
    def __init__(self, *args, **kwargs):
        self.vid_path_list = kwargs.get("wav_paths", args[0])
        self.nj = kwargs.get("nj", 24)
    def extract(self):
        print("Extracting video files")
        vid_list = []
        for vid_loc in tqdm(self.vid_path_list):
            frames = os.listdir(vid_loc)
            feats = []
            for frame in frames:
                feats.append(list(np.load(vid_loc + '/' + frame)))
            vid_list.append(np.array(feats))
        return vid_list

def unpack_torch_segment(padded_segment, duration):
    batch_num = padded_segment.size(0)
    result = []
    for idx in range(batch_num):
        cur_segment = padded_segment[idx]
        
        cur_dur = duration[idx]
        cut_seg = cur_segment[:cur_dur]
        result.append(cut_seg)
    resutl = torch.Tensor(result)
    return result

