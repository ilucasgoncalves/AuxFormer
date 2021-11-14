import numpy as np
from torch.utils.data.dataset import Dataset
import pickle
import os
from scipy import signal
import torch

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')
       

class Audiovisual_dataset(Dataset):
    def __init__(self, dataset, split_type='train'):
        super(Audiovisual_dataset, self).__init__()

        self.vision = torch.tensor(np.array(dataset[split_type]['vision']).astype(np.float32)).cpu().detach()
        self.audio = np.array(dataset[split_type]['audio']).astype(np.float32)
        self.audio[self.audio == -np.inf] = 0
        self.audio = torch.tensor(self.audio).cpu().detach()
        self.labels = torch.tensor(np.array(dataset[split_type]['labels']).astype(np.float32)).cpu().detach()
        
        self.n_modalities = 2 # vision/ audio
        
    def get_n_modalities(self):
        return self.n_modalities
    def get_seq_len(self):
        return self.audio.shape[1], self.vision.shape[1]
    def get_dim(self):
        return self.audio.shape[2], self.vision.shape[2]
    def get_lbl_info(self):
        return self.labels.shape[1], self.labels.shape[2]
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        X = (index, self.audio[index], self.vision[index])
        Y = self.labels[index]
        
        return X, Y  

