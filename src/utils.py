import torch
import os
from src.dataset import Audiovisual_dataset

def save_model(args, model, name=''):
    torch.save(model, f'saved_models/{name}.pth')


def load_model(args, name=''):
    model = torch.load(f'saved_models/{name}.pth')
    return model


def get_data(dataset, split='train'):
    data = Audiovisual_dataset( dataset, split)
    return data
