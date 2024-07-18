from PIL import Image
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from glob import glob
import numpy as np
import pandas as pd
import random

class HPDDataset(Dataset):
    def __init__(self, noise_data, score_data, seed, transform = None, mode = 'train'):

        self.noise_data = np.load(noise_data, allow_pickle=True)
        self.score_data = np.load(score_data, allow_pickle=True)
        self.transform = transform

        self.mode = mode
        idx = list(range(len(self.noise_data)))

        random.seed(seed)
        random.shuffle(idx)
        if mode == 'train':
            self.data = self.noise_data[idx[:int(len(idx)*0.8)]]
            self.score = self.score_data[idx[:int(len(idx)*0.8)]]
        if mode == 'test':
            self.data = self.noise_data[idx[int(len(idx)*0.8):]]
            self.score = self.score_data[idx[int(len(idx)*0.8):]]

        
    def __len__(self):  
        return len(self.data)

    def __getitem__(self, idx):

        img = self.data[idx]
        score = self.score[idx]
        if self.transform:
            img = self.transform(img)

        return img, score