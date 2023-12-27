from PIL import Image
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from glob import glob
import numpy as np
import pandas as pd
import random

class MIT5KDataset(Dataset):
    def __init__(self, input_path, enhanced_path, transform, mode='train'):
        
        self.originals = glob(input_path+'/*')
        self.enhanced  = glob(enhanced_path+'/*')
        self.transform = transform
        if mode == 'train':
            self.originals = self.originals[:int(len(self.originals)*0.8)]
            self.enhanced  = self.enhanced[:int(len(self.enhanced)*0.8)]
        else:
            self.originals = self.originals[int(len(self.originals)*0.8):]
            self.enhanced  = self.enhanced[int(len(self.enhanced)*0.8):]

    def __len__(self):
        return len(self.originals)

    def __getitem__(self, idx):

        orig_img = Image.open(self.originals[idx])
        enh_img = Image.open(self.enhanced[idx])

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed) 
        orig_img = self.transform(orig_img)
        orig_img = orig_img.permute(1, 2, 0)

        random.seed(seed)
        torch.manual_seed(seed)
        enh_img = self.transform(enh_img)
        enh_img = enh_img.permute(1, 2, 0)

        return orig_img, enh_img