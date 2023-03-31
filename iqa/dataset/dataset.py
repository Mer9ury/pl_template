from PIL import Image
import os

import torch
from torch.utils.data import Dataset
from torchvision import transforms

import pandas as pd

class KonIQDataset(Dataset):
    def __init__(self, data_dir, metadata, transform = None, mode = 'train'):
        self.data_dir = data_dir
        metadata = pd.read_csv(metadata)
        self.mode = mode
        if self.mode == 'train':
            self.imgs = metadata[(metadata.set=='training') | (metadata.set == 'validation')].reset_index()
            self.gt_labels = metadata[(metadata.set=='training') | (metadata.set == 'validation')].MOS.values
        else:
            self.imgs = metadata[metadata.set=='test'].reset_index()
            self.gt_labels = metadata[metadata.set=='test'].MOS.values

        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):


        path = os.path.join(self.data_dir, self.imgs['image_name'][idx])
        img = Image.open(path).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        label = torch.Tensor([self.gt_labels[idx]])

        return img, label