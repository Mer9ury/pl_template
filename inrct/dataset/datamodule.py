import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

from .dataset import MIT5KDataset
from .transforms import get_transform   


class MIT5KDataModule(pl.LightningDataModule):

    def __init__(self, train_dataset_cfg, eval_dataset_cfg, train_dataloder_cfg, eval_dataloder_cfg):
        super().__init__()

        data_transforms = get_transform()
        self.train_set = MIT5KDataset(**train_dataset_cfg, transform = data_transforms['MIT5K']['train'], mode = 'train')
        self.val_set = MIT5KDataset(**eval_dataset_cfg, transform = data_transforms['MIT5K']['test'], mode = 'test')
        self.test_set = MIT5KDataset(**eval_dataset_cfg, transform = data_transforms['MIT5K']['test'], mode = 'test')

        self.train_dataloder_cfg = train_dataloder_cfg
        self.eval_dataloder_cfg = eval_dataloder_cfg

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, **self.train_dataloder_cfg,drop_last = True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, **self.eval_dataloder_cfg,drop_last = True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, **self.eval_dataloder_cfg,drop_last = True)

def get_datamodule(name):
    if name == 'MIT5K':
        return MIT5KDataModule
    else:
        raise NotImplementedError
