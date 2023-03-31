import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from omegaconf import OmegaConf

from .dataset import KonIQDataset
from .transforms import get_transform   


class KonIQDataModule(pl.LightningDataModule):

    def __init__(self, train_dataset_cfg, eval_dataset_cfg, train_dataloder_cfg, eval_dataloder_cfg):
        super().__init__()

        data_transforms = get_transform()
        self.train_set = KonIQDataset(**train_dataset_cfg, transform = data_transforms['KonIQ']['train'], mode = 'train')
        self.val_set = KonIQDataset(**eval_dataset_cfg, transform = data_transforms['KonIQ']['test'], mode = 'test')
        self.test_set = KonIQDataset(**eval_dataset_cfg, transform = data_transforms['KonIQ']['test'], mode = 'test')

        self.train_dataloder_cfg = train_dataloder_cfg
        self.eval_dataloder_cfg = eval_dataloder_cfg

    def train_dataloader(self):
        return DataLoader(dataset=self.train_set, **self.train_dataloder_cfg,drop_last = True)

    def val_dataloader(self):
        return DataLoader(dataset=self.val_set, **self.eval_dataloder_cfg,drop_last = True)

    def test_dataloader(self):
        return DataLoader(dataset=self.test_set, **self.eval_dataloder_cfg,drop_last = True)