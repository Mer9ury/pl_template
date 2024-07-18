import pytorch_lightning as pl
import torch
import torch.nn as nn

import numpy as np
from pathlib import Path
import cv2

from src.utils.optimizer import build_optimizer
from src.utils.lr_scheduler import build_lr_scheduler
from src.utils.metrics import plcc, srocc
from src.models.utils import get_model
import json


class Runner(pl.LightningModule):

    def __init__(self, seed, output_dir, optimizer_cfg, lr_scheduler_cfg, model_cfg):
        super().__init__()

        self.model = get_model(model_cfg)
        self.loss = nn.L1Loss()
        self.eps = 1e-6
        self.seed = seed

        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.output_dir = Path(output_dir)

    def forward(self, x):
        return self.model(x)

    def run_step(self, batch, batch_idx, is_test = False):
        x, y = batch

        y_hat = self.model(x)

        loss = self.loss(y_hat, y)

        metrics = self.compute_per_example_metrics(y_hat, y)

        return {"loss": loss, "y_hat": y_hat, "y": y, **metrics}
    
    
    def training_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx)
        self.logging(outputs, "train", on_step=True, on_epoch=True)
        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.run_eval_step(batch, batch_idx)
        self.val_outputs.append(outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.run_eval_step(batch, batch_idx)
        self.test_outputs.append(outputs)
        return outputs


    def on_train_epoch_end(self):
        self.lr_scheduler.step()
            
    def on_validation_epoch_end(self):
        self.eval_epoch_end(self.val_outputs, "val")
        self.val_outputs.clear()

    def on_test_epoch_end(self):
        self.eval_epoch_end(self.test_outputs, "test")
        self.test_outputs.clear()

    def eval_epoch_end(self, outputs, run_type):
        """_summary_

        Args:
            outputs (_type_): _description_
            run_type (_type_): _description_
            moniter_key: "{val/test}_epoch_{mae/acc}_{exp/max}_metric"
        """
        if len(outputs) == 1:
            return
        y_pred = torch.cat([i['y_hat'] for i in outputs],dim=0).squeeze()
        y = torch.cat([i['y'] for i in outputs],dim=0).squeeze()

        
        stats = {}

        stats['loss'] = loss = torch.stack([i['loss'] for i in outputs]).mean()

        metrics = self.compute_per_example_metrics(y_pred,y)
        stats.update(**metrics)

        for k, _stats in stats.items():
            try:
                stats[k] = torch.cat(_stats).mean().item()
            except RuntimeError:
                stats[k] = torch.stack(_stats).mean().item()
            except TypeError:
                stats[k] = _stats.item()
            self.log(f"{run_type}_{k}", stats[k], on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        with open(str(self.output_dir / f"{run_type}_stats.json"), "a") as f:
            f.write(json.dumps(stats) + "\n")

    def configure_optimizers(self):
        optimizer = build_optimizer(model = self.model, **self.optimizer_cfg)
        scheduler = build_lr_scheduler(optimizer=optimizer, **self.lr_scheduler_cfg)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def logging(self, outputs: dict, run_type: str, on_step=True, on_epoch=True):
        for k, v in outputs.items():
            if self._valid_key(k):
                self.log(f"{run_type}_{k}", v.mean(), on_step=on_step, on_epoch=on_epoch, prog_bar=False, logger=True)

    def compute_per_example_metrics(self, y_pred, y):
        mae = torch.mean(torch.abs(y_pred - y))
        y_pred = y_pred.squeeze().detach().cpu().numpy().astype(np.float32)
        y = y.squeeze().detach().cpu().numpy().astype(np.float32)

        sr = torch.Tensor([srocc(y_pred, y)])
        pl = torch.Tensor([plcc(y_pred, y)])

        return {f"mae": mae, f"srocc": sr, f"plcc": pl}

    def denormalize(self, x):
        return x * 255

    def viz(self, idx, x, y, y_hat):
        x = self.denormalize(x.squeeze().cpu()).detach().numpy().astype(np.uint8)
        y = self.denormalize(y.squeeze().cpu()).detach().numpy().astype(np.uint8)
        y_hat = self.denormalize(y_hat.squeeze().cpu()).detach().numpy().astype(np.uint8)
        x = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        y = cv2.cvtColor(y, cv2.COLOR_RGB2BGR)
        y_hat = cv2.cvtColor(y_hat, cv2.COLOR_RGB2BGR)
        print(str(self.output_dir / f"{self.current_epoch}_{idx}_input.png"))
        cv2.imwrite(str(self.output_dir / f"{self.current_epoch}_{idx}_input.png"), x)
        cv2.imwrite(str(self.output_dir / f"{self.current_epoch}_{idx}_gt.png"), y)
        cv2.imwrite(str(self.output_dir / f"{self.current_epoch}_{idx}_pred.png"), y_hat)
        