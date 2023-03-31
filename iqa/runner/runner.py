import pytorch_lightning as pl
import torch
import torch.nn as nn

import numpy as np
from pathlib import Path

from mtiqa.models.my_model import MyModel
from mtiqa.utils.optimizer import build_optimizer
from mtiqa.utils.lr_scheduler import build_lr_scheduler
from mtiqa.utils.metrics import plcc, srocc


class Runner(pl.LightningModule):

    def __init__(self, seed, output_dir, optimizer_cfg, lr_scheduler_cfg, model_cfg):
        super().__init__()

        self.model = self.get_model(model_cfg)
        self.loss = nn.MSELoss()
        self.eps = 1e-6
        self.seed = seed

        self.optimizer_cfg = optimizer_cfg
        self.lr_scheduler_cfg = lr_scheduler_cfg
        self.output_dir = Path(output_dir)

    def forward(self, x):
        return self.model(x)

    def run_step(self, batch, batch_idx, is_test = False):
        x, y = batch
        y = y.float()

        y_hat = self(x)
        loss = self.loss(y_hat, y)

        metrics = self.compute_per_example_metrics(y_hat, y)
        return {"loss": loss, "y_hat": y_hat, "y": y, **metrics}
    
    def training_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx, is_test = False)

        return outputs

    def validation_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx, is_test = True)

        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self.run_step(batch, batch_idx, is_test = True)
        self.logging(outputs, "test", on_step=True, on_epoch=True)

        return outputs

    def eval_epoch_end(self, outputs, run_type):
        """_summary_

        Args:
            outputs (_type_): _description_
            run_type (_type_): _description_
            moniter_key: "{val/test}_epoch_{mae/acc}_{exp/max}_metric"
        """
        y_hats = [i['y_hat'] for i in outputs]
        y = [i['y'] for i in outputs]
        
        stats = {}
        metrics = self.compute_per_example_metrics(torch.cat(y_hats, dim=0), torch.cat(y, dim=0))
        stats.update(**metrics)

        mae = y_pred*100 - y

        for k, _stats in stats.items():
            try:
                stats[k] = torch.cat(_stats).mean().item()
            except RuntimeError:
                stats[k] = torch.stack(_stats).mean().item()
            except TypeError:
                stats[k] = _stats.item()
            self.log(f"{run_type}_{k}", stats[k], on_step=False, on_epoch=True, prog_bar=False, logger=True)

        stats["epoch"] = self.current_epoch
        stats["output_dir"] = str(self.output_dir)
        stats["ckpt_path"] = str(self.ckpt_path)
        with open(str(self.output_dir / f"{run_type}_stats.json"), "a") as f:
            f.write(json.dumps(stats) + "\n")

    def configure_optimizers(self):
        optimizer = build_optimizer(model = self.model, **self.optimizer_cfg)
        scheduler = build_lr_scheduler(optimizer=optimizer, **self.lr_scheduler_cfg)
        return optimizer
        
    def get_model(self, model_cfg):
        return MyModel()

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
    
    # def viz(self, outputs):
    #     i_f = [i['image_features'] for i in outputs]
    #     t_f = outputs[0]['text_features'].to(torch.float32)
    #     # print(torch.cdist(t_f[0][None,:], t_f[1][None,:]))
    #     # print(torch.cosine_similarity(t_f[0][None,:], t_f[1][None,:], dim=-1))
    #     i_f = torch.cat(i_f)
    #     # #do t-sne and visualize with i_t and t_f
    #     y = [i['y'].detach().cpu().numpy() for i in outputs]
    #     features = torch.cat([t_f,i_f], dim=0)
    #     labels = torch.cat([torch.zeros(i_f.shape[0]), torch.ones(t_f.shape[0])], dim=0)
    #     tsne = TSNE(n_components=2, init='pca', random_state=0)
    #     X_tsne = tsne.fit_transform(features.detach().cpu().numpy())
    #     color_y = np.concatenate(y[:features.shape[0]-1])
    #     plt.scatter(X_tsne[0, 0], X_tsne[0, 1], c='g')
    #     plt.scatter(X_tsne[1, 0], X_tsne[1, 1], c='r')
    #     plt.scatter(X_tsne[2:, 0], X_tsne[2:, 1], c=color_y, cmap=plt.cm.get_cmap("Blues", 10))
    #     plt.colorbar(ticks=range(10))
    #     plt.legend()
    #     plt.savefig(os.path.join(self.output_dir,f'{self.current_epoch}_tsne.png'))
    #     plt.clf() 

    #     #make scatter plot of y and y_hat
    #     y = np.array(y).flatten()
    #     y_pred = torch.cat(y_hats).detach().cpu().numpy()
    #     plt.scatter(y, y_pred, c='b', s=0.1)
    #     plt.savefig(os.path.join(self.output_dir,f'{self.current_epoch}_scatter.png'))
    #     plt.clf()

    #     plt.scatter(y, y - 100*y_pred, c = 'b', s=0.1)
    #     plt.savefig(os.path.join(self.output_dir,f'{self.current_epoch}_diff.png'))
    #     plt.clf()
    