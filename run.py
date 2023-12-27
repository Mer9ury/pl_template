
import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import WandbLogger

import argparse
from omegaconf import DictConfig, OmegaConf
import os
from pathlib import Path
import numpy as np

from inrct.runner.runner import Runner
from inrct.dataset.datamodule import *

parser = argparse.ArgumentParser()
parser.add_argument("--config", "-c", action="append", type=str, default=[])
parser.add_argument("--gpu", "-g", type=lambda x: [int(i) for i in x.split(',')],  default=[0], required=False)
parser.add_argument("--project", "-p", type=str, default=None)
parser.add_argument("--name", "-n", type=str, default=None)
parser.add_argument("--lr", type=float, default=None)
parser.add_argument("--test_only", action="store_true", default=False)
parser.add_argument("--seed", "-s", type=int, default=None)
parser.add_argument("--use_wandb", action="store_true", default=False)

parser.add_argument(
        "--cfg_options",
        default=[],
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )
    
def show(x):
    print(f"type: {type(x).__name__}, value: {repr(x)}")

def main(cfg: DictConfig):

    cfg = parse_cfg(args, instantialize_output_dir=True)

    pl.seed_everything(cfg.seed, True)

    w_logger = WandbLogger(project=args.project, name=args.name, save_dir=cfg.runner_cfg.output_dir)
    trainer = pl.Trainer(
        default_root_dir = cfg.runner_cfg.output_dir,
        logger=w_logger,
        **OmegaConf.to_container(cfg.trainer_cfg),
    )

    datamodule = get_datamodule(cfg.datamodule)(**OmegaConf.to_container(cfg.data_cfg))

    if not args.test_only:
        model = Runner(**OmegaConf.to_container(cfg.runner_cfg))
        trainer.fit(model, datamodule=datamodule)
    else:
        model = Runner.load_from_checkpoint(cfg.checkpoint_path, **OmegaConf.to_container(cfg.runner_cfg))
        trainer.test(model, datamodule=datamodule)

    
def parse_cfg(args, instantialize_output_dir=False):
    cfg = OmegaConf.merge(*[OmegaConf.load(config_) for config_ in args.config])
    extra_cfg = OmegaConf.from_dotlist(args.cfg_options)
    cfg = OmegaConf.merge(cfg, extra_cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.create())

    # Setup output_dir
    output_dir = Path(os.path.join('experiments',args.project, args.name))
    output_dir.mkdir(exist_ok=True, parents=True)

    seed = cfg.runner_cfg.seed if args.seed is None else args.seed 
    cli_cfg = OmegaConf.create(
        dict(
            config=args.config,
            test_only=args.test_only,
            runner_cfg=dict(seed=seed, output_dir=str(output_dir)),
            trainer_cfg=dict(fast_dev_run=False),
        )
    )
    cfg = OmegaConf.merge(cfg, cli_cfg)
    cfg.gpus = args.gpu
    cfg.seed = seed
    cfg.trainer_cfg.devices = args.gpu
    cfg.use_wandb = args.use_wandb
    if args.lr is not None:
        cfg.runner_cfg.optimizer_cfg.lr = args.lr

    if instantialize_output_dir:
        OmegaConf.save(cfg, str(output_dir / "config.yaml"))
    return cfg

if __name__ == "__main__":
    
    
    args = parser.parse_args()
    main(args)
