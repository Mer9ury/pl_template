"""
Modified from https://github.com/KaiyangZhou/deep-person-reid
"""
import logging
import warnings
from typing import Dict, Optional

import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
print = logger.info


def build_optimizer(
    model,
    optimizer_name: str,
    lr: float,
    weight_decay=0.0,
    momentum=0.0,
    sgd_dampening=0.0,
    sgd_nesterov=False,
    rmsprop_alpha=0.99,
    adam_beta1=0.9,
    adam_beta2=0.999,
    staged_lr: Optional[Dict] = None,
):
    """A function wrapper for building an optimizer.

    Args:
        model (nn.Module or iterable): model.
        optim_cfg (CfgNode): optimization config.
    """

    if staged_lr is not None:
        if not isinstance(model, nn.Module):
            warnings.warn(
                "When staged_lr is True, model given to "
                "build_optimizer() must be an instance of nn.Module."
                "You should reconstruct the param_groups manully"
            )
            param_groups = model
        else:
            if isinstance(model, nn.DataParallel):
                model = model.module

            param_groups = build_staged_lr_param_groups(model, lr, **staged_lr)

    else:
        if isinstance(model, nn.Module):
            param_groups = model.parameters()
        else:
            param_groups = model
    print(param_groups)
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )

    elif optimizer_name == "amsgrad":
        optimizer = torch.optim.Adam(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
            amsgrad=True,
        )

    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            dampening=sgd_dampening,
            nesterov=sgd_nesterov,
        )

    elif optimizer_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            param_groups,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            alpha=rmsprop_alpha,
        )

    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            betas=(adam_beta1, adam_beta2),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    return optimizer
