from functools import partial
from typing import NamedTuple

import numpy as np
import torch


NostARHeadOptimizerParams = NamedTuple(
    "NostARHeadOptimizerParams",
    epochs=int,
    batch_size=int,
    base_lr=float,
    weight_decay=float,
    min_lr_frac=float,
    warmup_ratio=float,
    adam_beta1=float,
    adam_beta2=float,
)


def cosine_anneal_multiplier(step, total_steps, min_value=0.0):
    ratio = np.clip(step / total_steps, a_min=None, a_max=1.0)
    cos_mult = (1 + np.cos(np.pi * ratio)) / 2
    return min_value + (1.0 - min_value) * cos_mult


def warmup_multiplier(step, warmup_steps, min_value=0.0):
    warmup_steps = max(warmup_steps, 1)  # avoid div by zero
    ratio = np.clip(step / warmup_steps, a_min=min_value, a_max=1.0)
    return ratio


def cosine_anneal_warmup_multiplier(
    step,
    total_steps,
    warmup_steps,
    min_value_warmup=0,
    min_value_decay=0,
):
    mult1 = warmup_multiplier(step, warmup_steps, min_value=min_value_warmup)
    mult2 = cosine_anneal_multiplier(step, total_steps, min_value=min_value_decay)
    return mult1 * mult2


def get_nost_ar_head_optimizers(
    model, opt_params: NostARHeadOptimizerParams
):
    non_decay_vars = []
    decay_vars = []

    for name, param in model.named_parameters():
        if "ln.weight" in name or "logit_head.bias" in name:
            print(f"assigning '{name}' to non_decay_vars")
            non_decay_vars.append(param)
        else:
            print(f"assigning '{name}' to decay_vars")
            decay_vars.append(param)

    opt_decay = torch.optim.AdamW(
        params=decay_vars,
        lr=opt_params.base_lr,
        weight_decay=opt_params.weight_decay,
        betas=(opt_params.adam_beta1, opt_params.adam_beta2),
    )

    opt_no_decay = torch.optim.Adam(
        params=non_decay_vars,
        lr=opt_params.base_lr,
        betas=(opt_params.adam_beta1, opt_params.adam_beta2),
    )

    return opt_decay, opt_no_decay


def get_nost_ar_head_scheduler(
    opt: torch.optim.Optimizer, opt_params: NostARHeadOptimizerParams, data_len: int
):
    total_steps = opt_params.epochs * data_len // opt_params.batch_size

    lr_lambda = partial(
        cosine_anneal_warmup_multiplier,
        total_steps=total_steps,
        warmup_steps=opt_params.warmup_ratio * total_steps,
        min_value_warmup=0.0,
        min_value_decay=opt_params.min_lr_frac,
    )

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
