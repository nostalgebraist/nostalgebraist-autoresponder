from functools import partial
from typing import NamedTuple, Optional

import numpy as np
import torch
import torch.nn.functional as F


NostARHeadOptimizerParams = NamedTuple(
    "NostARHeadOptimizerParams",
    epochs=int,
    batch_size=int,
    base_lr=float,
    weight_decay=float,
    min_lr_frac=float,
    warmup_ratio=float,
    decay_ratio=Optional[float],
    adam_beta1=float,
    adam_beta2=float,
    classic_behavior_lr_sched=bool,
    block_lr=float,
    no_weight_decay_in_blocks=bool,
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


def classic_cosine_anneal_warmup_multiplier(
    step,
    total_steps,
    warmup_steps,
    min_value_warmup=0,
    min_value_decay=0,
):
    mult1 = warmup_multiplier(step, warmup_steps, min_value=min_value_warmup)

    cos_mult__decay_to_zero = cosine_anneal_multiplier(step, total_steps, min_value=0.)
    mult2 = np.clip(cos_mult__decay_to_zero, a_min=min_value_decay, a_max=None)

    return np.where(step <= warmup_steps, mult1, mult2)


def get_nost_ar_head_optimizers(
    model, opt_params: NostARHeadOptimizerParams
):
    no_weight_decay_in_blocks = opt_params.no_weight_decay_in_blocks
    print(f"no_weight_decay_in_blocks: {no_weight_decay_in_blocks}")

    non_decay_vars = []
    decay_vars = []

    non_decay_vars_blocks = []
    decay_vars_blocks = []

    for name, param in model.named_parameters():
        if name.split('.')[-2].startswith('ln') or "logit_head.bias" in name:
            if 'block' in name:
                print(f"assigning '{name}' to non_decay_vars_blocks")
                non_decay_vars_blocks.append(param)
            else:
                print(f"assigning '{name}' to non_decay_vars")
                non_decay_vars.append(param)
        else:
            if 'block' in name:
                if no_weight_decay_in_blocks:
                    print(f"assigning '{name}' to non_decay_vars_blocks")
                    non_decay_vars_blocks.append(param)
                else:
                    print(f"assigning '{name}' to decay_vars_blocks")
                    decay_vars_blocks.append(param)
            else:
                print(f"assigning '{name}' to decay_vars")
                decay_vars.append(param)

    param_groups = [
        {"params": non_decay_vars, "weight_decay": 0.0},
        {"params": decay_vars},
        {"params": non_decay_vars_blocks, "lr": opt_params.block_lr, "weight_decay": 0.0},
        {"params": decay_vars_blocks, "lr": opt_params.block_lr},

    ]

    opt = torch.optim.AdamW(
        params=param_groups,
        lr=opt_params.base_lr,
        weight_decay=opt_params.weight_decay,
        betas=(opt_params.adam_beta1, opt_params.adam_beta2),
    )

    # opt_no_decay = torch.optim.Adam(
    #     params=non_decay_vars,
    #     lr=opt_params.base_lr,
    #     betas=(opt_params.adam_beta1, opt_params.adam_beta2),
    # )

    return opt


def get_nost_ar_head_scheduler(
    opt: torch.optim.Optimizer,
    opt_params: NostARHeadOptimizerParams,
    data_len: int,
    grad_acc_steps=1,
):
    decay_ratio = opt_params.decay_ratio or 1 - opt_params.warmup_ratio
    total_steps = opt_params.epochs * data_len // (grad_acc_steps * opt_params.batch_size)
    decay_steps = decay_ratio * total_steps

    classic_behavior = opt_params.classic_behavior_lr_sched
    sched_fn = classic_cosine_anneal_warmup_multiplier if classic_behavior else cosine_anneal_warmup_multiplier

    lr_lambda = partial(
        sched_fn,
        total_steps=decay_steps,
        warmup_steps=opt_params.warmup_ratio * total_steps,
        min_value_warmup=0.0,
        min_value_decay=opt_params.min_lr_frac,
    )

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)


def cross_entropy_with_flooding(input, target, flood_level):
    loss_unreduced = F.cross_entropy(input, target, reduction='none')
    loss_flooded = torch.abs(loss_unreduced - flood_level) + flood_level
    loss_reduced = loss_flooded.mean()
    return loss_reduced


def make_huber_loss_from_logits(huber_delta):
    def huber_loss_from_logits(input, target):
        logit_diff = input[:, 1] - input[:, 0]
        return F.smooth_l1_loss(input=logit_diff, target=target, beta=huber_delta)
    return huber_loss_from_logits
