from functools import partial

import torch
from transformers import GPTNeoForCausalLM, AutoConfig

from ml.split_checkpoint import SplitCheckpoint


# GPT-J 6B config
GPT_J_CONFIG = AutoConfig.from_pretrained("EleutherAI/gpt-neo-2.7B")
GPT_J_CONFIG.attention_layers = ["global"] * 28
GPT_J_CONFIG.attention_types = [["global"], 28]
GPT_J_CONFIG.num_layers = 28
GPT_J_CONFIG.num_heads = 16
GPT_J_CONFIG.hidden_size = 256 * GPT_J_CONFIG.num_heads
GPT_J_CONFIG.vocab_size = 50400
GPT_J_CONFIG.rotary = True
GPT_J_CONFIG.rotary_dim = 64
GPT_J_CONFIG.jax = True


def no_init(loading_code):
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy

    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]

    return result


def _load_gpt_j_split_ckpt(ckpt_dir, config=GPT_J_CONFIG):
    model = GPTNeoForCausalLM.from_pretrained(
        pretrained_model_name_or_path=None,
        config=config,
        state_dict=SplitCheckpoint(ckpt_dir),
    )

    return model


def _init_gptj(config=GPT_J_CONFIG):
    return GPTNeoForCausalLM(config=config)


def quick_init_gptj(config=GPT_J_CONFIG):
    return no_init(partial(_init_gptj, config=config))


def load_gpt_j_split_ckpt(ckpt_dir, config=GPT_J_CONFIG):
    return no_init(partial(_load_gpt_j_split_ckpt, ckpt_dir=ckpt_dir, config=config))


def load_gpt_j_split_ckpt_state_dict(ckpt_dir):
    return SplitCheckpoint(ckpt_dir)
