import torch
from transformers import GPTNeoForCausalLM, AutoConfig

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping
from pathlib import Path


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


class SplitCheckpoint(MutableMapping):
    def __init__(self, ckpt_dir, device="cpu"):
        self.device = device
        self.ckpt_dir = Path(ckpt_dir)
        self.checkpoint = torch.load(str(ckpt_dir / Path("m.pt")))

    def __len__(self):
        return len(self.checkpoint)

    def __getitem__(self, key):
        path = self.ckpt_dir / Path(self.checkpoint[key]).name
        return torch.load(str(path), map_location=self.device)

    def __setitem__(self, key, value):
        return

    def __delitem__(self, key):
        return

    def keys(self):
        return self.checkpoint.keys()

    def __iter__(self):
        for key in self.checkpoint:
            yield (key, self.__getitem__(key))

    def __copy__(self):
        return SplitCheckpoint(self.ckpt_dir, device=self.device)

    def copy(self):
        return SplitCheckpoint(self.ckpt_dir, device=self.device)


def load_gpt_j_split_ckpt(ckpt_dir, config=GPT_J_CONFIG):
    model = GPTNeoForCausalLM.from_pretrained(
        pretrained_model_name_or_path=None,
        GPT_J_CONFIG=GPT_J_CONFIG,
        state_dict=SplitCheckpoint(ckpt_dir),
    )

    return model
