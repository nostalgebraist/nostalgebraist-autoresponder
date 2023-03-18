import torch

try:
    from collections.abc import MutableMapping
except ImportError:
    from collections import MutableMapping
from pathlib import Path


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

    def __contains__(self, key):
        return key in self.checkpoint

    def __iter__(self):
        for key in self.checkpoint:
            yield (key, self.__getitem__(key))

    def __copy__(self):
        return SplitCheckpoint(self.ckpt_dir, device=self.device)

    def copy(self):
        return SplitCheckpoint(self.ckpt_dir, device=self.device)

    @staticmethod
    def save_from_state_dict(state_dict, ckpt_dir):
        ckpt_dir = Path(ckpt_dir)
        
        ckmap = {}
        ckid = 0
        for name in sorted(state_dict.keys()):
            ckmap[name] = f"{ckpt_dir}/b{ckid}.pt"
            ckid += 1
            torch.save(state_dict[name], ckmap[name])
        
        torch.save(ckmap, f"{ckpt_dir}/m.pt")
