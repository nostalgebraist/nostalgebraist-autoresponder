import gc
import subprocess
import inspect
import time
from typing import NamedTuple


def typed_namedtuple_to_dict(tup: NamedTuple) -> dict:
    return {name: getattr(tup, name) for name in tup._fields}


def copy_and_update_config(cls, config, **kwargs):
    old_d = typed_namedtuple_to_dict(config)
    new_d = {k: kwargs.get(k) if k in kwargs else v for k, v in old_d.items()}
    return cls(**new_d)


def collect_and_show():
    collect_out = gc.collect()
    if collect_out > 0:
        print(f"gc.collect(): {collect_out}")


def show_gpu():
    s = subprocess.check_output("nvidia-smi").decode()
    try:
        l, mid, r = s.partition("MiB / ")
        memstr = l.split(" ")[-1] + mid + r.split(" ")[0]
        print(memstr)
    except:
        # not 100% sure i did the above right
        print(s)


def render_call_stack():
    calling_names = []
    for _, fname, _, name, _, _ in inspect.stack()[::-1][:-1]:
        _, valid, suffix = fname.partition("src/")
        if valid:
            calling_names.append(name)
    return " --> ".join(calling_names)


def chardec(enc, tokens):
    return [enc.decode([t]) for t in tokens]


class Timer:
    def __init__(self):
        self.start_t = None
        self.delta = 0

    @property
    def running(self):
        return self.start_t is not None

    def reset(self):
        self.start_t = None
        self.delta = 0

    def start(self):
        if not self.running:
            self.start_t = time.time()

    def stop(self):
        if self.running:
            self.delta += time.time() - self.start_t
        self.start_t = None

    def finish(self) -> float:
        self.stop()
        delta = self.delta
        self.delta = 0
        return delta
