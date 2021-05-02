from typing import NamedTuple


def typed_namedtuple_to_dict(tup: NamedTuple) -> dict:
    return {name: getattr(tup, name) for name in tup._fields}


def copy_and_update_config(cls, config, **kwargs):
    old_d = typed_namedtuple_to_dict(config)
    new_d = {k: kwargs.get(k) if k in kwargs else v for k, v in old_d.items()}
    return cls(**new_d)
