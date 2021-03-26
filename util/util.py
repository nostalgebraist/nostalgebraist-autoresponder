from typing import NamedTuple


def typed_namedtuple_to_dict(tup: NamedTuple) -> dict:
    return {name: getattr(tup, name) for name in tup._fields}
