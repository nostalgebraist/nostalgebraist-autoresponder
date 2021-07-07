# TODO: (nwo) DELETE THIS FILE after nwo is stable!!

from functools import wraps

from util.util import render_call_stack


def record_nwo_deprecated_activity(stack_string, last_caller):
    msg = f"!! NWO failure, {last_caller} was called. call stack:\n{stack_string}\n"
    print(msg)
    with open("nwo_violations.txt", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def nwo_deprecated(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        stack_string = render_call_stack()
        last_caller = f.__name__
        record_nwo_deprecated_activity(stack_string, last_caller)
        return f(*args, **kwargs)
    return wrapped


@nwo_deprecated
def _test_nwo_deprecated():
    print('hi')


def test_nwo_deprecated():
    _test_nwo_deprecated()
