# TODO: (nwo) DELETE THIS FILE after nwo is stable!!

import inspect

from munging.autoresponder_static import find_all_control_chars_chinese, T_CHAR
from munging.autoresponder_static_v8 import final_munge_before_neural_v10_1, final_munge_after_neural_v10_1, \
    TIME_SIDECHANNEL_CHAR

USE_NWO_MUNGE_AFTER = True


def infer_using_nwo_from_text(text: str, verbose=True) -> bool:
    _, _, _, caller_of_caller, _, _ = inspect.stack()[2]

    cchars_chinese = find_all_control_chars_chinese(text)
    if TIME_SIDECHANNEL_CHAR in text:
        cchars_chinese.append(TIME_SIDECHANNEL_CHAR)
    if T_CHAR in text:
        cchars_chinese.append(T_CHAR)

    using_nwo = len(cchars_chinese) == 0

    if verbose:
        msg = f"{caller_of_caller}: using_nwo={using_nwo} (cchars {cchars_chinese})"
        print(msg)

    return using_nwo


def final_munge_before_neural_nwo_transition(doc, *args, **kwargs):
    using_nwo = infer_using_nwo_from_text(doc)

    if using_nwo:
        return doc

    return final_munge_before_neural_v10_1(doc, *args, **kwargs)


def final_munge_after_neural_nwo_transition(doc, *args, **kwargs):
    if USE_NWO_MUNGE_AFTER:
        return doc
    return final_munge_after_neural_v10_1(doc, *args, **kwargs)
