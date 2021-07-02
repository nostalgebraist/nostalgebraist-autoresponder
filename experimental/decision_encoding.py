import re
from datetime import datetime

from autoresponder_static import DEFAULT_CSC
from autoresponder_static_v8 import get_ordered_interlocutors, timestamp_to_v10_format

now = datetime.now()
orig_poster_regex = DEFAULT_CSC["ORIG_POST_CHAR_NAMED"].format(user_name="([^ ]*)")


def get_orig_poster_name_if_present(doc: str):
    if DEFAULT_CSC['ORIG_POST_CHAR_FORUMLIKE'] in doc:
        return DEFAULT_CSC['user_name']

    for m in re.finditer(orig_poster_regex, doc):
        return " " + m.group(1)


def get_final_name(doc: str):
    names, _ = get_ordered_interlocutors(doc)

    if len(names) == 0:
        final_name = get_orig_poster_name_if_present(doc)
    else:
        final_name = names[-1]

    if not final_name:
        print(f"get_final_name: weirdness: no names found in doc\n{repr(doc)}\n")
        return ""

    return final_name


def simulate_frank_as_final_poster(doc: str):
    skip_chars = [DEFAULT_CSC['ORIG_FICTION_CHAR_FORUMLIKE'], DEFAULT_CSC['REVIEW_CHAR_FORUMLIKE'], ]
    if any([c in doc for c in skip_chars]):
        return doc

    final_name = get_final_name(doc)

    return doc.replace(final_name, DEFAULT_CSC['user_name'])


def patch_time_in_forumlike_doc(doc: str, ts: datetime=now):
    ts = timestamp_to_v10_format(ts)

    time_seg_start = DEFAULT_CSC["posted_at"].format(time_text="")

    before, mid, after = doc.rpartition(time_seg_start)
    result = before + mid

    _, mid2, after2 = after.partition(" | ")

    result += ts + mid2 + after2
    return result


def prep_for_selector(doc: str, ts: datetime=now):
    doc = simulate_frank_as_final_poster(doc)
    doc = patch_time_in_forumlike_doc(doc, ts=ts)
    return doc
