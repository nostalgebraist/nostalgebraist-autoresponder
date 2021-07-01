from datetime import datetime

from autoresponder_static import DEFAULT_CSC
from autoresponder_static_v8 import get_ordered_interlocutors, timestamp_to_v10_format

now = datetime.now()


def simulate_frank_as_final_poster(doc: str, frank_name=" nostalgebraist-autoresponder"):
    skip_chars = [DEFAULT_CSC['ORIG_FICTION_CHAR_FORUMLIKE'], DEFAULT_CSC['REVIEW_CHAR_FORUMLIKE'], ]
    if any([c in doc for c in skip_chars]):
        return doc

    names, _ = get_ordered_interlocutors(doc)

    if len(names) == 0:
        print(f"simulate_frank_as_final_poster: weirdness: no names found in doc\n{repr(doc)}\n")

    final_name = names[-1]
    return doc.replace(final_name, frank_name)


def patch_time_in_forumlike_doc(doc: str, ts: datetime=now):
    ts = timestamp_to_v10_format(ts)

    time_seg_start = DEFAULT_CSC["posted_at"].format(time_text="")

    before, mid, after = doc.rpartition(time_seg_start)
    result = before + mid

    _, mid2, after2 = after.partition(" |")

    result += ts + mid2 + after2
    return result


def prep_for_selector(doc: str, ts: datetime=now, frank_name=" nostalgebraist-autoresponder"):
    doc = simulate_frank_as_final_poster(doc, frank_name=frank_name)
    doc = patch_time_in_forumlike_doc(doc, ts=ts)
    return doc
