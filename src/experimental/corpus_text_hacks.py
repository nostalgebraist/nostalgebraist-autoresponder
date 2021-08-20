"""
Hacky tools for modifying tumblr post structure after posts have already been converted to formatted text.

_Wherever possible_, it's better to modify posts as NPF and convert to text at the very end.

This file exists for the specials cases where that's not possible.

TODO: one day, realize the dream of a fully invertible tumblr -> text -> tumblr converter...
"""
import re
from datetime import datetime

from tumblr_to_text.classic.autoresponder_static import DEFAULT_CSC, find_control_chars_forumlike
from tumblr_to_text.classic.autoresponder_static_v8 import (
    timestamp_to_v10_format,
    format_segment_v8_interlocutors
)
from util.tz import TZ_PST


now = datetime.now(tz=TZ_PST)  # ensures same value in long-running jobs
orig_poster_regex = DEFAULT_CSC["ORIG_POST_CHAR_NAMED"].format(user_name="([^ ]*)")


def _cut_forumlike(to_cut, n_cchars_retained, newline_postfix="\n", verbose=False):
    cchars = find_control_chars_forumlike(to_cut)
    if len(cchars) < n_cchars_retained:
        if verbose:
            print(f"not cutting: only found cchars {cchars}")
        return to_cut

    cut_ix = cchars[-n_cchars_retained][1]
    if verbose:
        print(f"cutting at {cut_ix}")
    after_cut = to_cut[cut_ix:]

    segs = []
    last_ix = 0
    for i, (cchar, ix) in enumerate(find_control_chars_forumlike(after_cut)):
        segs.append(after_cut[last_ix:ix])
        last_ix = ix + len(cchar)

        if cchar.startswith("#"):
            _, sep, after = cchar.partition(" ")
            cchar = "#" + str(i + 1) + sep + after

        if i == 0:
            cchar = cchar.replace("commented:", "posted:")

        segs.append(cchar)
    segs.append(after_cut[last_ix:len(after_cut)])

    after_re_enumerate = "".join(segs)

    interlocutor_string = format_segment_v8_interlocutors(after_re_enumerate)
    prefix = (
        " " + interlocutor_string.rstrip(" ") + " |" + newline_postfix
        if len(interlocutor_string) > 0
        else ""
    )

    return prefix + after_re_enumerate


def cut_to_final_exchange_forumlike(to_cut, verbose=False):
    return _cut_forumlike(to_cut, n_cchars_retained=2, newline_postfix="\n", verbose=verbose)


def cut_to_n_most_recent_by_user_forumlike(to_cut, n, user_name="nostalgebraist-autoresponder", verbose=False):
    cchars_reversed = find_control_chars_forumlike(to_cut)[::-1]

    n_by_user = 0
    for i, (cchar, ix) in enumerate(cchars_reversed):
        segs = cchar.split(" ")
        cchar_uname = segs[1] if cchar.startswith("#") else segs[0]

        if cchar_uname == user_name:
            n_by_user += 1

        if verbose:
            print(f"cchar {repr(cchar)} --> cchar_uname {repr(cchar_uname)}, vs {repr(user_name)}")
            print(f"n_by_user: {n_by_user}")

        if n_by_user >= n:
            break

    n_cchars_retained = i + 1  # because it starts at zero
    return _cut_forumlike(to_cut, n_cchars_retained=n_cchars_retained, newline_postfix="\n", verbose=verbose)


def get_orig_poster_name_if_present(doc: str):
    if DEFAULT_CSC["ORIG_POST_CHAR_FORUMLIKE"] in doc:
        return DEFAULT_CSC["user_name"]

    for m in re.finditer(orig_poster_regex, doc):
        return " " + m.group(1)


def get_non_deduped_ordered_interlocutors(doc, control_seg_config=DEFAULT_CSC):
    cchars = find_control_chars_forumlike(
        doc, incl_number=False, control_seg_config=control_seg_config
    )

    names = []
    for c in cchars:
        for ph in control_seg_config["numbered_phrases"]:
            if ph in c[0]:
                names.append(c[0].partition(" " + ph)[0])
    return names


def get_final_name(doc: str, verbose=False):
    names = get_non_deduped_ordered_interlocutors(doc)

    if len(names) == 0:
        final_name = get_orig_poster_name_if_present(doc)
    else:
        final_name = names[-1]

    if not final_name:
        if verbose:
            print(f"get_final_name: no names found in doc\n{repr(doc)}\n")
        return ""

    return final_name


def simulate_frank_as_final_poster(doc: str):
    skip_chars = [
        DEFAULT_CSC["ORIG_FICTION_CHAR_FORUMLIKE"],
        DEFAULT_CSC["REVIEW_CHAR_FORUMLIKE"],
    ]
    if any([c in doc for c in skip_chars]):
        return doc

    final_name = get_final_name(doc)

    return doc.replace(final_name, " " + DEFAULT_CSC["user_name"])


def split_forumlike_doc(doc: str, newline_postfix="\n"):
    special_chars = [
        DEFAULT_CSC["ORIG_FICTION_CHAR_FORUMLIKE"],
        DEFAULT_CSC["REVIEW_CHAR_FORUMLIKE"],
    ]
    for c in special_chars:
        if doc.startswith(c):
            _, before, after = doc.partition(c)
            sep, time_segment, sep2 = "", "", ""
            tag_segment, sep3, final_content = after.partition(newline_postfix)
            return before, sep, time_segment, sep2, tag_segment, sep3, final_content
    else:
        time_seg_start = DEFAULT_CSC["posted_at"].format(time_text="")

        before, sep, after = doc.rpartition(time_seg_start)

        time_segment, sep2, after2 = after.partition(" | ")

        decision_tag_segment, sep3, final_content = after2.partition(newline_postfix)

        decision_segment, sep_dec_tag, tag_segment = decision_tag_segment.rpartition(
            " | "
        )
        sentiment_segment, sep_sent_sel, select_segment = decision_segment.partition(
            ", "
        )

    # return before, sep, time_segment, sep2, tag_segment, sep3, final_content
    return (
        before,
        sep,
        time_segment,
        sep2,
        sentiment_segment,
        sep_sent_sel,
        select_segment,
        sep_dec_tag,
        tag_segment,
        sep3,
        final_content,
    )


def extract_time_from_forumlike_doc(doc: str):
    (
        before,
        sep,
        time_segment,
        sep2,
        sentiment_segment,
        sep_sent_sel,
        select_segment,
        sep_dec_tag,
        tag_segment,
        sep3,
        final_content,
    ) = split_forumlike_doc(doc)

    return datetime.strptime(time_segment, "%I %p %B %Y")


def patch_time_in_forumlike_doc(doc: str, ts: datetime = now):
    skip_chars = [
        DEFAULT_CSC["ORIG_FICTION_CHAR_FORUMLIKE"],
        DEFAULT_CSC["REVIEW_CHAR_FORUMLIKE"],
    ]
    if any([c in doc for c in skip_chars]):
        return doc

    ts = timestamp_to_v10_format(ts)

    (
        before,
        sep,
        time_segment,
        sep2,
        sentiment_segment,
        sep_sent_sel,
        select_segment,
        sep_dec_tag,
        tag_segment,
        sep3,
        final_content,
    ) = split_forumlike_doc(doc)

    return (
        before
        + sep
        + ts
        + sep2
        + sentiment_segment
        + sep_sent_sel
        + select_segment
        + sep_dec_tag
        + tag_segment
        + sep3
        + final_content
    )


def prep_for_selector(doc: str, ts: datetime = now):
    doc = simulate_frank_as_final_poster(doc)
    doc = patch_time_in_forumlike_doc(doc, ts=ts)
    doc = cut_to_final_exchange_forumlike(
        doc
    )
    return doc


def prep_for_sentiment(doc: str):
    segs = split_forumlike_doc(doc)
    return segs[-1]


def prep_for_autoreviewer(doc: str, ts: datetime = now):
    doc = simulate_frank_as_final_poster(doc)
    doc = patch_time_in_forumlike_doc(doc, ts=ts)
    doc = cut_to_n_most_recent_by_user_forumlike(
        doc, n=2
    )
    return doc
