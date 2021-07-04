import re
import hashlib
from datetime import datetime

import numpy as np
from scipy.special import softmax

from autoresponder_static import DEFAULT_CSC, find_control_chars_forumlike
from autoresponder_static_v8 import (
    timestamp_to_v10_format,
    cut_to_final_exchange_forumlike,
)
from util.util import chardec

now = datetime.now()
orig_poster_regex = DEFAULT_CSC["ORIG_POST_CHAR_NAMED"].format(user_name="([^ ]*)")


SELECTOR_CCHAR = "Viral"
SENTIMENT_CCHAR = "Mood"


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

        decision_segment, sep_dec_tag, tag_segment = decision_tag_segment.partition(
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
    )  # imitates selector_cut_to_final_exchange
    return doc


def prep_for_sentiment(doc: str):
    segs = split_forumlike_doc(doc)
    return segs[-1]


def unique_id_for_doc(doc: str):
    return hashlib.md5(doc.encode("utf-8")).hexdigest()


### providing the info to the model


def make_sentiment_seg(sentiment):
    return f"{SENTIMENT_CCHAR} {round(sentiment, 0):+.0f}"


def make_select_seg(select):
    return f"{SELECTOR_CCHAR} {round(select, 1):.0%}"


def inject_side_judgments(doc, sentiment=None, select=None):
    special_chars = [
        " Original fiction by nostalgebraist-autoresponder\n\n nostalgebraist-autoresponder's tags: ",
        " Book review by nostalgebraist-autoresponder\n\n nostalgebraist-autoresponder's tags:",
    ]

    sent_seg, ssmid, select_seg = "", "", ""

    if sentiment and select:
        ssmid = ", "

    if sentiment:
        sent_seg = make_sentiment_seg(sentiment)

    if select:
        select_seg = make_select_seg(select)

    for c in special_chars:
        if doc.startswith(c):
            before, sep, after = doc.partition(" nostalgebraist-autoresponder's tags:")
            return before + " " + sent_seg + ssmid + select_seg + " |" + sep + after

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
        + time_segment
        + sep2
        + sent_seg
        + ssmid
        + select_seg
        + " | "
        + tag_segment
        + sep3
        + final_content
    )

def remove_side_judgments(doc):
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
        + time_segment
        + sep2
        + tag_segment
        + sep3
        + final_content
    )


### reading info off the model
# TODO: validate for fic+review

def get_sampled_mood(doc):
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

    return float(sentiment_segment.split(" ")[-1])


def get_sampled_select(doc):
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

    return float(select_segment.split(" ")[-1][:-1])/100


def make_prompts_mood(doc):
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

    target_before, space, target_after = sentiment_segment.partition(" ")
    prefix = before + sep + time_segment + sep2 + target_before + space
    return prefix + "-", prefix + "+"


def get_distribution_mood(doc, enc, model):
    import torch

    pminus, pplus = make_prompts_mood(s)

    with torch.no_grad():
        inp = {k: torch.as_tensor(v).to(transformers_model.device) for k, v in tokenizer([pminus]).items()}
        out_minus = transformers_model(**inp)['logits'][0, -2:, :].cpu().numpy()

    probs = softmax(out_minus, axis=-1)
    prob_of_minus_sign = probs[0, enc.encode(" -")[0]]
    prob_of_plus_sign = probs[0, enc.encode(" +")[0]]

    prob_of_minus_x = {i: probs[1, enc.encode(str(i))[0]] for i in range(10)}

    with torch.no_grad():
        inp = {k: torch.as_tensor(v).to(model.device) for k, v in tokenizer([pplus]).items()}
        out_plus = model(**inp)['logits'][0, -2:, :].cpu().numpy()

    probs = softmax(out_plus, axis=-1)
    prob_of_plus_x = {i: probs[1, enc.encode(str(i))[0]] for i in range(10)}

    return prob_of_minus_sign, prob_of_plus_sign, prob_of_minus_x, prob_of_plus_x


def make_prompt_select(doc):
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

    target_before, space, target_after = select_segment.partition(" ")
    return (
        before
        + sep
        + time_segment
        + sep2
        + sentiment_segment
        + sep_sent_sel
        + target_before
    )
