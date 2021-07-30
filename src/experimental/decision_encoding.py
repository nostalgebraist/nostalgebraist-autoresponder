import hashlib

import numpy as np
from scipy.special import softmax

from experimental.corpus_text_hacks import split_forumlike_doc


SELECTOR_CCHAR = "Viral"
SENTIMENT_CCHAR = "Mood"


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

    return float(select_segment.split(" ")[-1][:-1]) / 100


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


def get_distribution_mood(doc, enc, model, transformers_model, tokenizer):
    import torch

    pminus, pplus = make_prompts_mood(doc)

    with torch.no_grad():
        inp = {k: torch.as_tensor(v).to(transformers_model.device) for k, v in tokenizer([pminus]).items()}
        out_minus = transformers_model(**inp)['logits'][0, -2:, :].cpu().numpy()

    probs = softmax(out_minus, axis=-1).astype(np.float)
    prob_of_minus_sign = probs[0, enc.encode(" -")[0]]
    prob_of_plus_sign = probs[0, enc.encode(" +")[0]]

    prob_of_minus_x = {i: probs[1, enc.encode(str(i))[0]] for i in range(10)}

    with torch.no_grad():
        inp = {k: torch.as_tensor(v).to(model.device) for k, v in tokenizer([pplus]).items()}
        out_plus = model(**inp)['logits'][0, -2:, :].cpu().numpy()

    probs = softmax(out_plus, axis=-1).astype(np.float)
    prob_of_plus_x = {i: probs[1, enc.encode(str(i))[0]] for i in range(10)}

    return prob_of_minus_sign, prob_of_plus_sign, prob_of_minus_x, prob_of_plus_x


def postprocess_distribution_mood(prob_of_minus_sign, prob_of_plus_sign, prob_of_minus_x, prob_of_plus_x):
    out = {-1 * k: prob_of_minus_sign * v for k, v in prob_of_minus_x.items()}
    out.update({k: prob_of_plus_sign * v for k, v in prob_of_plus_x.items()})
    return out


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


def get_distribution_select(doc, enc, model, transformers_model, tokenizer):
    import torch

    pr = make_prompt_select(doc)

    with torch.no_grad():
        inp = {k: torch.as_tensor(v).to(transformers_model.device) for k, v in tokenizer([pr]).items()}
        out = transformers_model(**inp)['logits'][0, -1:, :].cpu().numpy()

    probs = softmax(out, axis=-1).astype(np.float)

    prob_of_x = {i / 100: probs[0, enc.encode(" " + str(i))[0]] for i in range(0, 110, 10)}

    return prob_of_x
