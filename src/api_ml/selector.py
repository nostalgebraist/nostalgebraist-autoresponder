"""
Helper functions for selecting one "best" output from GPT-2 from a list of such outputs.
"""
from functools import partial

import numpy as np
from textwrap import wrap

from multimodal.image_analysis_static import IMAGE_URL_DELIMITER
from tumblr_to_text.image_munging import mock_up_image_generation_tags_for_heads

from config.autoresponder_config import LOGGING_FLAGS
from tumblr_to_text.classic.autoresponder_static import EOT
from feels.mood import logit_diff_to_allen_schema
from util.times import now_pst


RESULT_STACK = {}

RETENTION_CUTOFF = 0.7
ENFORCE_RETENTION_CUTOFF = True

FIC_COLDSTART = False
REVIEW_COLDSTART = False
IMAGE_COLDSTART = True
IMAGE_COLDSTART_USE_ARGMAX = True
GIF_COLDSTART = False
QUOTES_COLDSTART = True
DREAMS_COLDSTART = False

FIC_COLDSTART_DELTA = 0.05
REVIEW_COLDSTART_DELTA = 0.05
IMAGE_COLDSTART_DELTA = 0.35  # !
GIF_COLDSTART_DELTA = -0.2 -(IMAGE_COLDSTART * IMAGE_COLDSTART_DELTA)
QUOTES_COLDSTART_DELTA = -0.25
DREAMS_COLDSTART_DELTA = 0.15

# getting the capts MVP work to properly
IMAGE_COLDSTART_DELIMITER = IMAGE_URL_DELIMITER.lstrip('\n')

WARN_ABOUT_LOST_KEYS = False


def logit_diff(sentiment):
    pos_logit = sentiment["logits"][0] if sentiment is not None else 0
    neg_logit = sentiment["logits"][1] if sentiment is not None else 0
    return pos_logit - neg_logit


def pos_sent(sentiment):
    if sentiment is None:
        return 0.0
    return sentiment["prob"] if sentiment["label"] == "1" else 1.0 - sentiment["prob"]


def show_note_probas(texts, probas, continuation_sentiments=None, other_proba=None):
    if continuation_sentiments is None:
        sent_segments = ["" for _ in texts]
    else:
        sent_segments = [
            f", pos_sent {pos_sent(sent):.1%}" for sent in continuation_sentiments
        ]

    if other_proba is None:
        other_proba_segments = ["" for _ in texts]
    else:
        other_proba_segments = [
            f", other_proba {p:.1%}" if p is not None else "other_proba None"
            for p in other_proba
        ]

    for tpe, proba, sseg, opseg in zip(
        texts, probas, sent_segments, other_proba_segments
    ):
        print(f"\tpredicted prob: {proba:.1%}{opseg}{sseg}\n")
        print("\n~_~_~_~_~_\n")
        print("\n".join(wrap(tpe)))
        print("\n~_~_~_~_~_\n")


def parse_continuation(continuation: str, verbose=LOGGING_FLAGS["parse_continuation"]):
    if verbose:
        msg = "parse_continuation_nwo: "
        msg += f"parsing the following raw output:\n------------------\n{continuation}\n------------------\n"
        print(msg)

    tag_text, _, post = continuation.partition("\n")
    if post.startswith('='):
        # getting the capts MVP work to properly
        if verbose:
            print(f"prepending newline to post: {repr(post)}")
        post = '\n' + post
    post = post.partition(EOT)[0]

    tags = []
    if len(tag_text) > 0:
        tags = [s.rstrip(" ") for s in tag_text.split("#")]
        tags = [t for t in tags if len(t) > 0]

    parsed = {"post": post, "tags": tags}
    if verbose:
        print(f"parsed to:\n{parsed}")
    return parsed


def winndow_probabilities(proba, lower=0.167, upper=0.833):
    proba_ = proba.copy()
    exclusion_mask = np.zeros_like(proba, dtype=bool)

    if (proba_ > upper).any():
        exclude_upper = proba <= upper
        print(f"winnowing {exclude_upper.sum()} of {len(proba)} with p<{upper}")
        exclusion_mask[exclude_upper] = True
    elif (proba_ > lower).any():
        exclude_lower = proba <= lower
        print(f"winnowing {exclude_lower.sum()} of {len(proba)} with p<{lower}")
        exclusion_mask[exclude_lower] = True

    proba_[exclusion_mask] = 0

    return proba_


def get_continuation_sentiments(sentiment_logit_diffs):
    return [logit_diff_to_allen_schema(sld) for sld in sentiment_logit_diffs]


def sentiment_screen(
    continuations,
    sentiment_logit_diffs,
    mood,
    selection_proba=None,
    continuation_side_data=None,
    autoreview_proba=None,
):
    if selection_proba is None:
        selection_proba = [None for _ in continuations]
    if continuation_side_data is None:
        continuation_side_data = [None for _ in continuations]
    if autoreview_proba is None:
        autoreview_proba = [None for _ in continuations]

    all_continuation_sentiments = get_continuation_sentiments(sentiment_logit_diffs)

    score_fn = mood["score_fn"]
    if score_fn == "logit_diff":
        scores = np.asarray(
            [logit_diff(sentiment) for sentiment in all_continuation_sentiments]
        )
    elif score_fn == "pos_sentiment":
        scores = np.asarray(
            [pos_sent(sentiment) for sentiment in all_continuation_sentiments]
        )
    else:
        raise ValueError(f"score_fn {score_fn} not understood")

    min_allowed_score = mood["min_allowed_score"]
    max_allowed_score = mood["max_allowed_score"]

    exclusion_mask = np.zeros_like(scores, dtype=bool)

    if (scores >= min_allowed_score).any():
        exclude_lower = scores < min_allowed_score
        print(
            f"excluding {exclude_lower.sum()} of {len(scores)} with {score_fn}<{min_allowed_score:.3f}"
        )
        exclusion_mask[exclude_lower] = True
    else:
        print(
            f"couldn't find any with {score_fn}>={min_allowed_score}, highest is {scores.max():.3f}"
        )

    if (scores <= max_allowed_score).any():
        exclude_upper = scores > max_allowed_score
        print(
            f"excluding {exclude_upper.sum()} of {len(scores)} with {score_fn}>{max_allowed_score:.3f}"
        )
        exclusion_mask[exclude_upper] = True
    else:
        print(
            f"couldn't find any with {score_fn}<={max_allowed_score}, lowest is {scores.min():.3f}"
        )

    if exclusion_mask.all():
        print(f"! not excluding any because all were excluded")
        exclusion_mask = np.zeros_like(scores, dtype=bool)

    retained_continuation_sentiments = [
        sent
        for mask, sent in zip(exclusion_mask, all_continuation_sentiments)
        if not mask
    ]

    retained_continuations = [
        cont for mask, cont in zip(exclusion_mask, continuations) if not mask
    ]

    retained_selection_proba = [
        p for mask, p in zip(exclusion_mask, selection_proba) if not mask
    ]
    retained_continuation_side_data = [
        m for mask, m in zip(exclusion_mask, continuation_side_data) if not mask
    ]

    retained_autoreview_proba = [
        m for mask, m in zip(exclusion_mask, autoreview_proba) if not mask
    ]

    return (
        retained_continuations,
        retained_continuation_sentiments,
        retained_selection_proba,
        all_continuation_sentiments,
        retained_continuation_side_data,
        retained_autoreview_proba,
    )


def do_coldstart(continuations, selection_proba, substring, delta):
    selection_proba_ = []
    for c, p in zip(continuations, selection_proba):
        if substring in c:
            print(f"coldstarting [substring {repr(substring)}] {repr(c)}: {p:.6f} -> {delta + p:.6f}")
            selection_proba_.append(delta + p)
        else:
            selection_proba_.append(p)
    return selection_proba_


do_fic_coldstart = partial(
    do_coldstart, substring="#original fiction", delta=FIC_COLDSTART_DELTA
)
do_review_coldstart = partial(
    do_coldstart, substring="Author: <b>", delta=REVIEW_COLDSTART_DELTA
)
do_image_coldstart = partial(
    do_coldstart, substring=IMAGE_COLDSTART_DELIMITER, delta=IMAGE_COLDSTART_DELTA
)
do_gif_coldstart = partial(
    do_coldstart, substring="[Animated GIF]", delta=GIF_COLDSTART_DELTA
)
do_quotes_coldstart = partial(
    do_coldstart, substring="#quotes", delta=QUOTES_COLDSTART_DELTA
)
do_dreams_coldstart = partial(
    do_coldstart, substring="#dreams", delta=DREAMS_COLDSTART_DELTA
)


def do_all_coldstarts(continuations, selection_proba):
    if FIC_COLDSTART:
        selection_proba = do_fic_coldstart(continuations, selection_proba)

    if REVIEW_COLDSTART:
        selection_proba = do_review_coldstart(continuations, selection_proba)

    if IMAGE_COLDSTART:
        selection_proba = do_image_coldstart(continuations, selection_proba)

    if GIF_COLDSTART:
        selection_proba = do_gif_coldstart(continuations, selection_proba)

    if QUOTES_COLDSTART:
        selection_proba = do_quotes_coldstart(continuations, selection_proba)

    if DREAMS_COLDSTART:
        selection_proba = do_dreams_coldstart(continuations, selection_proba)

    return selection_proba


def serve_selection(
    data,
    post_type,
    mood,
    retention_stack=None,
    strategy="proportional",
    eps=0.1,
):
    continuations = data["continuations"]
    selection_proba = data.get("selection_proba")
    continuation_side_data = data.get(
        "continuation_side_data", [{} for _ in continuations]
    )

    selection_proba = do_all_coldstarts(continuations, selection_proba)

    sentiment_logit_diffs = data.get("sentiment_logit_diffs")

    autoreview_proba = data.get("autoreview_proba", [None for _ in continuations])

    if (post_type == "textpost") and (strategy != "uniform"):
        continuations += sorted(retention_stack)
        if selection_proba is not None:
            (
                retention_stack,
                retention_stack_proba,
                retention_stack_logit_diffs,
                retention_stack_autoreview_proba,
            ) = get_retention_stack_judgments(retention_stack)
            if retention_stack_proba is not None:
                print(
                    f"len(retention_stack) {len(retention_stack)} vs len(retention_stack_proba) {len(retention_stack_proba)}"
                )
                selection_proba += retention_stack_proba
                sentiment_logit_diffs += retention_stack_logit_diffs
                autoreview_proba += retention_stack_autoreview_proba
            else:
                selection_proba += [None for _ in retention_stack]
            continuation_side_data += [{} for _ in retention_stack]

    do_mood_screen = mood.get("name") != "unrestricted"

    if do_mood_screen:
        (
            retained_continuations,
            continuation_sentiments,  # TODO: clearer name here
            retained_selection_proba,
            all_continuation_sentiments,
            retained_continuation_side_data,
            retained_autoreview_proba,
        ) = sentiment_screen(
            continuations,
            sentiment_logit_diffs,
            mood,
            selection_proba,
            continuation_side_data,
            autoreview_proba,
        )
    else:
        continuation_sentiments = get_continuation_sentiments(sentiment_logit_diffs)
        retained_continuations = continuations
        all_continuation_sentiments = continuation_sentiments
        retained_selection_proba = selection_proba
        retained_continuation_side_data = continuation_side_data

    proba = np.asarray(retained_selection_proba)  # TODO: clearer name here

    # diffusion coldstart
    if IMAGE_COLDSTART_USE_ARGMAX:
        if any(IMAGE_COLDSTART_DELIMITER in c for c in retained_continuations):
            strategy = "argmax"
            print("found an image, using argmax")

    if strategy == "argmax":
        choice_ix = proba.argmax()
    elif strategy == "eps_greedy":
        roll = np.random.rand()
        if roll < eps:
            print(f"choosing randomly: roll {roll} < eps {eps}")
            choice_ix = np.random.choice(list(range(len(proba))))
        else:
            print(f"choosing greedily: roll {roll} >= eps {eps}")
            choice_ix = proba.argmax()
    elif strategy == "proportional" or strategy == "proportional_winnowed":
        if strategy == "proportional_winnowed":
            proba_winnowed = winndow_probabilities(proba)
        else:
            proba_winnowed = proba

        probs = proba_winnowed / sum(proba_winnowed)
        choice_ix = np.random.choice(list(range(len(probs))), p=probs)
    elif strategy == "uniform":
        print("choosing randomly with uniform distribution")
        choice_ix = np.random.choice(list(range(len(retained_continuations))))
    else:
        raise ValueError(f"strategy {strategy}")

    continuation = retained_continuations[choice_ix]
    chosen_proba = proba[choice_ix]
    chosen_pos_sent = pos_sent(continuation_sentiments[choice_ix])
    chosen_continuation_side_data = retained_continuation_side_data[choice_ix]
    chosen_autoreview_proba = retained_autoreview_proba[choice_ix]

    chosen_mirotarg = chosen_continuation_side_data.get("mirotarg")
    chosen_miro_traces = chosen_continuation_side_data.get("miro_traces")
    chosen_prompt_for_neural = chosen_continuation_side_data.get("prompt_for_neural")
    chosen_model_info = chosen_continuation_side_data.get("model_info")

    chosen_prompt_selector = chosen_continuation_side_data.get("prompt_selector")
    chosen_prompt_autoreviewer = chosen_continuation_side_data.get("prompt_autoreviewer")

    autorev_for_display = (
        None if chosen_autoreview_proba is None else f"{chosen_autoreview_proba:.1%}"
    )
    print(
        f"\nselecting #{choice_ix} with pred {chosen_proba:.1%}, pos_sent {chosen_pos_sent:.1%}, autorev {autorev_for_display}, mirotarg {chosen_mirotarg}:\n{continuation}"
    )

    if post_type == "textpost":
        for i, p in enumerate(selection_proba):
            if p > RETENTION_CUTOFF and continuations[i] not in retention_stack:
                retention_stack.add(continuations[i])

        if continuation in retention_stack:
            retention_stack.remove(continuation)

        retention_stack, retention_stack_logit_diffs = apply_retention_cutoff(retention_stack)

    parsed = parse_continuation(continuation)
    parsed["proba"] = float(chosen_proba)
    parsed["pos_sentiment"] = float(chosen_pos_sent)
    parsed["autoreview_proba"] = (
        None if chosen_autoreview_proba is None else float(chosen_autoreview_proba)
    )
    parsed["mirotarg"] = chosen_mirotarg
    parsed["miro_traces"] = chosen_miro_traces
    parsed["prompt_for_neural"] = chosen_prompt_for_neural
    parsed["model_info"] = chosen_model_info
    parsed["prompt_selector"] = chosen_prompt_selector
    parsed["prompt_autoreviewer"] = chosen_prompt_autoreviewer

    parsed["all_pos_sentiment"] = [
        float(pos_sent(s)) for s in all_continuation_sentiments
    ]
    parsed["all_proba"] = [float(p) for p in selection_proba]
    parsed["all_autoreview_proba"] = [
        None if p is None else float(p) for p in autoreview_proba
    ]
    parsed["all_mirotarg"] = [sd.get("mirotarg") for sd in continuation_side_data]
    for k in data.keys():
        if "alt_selection_proba" in k:
            parsed[f"all_{k}"] = [float(p) for p in data[k]]

    parsed["choice_ix"] = int(choice_ix)
    parsed["mood"] = mood
    parsed["all_continuations"] = continuations
    all_parsed = [parse_continuation(c, verbose=False) for c in continuations]
    all_posts = [p["post"] for p in all_parsed]
    all_tags = [p["tags"] for p in all_parsed]
    parsed["all_posts"] = all_posts
    parsed["all_tags"] = all_tags

    lost_keys = [k for k in data.keys() if k not in parsed]
    if WARN_ABOUT_LOST_KEYS and len(lost_keys) > 0:
        print(f"traceability warning: the following fields are not being saved")
        for k in lost_keys:
            print(f"\t{k}")
        print("consider modifying selector.py to include them")

    return parsed, retention_stack, retention_stack_logit_diffs


def get_retention_stack_judgments(retention_stack,
                                  blog_name="nostalgebraist-autoresponder",  # TODO (cleanup): improve
                                  timestamp=None
                                  ):
    from api_ml.ml_connector import (
        selection_proba_from_gpt,
        sentiment_logit_diffs_from_gpt,
        autoreview_proba_from_gpt,
    )
    from tumblr_to_text.nwo_munging import make_nwo_textpost_prompts

    if timestamp is None:
        timestamp = now_pst()

    if len(retention_stack) == 0:
        proba, logit_diffs, autoreview_proba = [], [], []
        return proba, logit_diffs, autoreview_proba

    base_texts = sorted(retention_stack)

    prompts, prompts_selector, prompts_autoreviewer, _ = make_nwo_textpost_prompts(
        blog_name=blog_name,
        timestamp=now_pst()
    )

    base_texts_for_selector_and_autoreviewer = [
        mock_up_image_generation_tags_for_heads(c, guidance_scale=2)  # fixed value for determinism
        for c in base_texts
    ]

    selector_texts = [prompts_selector[prompts[0]] + c for c in base_texts_for_selector_and_autoreviewer]
    sentiment_texts = base_texts
    autoreviewer_texts = [prompts_autoreviewer[prompts[0]] + c for c in base_texts_for_selector_and_autoreviewer]

    proba = selection_proba_from_gpt(selector_texts)

    proba = do_all_coldstarts(base_texts, proba)

    logit_diffs = sentiment_logit_diffs_from_gpt(sentiment_texts)

    autoreview_proba = autoreview_proba_from_gpt(
        autoreviewer_texts,
    )

    return base_texts, proba, logit_diffs, autoreview_proba


def apply_retention_cutoff(retention_stack):
    retention_stack, retention_stack_proba, retention_stack_logit_diffs, _ = get_retention_stack_judgments(retention_stack)

    n_before_stack, n_before_proba = len(retention_stack), len(retention_stack_proba)
    retain = [p > RETENTION_CUTOFF for p in retention_stack_proba]

    if False in retain:
        for s, p, r in zip(sorted(retention_stack), retention_stack_proba, retain):
            if not r:
                print(f"excluding, p={p:.1%}: {repr(s[:100])} [...]")
            else:
                print(f"keeping, p={p:.1%}: {repr(s[:100])} [...]")

    new_stack = [s for s, r in zip(sorted(retention_stack), retain) if r]
    new_proba = [p for p, r in zip(retention_stack_proba, retain) if r]
    new_logit_diffs = [p for p, r in zip(retention_stack_logit_diffs, retain) if r]

    retention_stack = set(new_stack)
    retention_stack_proba = new_proba
    retention_stack_logit_diffs = new_logit_diffs

    n_after_stack, n_after_proba = len(retention_stack), len(retention_stack_proba)

    all_unchanged = (n_before_stack == n_after_stack) and (
        n_before_proba == n_after_proba
    )
    if not all_unchanged:
        print(
            f"before: {n_before_stack} in retention_stack, {n_before_proba} in retention_stack_proba"
        )
        print(
            f"after: {n_after_stack} in retention_stack, {n_after_proba} in retention_stack_proba"
        )
    print(
        f"len(retention_stack) {len(retention_stack)} vs len(retention_stack_proba) {len(retention_stack_proba)}"
    )
    return retention_stack, retention_stack_logit_diffs
