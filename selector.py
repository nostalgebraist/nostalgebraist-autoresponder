"""
Runs the selector, regularly polling the bridge service and handling selection needs.

This was copied out of a jupyter notebook and hasn't been edited much since then, so
code quality is even uglier than usual :(
"""
from functools import partial

import numpy as np
import pickle
import sys
from textwrap import wrap


import requests
import time
from tqdm import tqdm

from bot_config import BotSpecificConstants
from side_judgments import SideJudgmentCache, SELECT_VIA_GENERATOR
from image_analysis import IMAGE_DELIMITER_WHITESPACED

Q_CHAR = "会"
A_CHAR = "域"
T_CHAR = "职"
ORIG_POST_CHAR = "翰"

AB_TEST_A_SEQUENCE = "\uFFFA"
AB_TEST_B_SEQUENCE = "\uFFFB"

bot_specific_constants = BotSpecificConstants.load()
selector_url = bot_specific_constants.bridge_service_url + "/pollselector"
generator_url = bot_specific_constants.bridge_service_url + "/pollgenerator"

retention_stack = {}
RESULT_STACK = {}
wrapped = None

MODEL_NAME = "bert_6_25_20_WITH_FIX"
VERIFY_MODEL = False

RETENTION_CUTOFF = 0.6
ENFORCE_RETENTION_CUTOFF = True

EOT_WORKAROUND = True
eot_end_segment = "<|endoftext|>" if EOT_WORKAROUND else "<|"

FIC_COLDSTART = True
REVIEW_COLDSTART = False
IMAGE_COLDSTART = False

FIC_COLDSTART_DELTA = 0.2  # 0.1
REVIEW_COLDSTART_DELTA = 0.1
IMAGE_COLDSTART_DELTA = 0.1

WARN_ABOUT_LOST_KEYS = False


def logit_diff(sentiment):
    pos_logit = sentiment["logits"][0] if sentiment is not None else 0
    neg_logit = sentiment["logits"][1] if sentiment is not None else 0
    return pos_logit - neg_logit


def pos_sent(sentiment):
    if sentiment is None:
        return 0.0
    return sentiment["prob"] if sentiment["label"] == "1" else 1.0 - sentiment["prob"]


def show_note_preds(texts, preds):
    for tpe, pred in zip(texts, preds):
        print(f"\tpredicted notes: {pred:.1f}\n")
        print("\n~_~_~_~_~_\n")
        print("\n".join(wrap(tpe)))
        print("\n~_~_~_~_~_\n")


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


def verify_new_model():
    global wrapped
    with open("reward/textpost_examples.pkl.gz", "rb") as f:
        textpost_examples = pickle.load(f)
    textpost_examples = [s.lstrip(ORIG_POST_CHAR) for s in textpost_examples]
    proba_tpe = wrapped.predict_proba(textpost_examples)[:, 1]
    show_note_probas(textpost_examples, proba_tpe)


def parse_continuation(continuation: str, verbose=True):
    if verbose:
        print(
            f"parsing the following raw output:\n------------------\n{continuation}\n------------------\n"
        )

    # split out tags, if present
    post, _, tag_text = continuation.partition(T_CHAR)
    tag_text = tag_text.partition(eot_end_segment)[
        0
    ]  # drop stuff after eot_end_segment
    tag_text = tag_text.partition("<|")[0]  # temporarily support old EOT format

    tags = []
    if len(tag_text) > 0:
        tags = [s.rstrip(" ") for s in tag_text.split("#")]

    # handle mistake i made in AR V6 :(
    if "#original fiction" in post:
        post_after_fic_tag = post[post.index("#original fiction") :]
        if len(post_after_fic_tag.split()) < 10:
            fic_tags = [s.rstrip(" ") for s in post_after_fic_tag.split("#")]
            print(f"converting {post_after_fic_tag} to {fic_tags}")
            tags = fic_tags + tags
            post = post[: post.index("#original fiction")]

    post = post.lstrip(
        ORIG_POST_CHAR
    )  # TODO: fix this in get_prompted_continuation_with_length_proportional_sampling
    parsed = {"post": post, "tags": tags}
    return parsed


def winndow_probabilities(proba, lower=0.333, upper=0.667):
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


def get_continuation_sentiments(side_judgment_cache, continuations, sleep_time=0.2):
    continuation_sentiments = [
        side_judgment_cache.query(c, sleep_time=sleep_time)["sentiment"]["allen_schema"]
        for c in tqdm(continuations)
    ]
    return continuation_sentiments


def sentiment_screen(
    side_judgment_cache, continuations, mood, selection_proba=None, mirotarg=None
):
    if selection_proba is None:
        selection_proba = [None for _ in continuations]
    if mirotarg is None:
        mirotarg = [None for _ in continuations]

    all_continuation_sentiments = get_continuation_sentiments(
        side_judgment_cache, continuations
    )

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

    print(f"{score_fn}: {scores}\n")

    exclusion_mask = np.zeros_like(scores, dtype=bool)

    if (scores >= min_allowed_score).any():
        exclude_lower = scores < min_allowed_score
        print(
            f"excluding {exclude_lower.sum()} of {len(scores)} with {score_fn}<{min_allowed_score}"
        )
        exclusion_mask[exclude_lower] = True
    else:
        print(
            f"couldn't find any with {score_fn}>={min_allowed_score}, highest is {scores.max()}"
        )

    if (scores <= max_allowed_score).any():
        exclude_upper = scores > max_allowed_score
        print(
            f"excluding {exclude_upper.sum()} of {len(scores)} with {score_fn}>{max_allowed_score}"
        )
        exclusion_mask[exclude_upper] = True
    else:
        print(
            f"couldn't find any with {score_fn}<={max_allowed_score}, lowest is {scores.min()}"
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
    retained_mirotarg = [m for mask, m in zip(exclusion_mask, mirotarg) if not mask]

    return (
        retained_continuations,
        retained_continuation_sentiments,
        retained_selection_proba,
        all_continuation_sentiments,
        retained_mirotarg,
    )


def sentiment_screen_legacy(proba, continuations, mood):
    continuation_sentiments = get_continuation_sentiments(continuations)

    score_fn = mood["score_fn"]
    if score_fn == "logit_diff":
        scores = np.asarray(
            [logit_diff(sentiment) for sentiment in continuation_sentiments]
        )
    elif score_fn == "pos_sentiment":
        scores = np.asarray(
            [pos_sent(sentiment) for sentiment in continuation_sentiments]
        )
    else:
        raise ValueError(f"score_fn {score_fn} not understood")

    proba_ = proba.copy()
    exclusion_mask = np.zeros_like(proba, dtype=bool)

    min_allowed_score = mood["min_allowed_score"]
    max_allowed_score = mood["max_allowed_score"]

    print(f"proba: {proba}\nscores: {scores}\n")

    if (scores >= min_allowed_score).any():
        exclude_lower = scores < min_allowed_score
        print(
            f"excluding {exclude_lower.sum()} of {len(proba)} with {score_fn}<{min_allowed_score}"
        )
        exclusion_mask[exclude_lower] = True
    else:
        print(
            f"couldn't find any with {score_fn}>={min_allowed_score}, highest is {scores.max()}"
        )

    if (scores <= max_allowed_score).any():
        exclude_upper = scores > max_allowed_score
        print(
            f"excluding {exclude_upper.sum()} of {len(proba)} with {score_fn}>{max_allowed_score}"
        )
        exclusion_mask[exclude_upper] = True
    else:
        print(
            f"couldn't find any with {score_fn}<={max_allowed_score}, lowest is {scores.min()}"
        )

    proba_[exclusion_mask] = 0

    return proba_, continuation_sentiments


def record_side_judgements(
    side_judgment_cache, continuations, selection_proba, sentiment_logit_diffs
):
    for c, sp, sld in zip(continuations, selection_proba, sentiment_logit_diffs):
        side_judgment_cache.record(
            c, {"selection_proba": [sp], "sentiment_logit_diffs": [sld]}
        )


def serve_selection(
    data, side_judgment_cache, retention_stack=None, retention_stack_proba=None
):
    global wrapped

    continuations = data["continuations"]
    selection_proba = data.get("selection_proba")
    mirotarg = data.get("mirotarg", [None for _ in continuations])

    if FIC_COLDSTART:
        selection_proba = do_fic_coldstart(continuations, selection_proba)

    if REVIEW_COLDSTART:
        selection_proba = do_review_coldstart(continuations, selection_proba)

    if IMAGE_COLDSTART:
        selection_proba = do_image_coldstart(continuations, selection_proba)

    sentiment_logit_diffs = data.get("sentiment_logit_diffs")
    if selection_proba is not None:
        print(
            f"len(selection_proba): {len(selection_proba)} vs len(continuations): {len(continuations)}"
        )
    else:
        print("selection_proba is None")

    if selection_proba is not None and sentiment_logit_diffs is not None:
        record_side_judgements(
            side_judgment_cache, continuations, selection_proba, sentiment_logit_diffs
        )

    kwargs = data["kwargs"]
    mood = kwargs.get("mood")
    return_all_conts = kwargs.get("return_all_conts", False)

    strategy = "proportional"
    if "strategy" in kwargs:
        strategy = kwargs["strategy"]
    eps = 0.1
    if "eps" in kwargs:
        eps = kwargs["eps"]

    if (data["type"] == "textpost") and (strategy != "uniform"):
        continuations += sorted(retention_stack)
        if selection_proba is not None:
            if retention_stack_proba is not None:
                selection_proba += retention_stack_proba
            else:
                selection_proba += [None for _ in retention_stack]
            # TODO: store retention_stack mirotarg
            mirotarg += [None for _ in retention_stack]

    base_id = data["base_id"]

    do_mood_screen = False
    if mood is not None:
        do_mood_screen = mood.get("name") != "unrestricted"

    if do_mood_screen:
        (
            continuations_screened,
            continuation_sentiments,
            selection_proba_screened,
            all_continuation_sentiments,
            retained_mirotarg,
        ) = sentiment_screen(
            side_judgment_cache, continuations, mood, selection_proba, mirotarg
        )
    else:
        continuation_sentiments = get_continuation_sentiments(
            side_judgment_cache, continuations
        )
        continuations_screened = continuations
        all_continuation_sentiments = continuation_sentiments
        selection_proba_screened = selection_proba
        retained_mirotarg = mirotarg

    if SELECT_VIA_GENERATOR:
        proba = np.asarray(selection_proba_screened)
        show_note_probas(continuations_screened, proba, continuation_sentiments)
    else:
        proba = wrapped.predict_proba([s.lstrip("翰") for s in continuations_screened])[
            :, 1
        ]
        show_note_probas(
            continuations_screened,
            proba,
            continuation_sentiments,
            other_proba=selection_proba_screened,
        )

    if strategy == "argmax":
        choice_ix = proba.argmax()
    elif strategy == "eps_greedy":
        print(f"choosing between preds {proba}\n")
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
        print(f"choosing between preds {proba_winnowed}\nprobs {probs}")
        choice_ix = np.random.choice(list(range(len(probs))), p=probs)
    elif strategy == "uniform":
        print("choosing randomly with uniform distribution")
        choice_ix = np.random.choice(list(range(len(continuations_screened))))
    else:
        raise ValueError(f"strategy {strategy}")

    continuation = continuations_screened[choice_ix]
    chosen_proba = proba[choice_ix]
    chosen_pos_sent = pos_sent(continuation_sentiments[choice_ix])
    chosen_mirotarg = retained_mirotarg[choice_ix]
    print(
        f"\nselecting #{choice_ix} with pred {chosen_proba:.1%}, pos_sent {chosen_pos_sent:.1%}:\n{continuation}, mirotarg {chosen_mirotarg}\n"
    )

    if data["type"] == "textpost":
        for i, p in enumerate(selection_proba):
            if p > RETENTION_CUTOFF and continuations[i] not in retention_stack:
                retention_stack.add(continuations[i])

        if continuation in retention_stack:
            retention_stack.remove(continuation)

    parsed = parse_continuation(continuation)
    parsed["proba"] = float(chosen_proba)
    parsed["pos_sentiment"] = float(chosen_pos_sent)
    parsed["mirotarg"] = chosen_mirotarg
    parsed["all_pos_sentiment"] = [
        float(pos_sent(s)) for s in all_continuation_sentiments
    ]
    parsed["all_proba"] = [float(p) for p in selection_proba]
    parsed["all_mirotarg"] = mirotarg
    for k in data.keys():
        if "alt_selection_proba" in k:
            parsed[f"all_{k}"] = [float(p) for p in data[k]]

    parsed["choice_ix"] = int(choice_ix)
    parsed["mood"] = mood
    if return_all_conts:
        all_parsed = [parse_continuation(c) for c in continuations]
        all_posts = [p["post"] for p in all_parsed]
        all_tags = [p["tags"] for p in all_parsed]
        parsed["all_posts"] = all_posts
        parsed["all_tags"] = all_tags

    parsed["base_id"] = base_id

    if "AB_fork" in kwargs:
        fork = kwargs["AB_fork"]
        parsed["AB_fork"] = fork

        post = parsed["post"]
        non_newline_ixs = [ix for ix, c in enumerate(post) if c != "\n"]
        if len(non_newline_ixs) > 0:
            newline_switch_ix = max(non_newline_ixs) + 1
            post = post[:newline_switch_ix]
    else:
        print(f"not AB testing, have kwargs {kwargs}")

    if "model_info" in data:
        parsed["model_info"] = data["model_info"]

    if "prompt_for_neural" in data:
        parsed["prompt_for_neural"] = data["prompt_for_neural"]

    print(f"sending back: {parsed}")

    lost_keys = [k for k in data.keys() if k not in parsed]
    if WARN_ABOUT_LOST_KEYS and len(lost_keys) > 0:
        print(f"traceability warning: the following fields are not being saved")
        for k in lost_keys:
            print(f"\t{k}")
        print("consider modifying selector.py to include them")

    return parsed, retention_stack, retention_stack_proba


def select_one(data):
    global wrapped
    texts = data["texts"]

    proba = wrapped.predict_proba([s.lstrip("翰") for s in texts])[:, 1]

    selection_proba = [float(p) for p in proba]
    results = {"selection_proba": selection_proba}

    print(f"sending back: {results}")
    return results


def do_coldstart(continuations, selection_proba, substring, delta):
    selection_proba_ = []
    for c, p in zip(continuations, selection_proba):
        if substring in c:
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
    do_coldstart, substring=IMAGE_DELIMITER_WHITESPACED, delta=IMAGE_COLDSTART_DELTA
)


def apply_retention_cutoff(retention_stack, retention_stack_proba):
    if FIC_COLDSTART:
        retention_stack_proba = do_fic_coldstart(
            sorted(retention_stack),
            retention_stack_proba,
        )

    if REVIEW_COLDSTART:
        retention_stack_proba = do_review_coldstart(
            sorted(retention_stack),
            retention_stack_proba,
        )

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

    retention_stack = set(new_stack)
    retention_stack_proba = new_proba

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
    return retention_stack, retention_stack_proba


def poll():
    global RESULT_STACK
    global retention_stack
    global retention_stack_proba

    r = requests.post(
        selector_url,
        json={
            "results": RESULT_STACK,
            "retention_stack": sorted(retention_stack),
        },
    )

    received_data = r.json()
    PROMPT_STACK = received_data["SELECTION_PROMPT_STACK"]
    retention_stack_proba = received_data["retention_stack_proba"]

    if ENFORCE_RETENTION_CUTOFF and retention_stack_proba is not None:
        apply_retention_cutoff()

    RESULT_STACK = {
        k: v for k, v in RESULT_STACK.items() if k in PROMPT_STACK
    }  # clean out already used results

    if len(PROMPT_STACK) > 0:
        print(f"got prompt stack: {PROMPT_STACK}")

    for prompt_id, data in PROMPT_STACK.items():
        print("selecting...")
        if data.get("raw_selection_request", False):
            RESULT_STACK[prompt_id] = select_one(data)
        else:
            RESULT_STACK[prompt_id] = serve_selection(data)

    if len(RESULT_STACK) > 0:
        requests.post(
            selector_url,
            json={"results": RESULT_STACK, "n_retention": len(retention_stack)},
        )

    if len(PROMPT_STACK) > 0 and not data.get("raw_selection_request", False):
        r = requests.post(generator_url, json={"results": RESULT_STACK})
        time.sleep(1)


def loop_poll(period=60):
    while True:
        try:
            poll()
        except Exception as e:
            print(f"{type(e)}: {e}")
            time.sleep(period * 10)
        time.sleep(period)


def selector_main_loop():
    global RESULT_STACK
    global retention_stack

    load_retention()
    if not SELECT_VIA_GENERATOR:
        raise ValueError("SELECT_VIA_GENERATOR=False is no longer implemented")

    loop_poll(period=5)


if __name__ == "__main__":
    sys.exit(selector_main_loop())
