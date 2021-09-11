import time
import pickle
from textwrap import wrap, fill
from string import whitespace, punctuation
from itertools import chain, product
from typing import List

import requests
import numpy as np
import pandas as pd

from config.autoresponder_config import *
from tumblr_to_text.classic.autoresponder_static_v8 import *

from api_ml.bridge_shared import bridge_service_unique_id
from feels.mood import get_mood_by_name, load_logit_diff_sample, estimate_expected_rejections, logit_diff_to_pos_sent
from api_ml.selector import serve_selection

from api_ml import bridge_cache_singleton
from api_ml.bridge_shared import get_bridge_service_url

from util.error_handling import LogExceptionAndSkip

from smart_open import open

TRADE_QUALITY_FOR_SPEED = True

logit_diff_sample_series = load_logit_diff_sample()
EXPECTED_REJECTION_MULT = 0.5 if (not TRADE_QUALITY_FOR_SPEED) else 0.4

TEXTPOST_N_CANDIDATES_TARGET = 10 if (not TRADE_QUALITY_FOR_SPEED) else 7

# TODO: set DEFAULT_CSC using autoresponder_config constants
CONTROL_SEG_CONFIG = DEFAULT_CSC

ORIG_POST_CHAR = CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"]

SAYS_FRANK_STRINGS = {
    prefix + "Frank" + suffix
    for prefix, suffix in chain(
        product(whitespace, punctuation),
        product(whitespace, whitespace),
    )
}


class MLModelInterface:
    name: str
    uses_bridge_cache: bool

    def __init__(self):
        raise NotImplementedError

    def do(self, method, *args, repeat_until_done_signal=False, uses_bridge_cache=False, **kwargs):
        bridge_service_url = get_bridge_service_url()
        data = {
            "model": self.name,
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "repeat_until_done_signal": repeat_until_done_signal,
        }
        if uses_bridge_cache or self.uses_bridge_cache:
            response = bridge_cache_singleton.BRIDGE_CACHE.query(data)
            bridge_cache_singleton.BRIDGE_CACHE.save()
            return response

        new_id = bridge_service_unique_id(bridge_service_url, data)

        data_to_send = dict()
        data_to_send.update(data)
        data_to_send["id"] = new_id

        requests.post(bridge_service_url + "/requestml", json=data_to_send)

        return new_id


class GeneratorModelInterface(MLModelInterface):
    def __init__(self):
        self.name = "generator"
        self.uses_bridge_cache = False

    def write(self, *args, repeat_until_done_signal=False, **kwargs):
        return self.do(
            "write", repeat_until_done_signal=repeat_until_done_signal, *args, **kwargs
        )

    def write_random_prompt(self, *args, repeat_until_done_signal=False, **kwargs):
        return self.do(
            "write_random_prompt",
            repeat_until_done_signal=repeat_until_done_signal,
            *args,
            **kwargs,
        )

    def get_prob_delta_over_ref_multi(self, *args, repeat_until_done_signal=False, **kwargs):
        return self.do(
            "get_prob_delta_over_ref_multi",
            repeat_until_done_signal=repeat_until_done_signal,
            uses_bridge_cache=True,
            *args,
            **kwargs,
        )


class SideJudgmentModelInterface(MLModelInterface):
    def __init__(self, name):
        self.name = name
        self.uses_bridge_cache = True

    def predict_proba(self, *args, repeat_until_done_signal=False, **kwargs):
        return self.do(
            "predict_proba",
            repeat_until_done_signal=repeat_until_done_signal,
            *args,
            **kwargs,
        )

    def _predict(self, *args, repeat_until_done_signal=False, **kwargs):
        return self.do(
            "_predict",
            repeat_until_done_signal=repeat_until_done_signal,
            *args,
            **kwargs,
        )


generator_model = GeneratorModelInterface()
selector_est = SideJudgmentModelInterface("selector")
sentiment_est = SideJudgmentModelInterface("sentiment")
autoreviewer_est = SideJudgmentModelInterface("autoreviewer")


def parse_continuation(continuation: str, verbose=True):
    if verbose:
        print(
            f"parsing the following raw output:\n------------------\n{fill(continuation)}\n------------------\n"
        )

    # split out tags, if present
    if V8:
        post, _, tag_text = continuation.partition("\n")
    else:
        post, _, tag_text = continuation.partition(T_CHAR)
    tags = []
    if len(tag_text) > 0:
        tags = [s.rstrip(" ") for s in tag_text.split("#")]

    post = post.lstrip(ORIG_POST_CHAR)
    parsed = {"post": post, "tags": tags}
    return parsed


def get_textpost_prompts():
    prompts = []
    overrides = []
    probs = []

    prompts.append(CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"])
    overrides.append({"v10_timestamp": ""})
    probs.append(FORUMLIKE_REVIEW_PROB)

    prompts.append(CONTROL_SEG_CONFIG["ORIG_FICTION_CHAR_FORUMLIKE"])
    overrides.append(
        {"v10_timestamp": "", "tag_string_raw": "#original fiction"}
    )
    probs.append(FORUMLIKE_FIC_PROB)

    prompts.append(CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"])
    overrides.append({})
    probs.append(1 - FORUMLIKE_FIC_PROB - FORUMLIKE_REVIEW_PROB)

    return prompts, overrides, probs


profane_substrings = {
    "shit",
    "fuck",
    "sex",
    "crap",
    "hell",
    "damn",
    "vagina",
    "penis",
    "genital",
    "piss",
    "gay",
}


def basic_n_continuations(
    prompt,
    N,
    random_prompts=None,
    random_prompts_probs=None,
    avoid_if_under=20,
    avoid_half_if_under=40,
    avoid_initial_blockquote=False,
    avoid_if_profane=False,
    avoid_if_says_frank=False,
    mirotarg=None,
    verbose=False,
):
    bridge_service_url = get_bridge_service_url()

    continuation_side_data = []

    if random_prompts is not None and random_prompts_probs is not None:
        # nwo random prompting
        print(f"using prompts:")
        for _p in random_prompts:
            print(repr(_p))
        bridge_id = generator_model.write_random_prompt(
            random_prompts,
            random_prompts_probs,
            repeat_until_done_signal=True,
            verbose=verbose,
            mirotarg=mirotarg,
        )
    else:
        print(f"neural model will see:\n\n{repr(prompt)}")

        bridge_id = generator_model.write(
            prompt,
            repeat_until_done_signal=True,
            verbose=verbose,
            mirotarg=mirotarg,
        )

    all_prompts = []
    continuations = []
    n_batches_so_far = 0

    while len(continuations) < N:
        time.sleep(5)
        batches_written_raw = requests.post(
            bridge_service_url + "/getresult", data={"id": bridge_id}
        ).json()

        batches_written = [entry["result"] for entry in batches_written_raw]
        model_infos = [entry["model_info"] for entry in batches_written_raw]

        if len(batches_written) <= n_batches_so_far:
            continue

        this_batch_continuations = [
            entry
            for batch in batches_written[n_batches_so_far:]
            for entry in batch["continuations"]
        ]

        this_batch_side_data = [
            batch["side_data"]
            for batch in batches_written[n_batches_so_far:]
            for _ in batch["continuations"]
        ]

        this_batch_model_info = [
            minfo
            for batch, minfo in zip(
                batches_written[n_batches_so_far:], model_infos[n_batches_so_far:]
            )
            for _ in batch["continuations"]
        ]

        n_batches_so_far = len(batches_written)

        def _tabfill(s, ntab=2, escape=True, **kwargs):
            kwargs_ = {"width": 80}
            kwargs_.update(kwargs)

            tabs = ntab * "\t"
            sep = "\n" + tabs
            c_ = s.encode("unicode_escape").decode() if escape else s

            return sep + sep.join(wrap(c_, **kwargs_))

        for c, sdata, minfo in zip(
            this_batch_continuations, this_batch_side_data, this_batch_model_info
        ):
            pr = sdata["prompt_for_neural"]

            if contains_control_chars(c, control_seg_config=CONTROL_SEG_CONFIG):
                _cchar, min_ix = first_control_char(
                    c, control_seg_config=CONTROL_SEG_CONFIG
                )
                csub = c[:min_ix]
                print(f"\n\tsplitting on control char {repr(c[min_ix:min_ix+len(_cchar)])}:")
                print(
                    f"\n\t\t{len(c)} chars, {len(c.split(' '))} words-->\n\t{len(csub)} chars, {len(csub.split(' '))} words\n"
                )
                if len(c) < 1000:
                    print(f"was originally: {repr(c)}")
                c = csub

            roll = np.random.rand()
            # NOTE: the < 100 check is for weird posts where the newline doesn't happen
            if len(c.partition("\n")[2].split(" ")) < avoid_if_under and len(c) < 100:
                print(
                    f"\n\trejecting because length under {avoid_if_under}: {_tabfill(c)}\n"
                )
            elif (
                len(c.partition("\n")[2].split(" ")) < avoid_half_if_under
            ) and roll < 0.5:
                print(
                    f"\n\trejecting because length under {avoid_half_if_under} and roll {roll}: {_tabfill(c)}\n"
                )
            elif (
                c.partition("\n")[2].lstrip(" \n").startswith("<blockquote")
            ) and avoid_initial_blockquote:
                print(f"\n\trejecting because initial blockquote: {_tabfill(c)}\n")
            elif len([char for char in c if char == T_CHAR]) >= 2:
                print(f"\n\trejecting because multiple T_CHAR: {_tabfill(c)}\n")
            elif (
                any([subs in c.lower() for subs in profane_substrings])
                and avoid_if_profane
            ):
                print(f"\n\trejecting because profane: {_tabfill(c)}\n")
            elif avoid_if_says_frank and any([fs in c for fs in SAYS_FRANK_STRINGS]):
                print(f"\n\trejecting because says 'Frank': {_tabfill(c)}\n")
            elif normalize_for_generator(
                c.partition(T_CHAR)[0].strip(whitespace)
            ) in normalize_for_generator(pr):
                print(f"\n\trejecting because repeating myself: {_tabfill(c)}\n")
            else:
                if len(c.partition("\n")[2].split(" ")) < avoid_half_if_under:
                    print(
                        f"\n\tkeeping with roll {roll}, although length under {avoid_half_if_under}\n"
                    )
                continuations.append(c)
                sdata_plus_minfo = {k: v for k, v in sdata.items()}
                sdata_plus_minfo["model_info"] = minfo
                continuation_side_data.append(sdata_plus_minfo)
                all_prompts.append(pr)

        if len(this_batch_continuations) > 0:
            print(f"have {len(continuations)} of {N}... ", end="", flush=True)

    requests.post(bridge_service_url + "/done", json={"id": bridge_id})

    return continuations, continuation_side_data


def show_note_probas(texts, probas, sentiment_logit_diffs=None, console_width=110):
    if sentiment_logit_diffs is None:
        for tpe, proba in zip(texts, probas):
            print(f"\tpredicted prob: {proba:.1%}\n")
            print("\n~_~_~_~_~_\n")
            print("\n".join(wrap(tpe, replace_whitespace=False, width=console_width)))
            print("\n~_~_~_~_~_\n")
    else:
        for tpe, proba, ld in zip(texts, probas, sentiment_logit_diffs):
            print(
                f"\tpredicted prob: {proba:.1%}, sentiment_logit_diff {ld:.4f}, pos_sent {logit_diff_to_pos_sent(ld):.1%}\n"
            )
            print("\n~_~_~_~_~_\n")
            print("\n".join(wrap(tpe, replace_whitespace=False, width=console_width)))
            print("\n~_~_~_~_~_\n")


def predict_select(data, verbose=True):
    t1 = time.time()

    if len(data) == 0:
        # this can happen if the retention_stack is big enough
        return np.array([])

    selector_input = []
    for text in data.selector_input:
        if text.endswith(EOT):
            text = text[: -len(EOT)]
        if T_CHAR not in text and (not V8):
            text = text + T_CHAR

        if not text.startswith(EOT):
            text = EOT + text

        selector_input.append(text)

    if LOGGING_FLAGS["side_judg_inputs"]:
        print(f"predict_select innards: model will see\n{repr(selector_input)}")

    data.loc[:, "selector_input"] = selector_input

    data = data.to_dict(orient="records")

    response_data = selector_est.predict_proba(data)

    result = np.array(response_data[0]["result"])
    probs = result[:, 1]

    delta_t = time.time() - t1
    if verbose:
        print(f'predict_select: served in {delta_t:.1f}s')

    return probs


def predict_sentiment(data, verbose=True):
    t1 = time.time()

    data["prompt_finalchar"] = ["" for _ in data.selector_input.values]
    if len(data) == 0:
        # this can happen if the retention_stack is big enough
        return np.array([])

    selector_input = []
    for text in data.selector_input:
        if text.endswith(EOT):
            text = text[: -len(EOT)]
        if T_CHAR not in text:
            text = text + T_CHAR
        text = text.partition(T_CHAR)[0]
        text = normalize_for_generator(text)
        text = re.sub(r"\<.*?\>", "", text)  # sentiment-specific

        if not text.startswith(EOT):
            text = EOT + text

        selector_input.append(text)

    if LOGGING_FLAGS["side_judg_inputs"]:
        print(f"predict_sentiment innards: model will see\n{repr(selector_input)}")

    data.loc[:, "selector_input"] = selector_input

    data = data.to_dict(orient="records")

    response_data = sentiment_est._predict(data, key="logits")

    logits = np.array(response_data[0]["result"])

    logit_diffs = logits[:, 1:] - logits[:, :1]

    delta_t = time.time() - t1
    if verbose:
        print(f'predict_sentiment: served in {delta_t:.1f}s')

    return logit_diffs


def predict_autoreview(data, verbose=True):
    t1 = time.time()

    selector_input = []
    for text in data.selector_input:
        if text.endswith(EOT):
            text = text[: -len(EOT)]

        if not text.startswith(EOT):
            text = EOT + text

        selector_input.append(text)

    if LOGGING_FLAGS["side_judg_inputs"]:
        print(f"predict_autoreview innards: model will see\n{repr(selector_input)}")

    data.loc[:, "selector_input"] = selector_input

    data = data.to_dict(orient="records")

    response_data = autoreviewer_est.predict_proba(data)

    result = np.array(response_data[0]["result"])
    probs = result[:, 1]

    delta_t = time.time() - t1
    if verbose:
        print(f'predict_autoreview: served in {delta_t:.1f}s')

    return probs


def save_retention(retention_stack):
    with open("gs://nost-trc/nbar_data/retention_stack.pkl", "wb") as f:
        pickle.dump(retention_stack, f)


def adjust_best_of(best_of, mood):
    expected_rejection_frac = estimate_expected_rejections(
        min_logit_diff=mood["min_allowed_score"],
        max_logit_diff=mood["max_allowed_score"],
        logit_diff_sample_series=logit_diff_sample_series,
    )

    raw_extra_best_of = (
        int(np.round(best_of / (1 - expected_rejection_frac)))
        - best_of
    )
    discounted_extra_best_of = int(
        np.round(raw_extra_best_of * EXPECTED_REJECTION_MULT)
    )

    print(
        f"expecting to reject {expected_rejection_frac:.1%}, need {raw_extra_best_of} extra over best_of={best_of}"
    )
    best_of += discounted_extra_best_of
    print(f"discounting to {discounted_extra_best_of} --> best_of={best_of}")
    return best_of


def answer_from_gpt(
        prompt,
        prompt_selector,
        prompt_autoreviewer,
        asking_name="",
        mood_name=None,
        write_fic_override=False,
        write_review_override=False,
        avoid_initial_blockquote=False,
):
    t1 = time.time()

    mood = get_mood_by_name(mood_name)

    result_generator = old_bridge_call__answer(
        prompt=prompt,
        prompt_selector=prompt_selector,
        prompt_autoreviewer=prompt_autoreviewer,
        asking_name=asking_name,
        mood=mood,
        write_fic_override=write_fic_override,
        write_review_override=write_review_override,
        avoid_initial_blockquote=avoid_initial_blockquote
    )

    # strategy = "proportional_winnowed"
    strategy = "eps_greedy"
    eps = 0.15

    result, _ = serve_selection(
        data=result_generator,
        post_type="answer",
        mood=mood,
        strategy=strategy,
        eps=eps
    )

    # for logging, add input fields that didn't make the round trip
    result["question"] = prompt  # TODO: (cleanup): rename if safe
    result["asking_name"] = asking_name
    result["mood"] = mood_name

    delta_t = time.time() - t1
    log_prefix = ""
    if write_fic_override:
        log_prefix += " (story)"
    print(f'answer_from_gpt{log_prefix}: served in {delta_t:.1f}s')

    return result


def old_bridge_call__answer(
        prompt,
        prompt_selector,
        prompt_autoreviewer,
        asking_name="",
        mood=None,
        write_fic_override=False,
        write_review_override=False,
        avoid_initial_blockquote=False
):
    best_of = 11 if (not TRADE_QUALITY_FOR_SPEED) else 8

    if write_fic_override or write_review_override:
        best_of = 6 if not (TRADE_QUALITY_FOR_SPEED) else 4

    best_of = adjust_best_of(best_of, mood)

    avoid_if_under = 15
    if write_fic_override:
        avoid_if_under = 75
    avoid_half_if_under = 15
    avoid_if_profane = False
    avoid_if_says_frank = False
    random_year_for_generator = True
    if asking_name == "bukbot":
        avoid_if_profane = True

    # old serve_answer
    print("\n------------\n")
    prompt = prompt.rstrip(whitespace)

    print(f"write_fic_override: {write_fic_override}")

    continuations, continuation_side_data = basic_n_continuations(
        prompt,
        N=best_of,
        avoid_if_under=avoid_if_under,
        avoid_half_if_under=avoid_half_if_under,
        avoid_initial_blockquote=avoid_initial_blockquote,
        avoid_if_profane=avoid_if_profane,
        avoid_if_says_frank=avoid_if_says_frank,

    )

    response_data = {}
    response_data["continuations"] = continuations

    for c, sdata in zip(continuations, continuation_side_data):
        sdata["prompt_selector"] = prompt_selector
        sdata["prompt_autoreviewer"] = prompt_autoreviewer

    response_data["continuation_side_data"] = continuation_side_data

    # selector

    selector_inputs = [prompt_selector + c for c in continuations]

    selector_inputs = pd.DataFrame(
        {
            "selector_input": selector_inputs,
            "prompt_finalchar": [f"{A_CHAR}a" for _ in range(len(selector_inputs))],
        }
    )

    selection_results = predict_select(
        selector_inputs,
    )
    response_data["selection_proba"] = [float(p) for p in selection_results]

    # sentiment

    sentiment_inputs = pd.DataFrame({"selector_input": response_data["continuations"]})
    sentiment_results = predict_sentiment(sentiment_inputs)
    response_data["sentiment_logit_diffs"] = [float(p) for p in sentiment_results]

    # autoreview

    autoreview_inputs = [prompt_autoreviewer + c for c in continuations]

    autoreview_inputs = pd.DataFrame(
        {
            "selector_input": autoreview_inputs,
            "prompt_finalchar": ["" for _ in range(len(autoreview_inputs))],
        }
    )
    autoreview_results = predict_autoreview(
        autoreview_inputs,
    )
    response_data["autoreview_proba"] = [float(p) for p in autoreview_results]

    return response_data


def text_post_from_gpt(loop_persistent_data,
                       prompts,
                       prompts_selector,
                       prompts_autoreviewer,
                       prompts_probs,
                       mood_name=None,
                       ):
    t1 = time.time()

    mood = get_mood_by_name(mood_name)

    n_retention = len(loop_persistent_data.retention_stack)

    result_generator = old_bridge_call__textpost(n_retention=n_retention,
                                                 mood=mood,
                                                 prompts=prompts,
                                                 prompts_selector=prompts_selector,
                                                 prompts_autoreviewer=prompts_autoreviewer,
                                                 prompts_probs=prompts_probs
                                                 )

    # strategy = "proportional_winnowed"
    strategy = "eps_greedy"
    eps = 0.15

    result, retention_stack = serve_selection(
        data=result_generator,
        post_type="textpost",
        mood=mood,
        strategy=strategy,
        eps=eps,
        retention_stack=loop_persistent_data.retention_stack,
    )

    save_retention(retention_stack)

    loop_persistent_data.retention_stack = retention_stack

    # for logging, add input fields that didn't make the round trip
    result["mood"] = mood_name

    delta_t = time.time() - t1
    print(f'text_post_from_gpt: served in {delta_t:.1f}s')

    return result, loop_persistent_data


def old_bridge_call__textpost(
        n_retention,
        prompts,
        prompts_selector,
        prompts_autoreviewer,
        prompts_probs,
        mood=None,
):
    avoid_if_under = 15
    avoid_half_if_under = 15
    avoid_initial_blockquote = False
    avoid_if_says_frank = False

    best_of = TEXTPOST_N_CANDIDATES_TARGET
    best_of = adjust_best_of(best_of, mood)

    if n_retention is not None:
        best_of = max(1, best_of - n_retention)
        print(f"with {n_retention} on stack, only need {best_of}")

    print(f"n_retention {n_retention}")

    # old serve_textpost

    continuations, continuation_side_data = basic_n_continuations(
        prompt="",
        random_prompts=prompts,
        random_prompts_probs=prompts_probs,
        N=best_of,
        avoid_if_under=avoid_if_under,
        avoid_half_if_under=avoid_half_if_under,
        avoid_initial_blockquote=avoid_initial_blockquote,
        avoid_if_says_frank=avoid_if_says_frank,
    )

    response_data = {}
    response_data["continuations"] = continuations

    for c, sdata in zip(continuations, continuation_side_data):
        prompt_selector_for_c = prompts_selector[sdata["prompt_for_neural"]]
        sdata["prompt_selector"] = prompt_selector_for_c

        prompt_autoreviewer_for_c = prompts_autoreviewer[sdata["prompt_for_neural"]]
        sdata["prompt_autoreviewer"] = prompt_autoreviewer_for_c

    response_data["continuation_side_data"] = continuation_side_data

    selector_inputs = [sdata["prompt_selector"] + c
                       for c, sdata in zip(continuations, continuation_side_data)]

    selector_inputs = pd.DataFrame(
        {
            "selector_input": selector_inputs,
            "prompt_finalchar": [
                ORIG_POST_CHAR_CHINESE for _ in range(len(selector_inputs))
            ],
        }
    )

    selection_results = predict_select(
        selector_inputs,
    )
    response_data["selection_proba"] = [float(p) for p in selection_results]

    # TODO: (cleanup) remove later
    with LogExceptionAndSkip("log fic/review probs"):
        for c, sd, p in zip(continuations, continuation_side_data, response_data["selection_proba"]):
            if "Original fic" in sd.get("prompt_for_neural", "") or "Book review" in sd.get("prompt_for_neural", ""):
                print(f"\n--------Selection proba {p:.1%} for: \n--------")
                print(c[:4000])
                print("--------")

    sentiment_inputs = pd.DataFrame({"selector_input": response_data["continuations"]})
    sentiment_results = predict_sentiment(sentiment_inputs)
    response_data["sentiment_logit_diffs"] = [float(p) for p in sentiment_results]

    autoreview_inputs = selector_inputs

    # TODO: (nwo) maybe combine prompts_autoreviewer with prompts_selector?
    autoreview_inputs["selector_input"] = [
        sdata["prompt_autoreviewer"] + c
        for c, sdata in zip(continuations, continuation_side_data)
    ]

    autoreview_results = predict_autoreview(
        autoreview_inputs,
    )
    response_data["autoreview_proba"] = [float(p) for p in autoreview_results]

    if GLOBAL_DEBUG:
        print(f"sending back: {response_data}")

    return response_data


# TODO (cleanup): call these fns inside the answer/textpost fns
def selection_proba_from_gpt(texts: List[str], verbose=True):
    selector_inputs = pd.DataFrame(
        {
            "selector_input": texts,
            "prompt_finalchar": ["" for _ in range(len(texts))],  # unused but necessary
        }
    )
    selection_results = predict_select(
        selector_inputs, verbose=verbose
    )
    results = [float(p) for p in selection_results]

    return results


def sentiment_logit_diffs_from_gpt(texts: List[str], verbose=True):
    sentiment_inputs = pd.DataFrame({"selector_input": texts})
    sentiment_results = predict_sentiment(sentiment_inputs, verbose=verbose)
    results = [float(p) for p in sentiment_results]

    return results


def autoreview_proba_from_gpt(texts: List[str], verbose=True):
    autoreview_inputs = pd.DataFrame(
        {
            "selector_input": texts,
            "prompt_finalchar": ["" for _ in range(len(texts))],
        }
    )
    autoreview_results = predict_autoreview(
        autoreview_inputs, verbose=verbose
    )
    results = [float(p) for p in autoreview_results]

    return results


def prob_delta_from_gpt(text: List[str], text_ref: List[str], token_str: str,
                        forbidden_strings: List[List[str]]):
    raw = generator_model.get_prob_delta_over_ref_multi(text=text, text_ref=text_ref, token_str=token_str,
                                                        forbidden_strings=forbidden_strings)
    return raw[0]["result"]
