import time
import pickle
from textwrap import wrap, fill
from string import whitespace, punctuation
from itertools import chain, product
from typing import List

import requests
import pandas as pd

from autoresponder_config import *
from autoresponder_static_v8 import *

from bridge_shared import bridge_service_unique_id, bridge_service_url
from mood import get_mood_by_name, load_logit_diff_sample, estimate_expected_rejections, logit_diff_to_pos_sent
from selector import serve_selection

from experimental.year_munging import sample_and_substitute_year_v10

import bridge_cache_singleton

TRADE_QUALITY_FOR_SPEED = True

logit_diff_sample_series = load_logit_diff_sample()
EXPECTED_REJECTION_MULT = 0.5 if (not TRADE_QUALITY_FOR_SPEED) else 0.4

TEXTPOST_N_CANDIDATES_TARGET = 15 if (not TRADE_QUALITY_FOR_SPEED) else 12

Q_CHAR = "会"
A_CHAR = "域"
T_CHAR = "职"

UNAME_CHAR = "友"
ORIG_POST_CHAR = "翰"

AB_TEST_SELECTOR = True
AB_TEST_A_SEQUENCE = "\uFFF9"
AB_TEST_B_SEQUENCE = "\uFFF9\uFFFA\uFFFB"

DO_FAKE_V10_YEAR_MONTH = False
FAKE_V10_YEAR_MONTH = "December 2020"

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


def finalize_prompt_for_neural(
    prompt,
    override_disable_forumlike=False,
    forced_tags_string=None,
    write_fic_override=False,
):
    if GLOBAL_DEBUG:
        print(f"in finalize_prompt_for_neural, got prompt: {repr(prompt)}")
    prompt = final_munge_before_neural(
        prompt,
        override_disable_forumlike=override_disable_forumlike,
        left_strip_newline=SELECTOR_LEFT_STRIP_NEWLINE_IN_FORUMLIKE,
        forced_tags_string=forced_tags_string,
        write_fic_override=write_fic_override,
    )
    prompt = prompt.replace(EOT_FULL, "")
    if EOT_PREPEND:
        prompt = EOT_FULL + prompt
    if GLOBAL_DEBUG:
        print(f"finalize_prompt_for_neural, using prompt (munged): {repr(prompt)}")
    return prompt


class MLModelInterface:
    name: str
    uses_bridge_cache: bool

    def __init__(self):
        raise NotImplementedError

    def do(self, method, *args, repeat_until_done_signal=False, **kwargs):
        data = {
            "model": self.name,
            "method": method,
            "args": args,
            "kwargs": kwargs,
            "repeat_until_done_signal": repeat_until_done_signal,
        }
        if self.uses_bridge_cache:
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
    avoid_if_under=20,
    avoid_half_if_under=40,
    use_textpost_prompt=False,
    avoid_initial_blockquote=False,
    avoid_if_profane=False,
    avoid_if_says_frank=False,
    v10_timestamp="",
    mirotarg=None,
    forced_tags_string=None,
    write_fic_override=False,
    override_disable_forumlike=False,
    verbose=False,
):
    continuation_side_data = []

    if use_textpost_prompt:
        raw_prompts, overrides, probs = get_textpost_prompts()
        prompts = []
        for p, o in zip(raw_prompts, overrides):
            this_v10_timestamp = o.get("v10_timestamp", v10_timestamp)

            ts_string = format_segment_v8_time(
                this_v10_timestamp, control_seg_config=CONTROL_SEG_CONFIG
            )
            if CONTROL_SEG_CONFIG["flags"]["add_control_prefix_to_forced_tag_strings"]:
                tag_string = format_segment_v8_tags(
                    o.get("tag_string_raw", ""),
                    control_seg_config=CONTROL_SEG_CONFIG,
                )
            else:
                tag_string = o.get("tag_string_raw", "")
            prompt = globally_format_v8(
                doc_tagless=p,
                ts_string=ts_string,
                interlocutor_string=format_segment_v8_interlocutors(""),
                tag_string=tag_string,
                control_seg_config=CONTROL_SEG_CONFIG,
            )
            prompt = finalize_prompt_for_neural(
                prompt,
                override_disable_forumlike=use_textpost_prompt
                or override_disable_forumlike,
                forced_tags_string=forced_tags_string,
                write_fic_override=write_fic_override,
            )
            prompts.append(prompt)

        print(f"formed prompts:")
        for _p in prompts:
            print(repr(_p))
        bridge_id = generator_model.write_random_prompt(
            prompts,
            probs,
            repeat_until_done_signal=True,
            verbose=verbose,
            mirotarg=mirotarg,
        )
    elif V8:
        prompt = join_time_sidechannel(prompt, v10_timestamp)

        prompt = finalize_prompt_for_neural(
            prompt,
            override_disable_forumlike=use_textpost_prompt
            or override_disable_forumlike,
            forced_tags_string=forced_tags_string,
            write_fic_override=write_fic_override,
        )

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

    continuations_ = []
    for continuation, pr in zip(continuations, all_prompts):
        if use_textpost_prompt:
            continuation = pr + continuation
            if EOT_PREPEND and continuation.startswith("<|endoftext|>"):
                continuation = continuation[len("<|endoftext|>") :]
            if continuation.startswith(ORIG_POST_CHAR_CHINESE):
                continuation = CONTROL_SEG_CONFIG[
                    "ORIG_POST_CHAR_FORUMLIKE"
                ] + continuation.lstrip(ORIG_POST_CHAR_CHINESE)
        continuations_.append(continuation)
    continuations = continuations_

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


def predict_select(data, override_disable_forumlike=False):
    t1 = time.time()

    if len(data) == 0:
        # this can happen if the retention_stack is big enough
        return np.array([])

    selector_input = []
    for text in data.selector_input:
        for end_segment in {
            eot_end_segment,
            "<|",
        }:  # explicitly support old <| thing, for now
            if text.endswith(end_segment):
                text = text[: -len(end_segment)]
        if T_CHAR not in text and (not V8):
            text = text + T_CHAR

        text = final_munge_before_neural(
            text,
            override_disable_forumlike=override_disable_forumlike,
            left_strip_newline=SELECTOR_LEFT_STRIP_NEWLINE_IN_FORUMLIKE,
        )

        if EOT_PREPEND:
            if (not SELECTOR_EOT_PREPEND) and text.startswith(EOT_FULL):
                text = text[len(EOT_FULL) :]
            if SELECTOR_EOT_PREPEND and (not text.startswith(EOT_FULL)):
                text = EOT_FULL + text

        selector_input.append(text)
    data.loc[:, "selector_input"] = selector_input

    data = data.to_dict(orient="records")

    response_data = selector_est.predict_proba(data)

    result = np.array(response_data[0]["result"])
    probs = result[:, 1]

    delta_t = time.time() - t1
    print(f'predict_select: served in {delta_t:.1f}s')

    return probs


def predict_sentiment(data):
    t1 = time.time()

    data["prompt_finalchar"] = ["" for _ in data.selector_input.values]
    if len(data) == 0:
        # this can happen if the retention_stack is big enough
        return np.array([])

    selector_input = []
    for text in data.selector_input:
        for end_segment in {
            eot_end_segment,
            "<|",
        }:  # explicitly support old <| thing, for now
            if text.endswith(end_segment):
                text = text[: -len(end_segment)]
        if T_CHAR not in text:
            text = text + T_CHAR
        text = text.partition(T_CHAR)[0]
        text = normalize_for_generator(text)
        text = re.sub(r"\<.*?\>", "", text)  # sentiment-specific

        if EOT_PREPEND:
            if (not SELECTOR_EOT_PREPEND) and text.startswith(EOT_FULL):
                text = text[len(EOT_FULL) :]
            if SELECTOR_EOT_PREPEND and (not text.startswith(EOT_FULL)):
                text = EOT_FULL + text

        selector_input.append(text)
    data.loc[:, "selector_input"] = selector_input

    data = data.to_dict(orient="records")

    response_data = sentiment_est._predict(data, key="logits")

    logits = np.array(response_data[0]["result"])

    logit_diffs = logits[:, 1:] - logits[:, :1]

    delta_t = time.time() - t1
    print(f'predict_sentiment: served in {delta_t:.1f}s')

    return logit_diffs


def predict_autoreview(data, debug=False, override_disable_forumlike=False):
    t1 = time.time()

    selector_input = []
    for text in data.selector_input:
        if text.endswith(eot_end_segment):
            text = text[: -len(eot_end_segment)]

        text = final_munge_before_neural(
            text, override_disable_forumlike=override_disable_forumlike
        )

        if EOT_PREPEND:
            if (not SELECTOR_EOT_PREPEND) and text.startswith(EOT_FULL):
                text = text[len(EOT_FULL) :]
            if SELECTOR_EOT_PREPEND and (not text.startswith(EOT_FULL)):
                text = EOT_FULL + text

        selector_input.append(text)
    if debug:
        print("autoreviewer model will see exactly the following:\n")
        for s in selector_input:
            print(repr(s))
        print()
    data.loc[:, "selector_input"] = selector_input

    data = data.to_dict(orient="records")

    response_data = autoreviewer_est.predict_proba(data)

    result = np.array(response_data[0]["result"])
    probs = result[:, 1]

    delta_t = time.time() - t1
    print(f'predict_autoreview: served in {delta_t:.1f}s')

    return probs


RESULT_STACK = {}


def _make_alt_timestamps(v10_timestamp):
    if v10_timestamp is None:
        return []

    alts = []
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "November",
        "December",
    ]
    years = ["2019", "2020", "2021"][::-1]
    for year in years:
        for month in months:
            alts.append(v10_timestamp.replace("January", month).replace("2021", year))
    return alts


def save_retention(retention_stack):
    with open("data/retention_stack.pkl.gz", "wb") as f:
        pickle.dump(retention_stack, f)

    with open("data/retention_stack_backup.pkl.gz", "wb") as f:
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
        asking_name="",
        mood_name=None,
        exact_prompt=False,
        forced_tags_string="",
        write_fic_override=False,
        write_review_override=False,
        selector_cut_to_final_exchange=False,
        avoid_initial_blockquote=False,
        ts=None,
        no_timestamp=False
):
    t1 = time.time()

    if ts is None:
        ts = datetime.now()
    v10_timestamp = timestamp_to_v10_format(ts)
    if DO_FAKE_V10_YEAR_MONTH:
        v10_timestamp= (
            " ".join(v10_timestamp.split(" ")[:2]) + " " + FAKE_V10_YEAR_MONTH
        )
    if no_timestamp:
        v10_timestamp = ""

    mood = get_mood_by_name(mood_name)

    result_generator = old_bridge_call__answer(
        prompt=prompt,
        asking_name=asking_name,
        mood=mood,
        exact_prompt=exact_prompt,
        v10_timestamp=v10_timestamp,
        forced_tags_string=forced_tags_string,
        write_fic_override=write_fic_override,
        write_review_override=write_review_override,
        selector_cut_to_final_exchange=selector_cut_to_final_exchange,
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
    result["question"] = prompt  # TODO [cleanup]: rename if safe
    result["asking_name"] = asking_name
    result["v10_timestamp"] = v10_timestamp
    result["mood"] = mood_name

    delta_t = time.time() - t1
    log_prefix = ""
    if write_fic_override:
        log_prefix += " (story)"
    print(f'answer_from_gpt{log_prefix}: served in {delta_t:.1f}s')

    return result


def generator_and_selector_timestamps(random_year_for_generator: bool, v10_timestamp: str):
    if random_year_for_generator and v10_timestamp is not None:
        generator_v10_timestamp = sample_and_substitute_year_v10(v10_timestamp)
        selector_v10_timestamp = v10_timestamp
    else:
        generator_v10_timestamp = v10_timestamp
        selector_v10_timestamp = v10_timestamp

    print(f"generator_v10_timestamp: {repr(generator_v10_timestamp)}")
    print(f"selector_v10_timestamp: {repr(selector_v10_timestamp)}")

    return generator_v10_timestamp, selector_v10_timestamp


def old_bridge_call__answer(
        prompt,
        asking_name="",
        mood=None,
        exact_prompt=False,
        v10_timestamp="",
        forced_tags_string="",
        write_fic_override=False,
        write_review_override=False,
        selector_cut_to_final_exchange=False,
        avoid_initial_blockquote=False
):
    if not exact_prompt:
        prompt = (
            UNAME_CHAR
            + asking_name
            + DEFAULT_CSC["ASK_CHAR"]
            + "\n"
            + prompt
            + "\n"
            + A_CHAR
        )
        print(f"formed prompt: {prompt}")

    best_of = 13 if (not TRADE_QUALITY_FOR_SPEED) else 10

    if write_fic_override or write_review_override:
        best_of = 6 if not (TRADE_QUALITY_FOR_SPEED) else 4

    best_of = adjust_best_of(best_of, mood)

    avoid_if_under = 5
    if write_fic_override:
        avoid_if_under = 50
    avoid_half_if_under = 5
    avoid_if_profane = False
    avoid_if_says_frank = False
    random_year_for_generator = True
    if asking_name == "bukbot":
        avoid_if_profane = True

    # old serve_answer
    print("\n------------\n")
    prompt = prompt.rstrip(whitespace)

    if EOT_PREPEND and not V8:
        prompt = "<|endoftext|>" + prompt

    print(f"write_fic_override: {write_fic_override}")

    generator_v10_timestamp, selector_v10_timestamp = generator_and_selector_timestamps(
        random_year_for_generator, v10_timestamp
    )

    override_disable_forumlike = False
    if prompt.startswith(CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"]):
        override_disable_forumlike = True

    continuations, continuation_side_data = basic_n_continuations(
        prompt,
        N=best_of,
        avoid_if_under=avoid_if_under,
        avoid_half_if_under=avoid_half_if_under,
        use_textpost_prompt=False,
        avoid_initial_blockquote=avoid_initial_blockquote,
        avoid_if_profane=avoid_if_profane,
        avoid_if_says_frank=avoid_if_says_frank,
        v10_timestamp=generator_v10_timestamp,
        forced_tags_string=forced_tags_string,
        write_fic_override=write_fic_override,
        override_disable_forumlike=override_disable_forumlike,
    )
    response_data = {}
    delete_title = write_fic_override and CONTROL_SEG_CONFIG["flags"].get("fic_override_v2", False)
    response_data["continuations"] = [final_munge_after_neural(c, delete_title=delete_title) for c in continuations]
    response_data["continuation_side_data"] = continuation_side_data
    response_data["generator_v10_timestamp"] = generator_v10_timestamp
    response_data["selector_v10_timestamp"] = selector_v10_timestamp

    if SELECTOR_CAN_SEE_PROMPTS:
        if selector_cut_to_final_exchange and not override_disable_forumlike:
            prompt_cut = cut_to_final_exchange_chinese(prompt)
            selector_inputs = [
                prompt_cut + final_munge_after_neural(c, delete_title=delete_title) for c in continuations
            ]
        else:
            selector_inputs = [prompt + c for c in continuations]
    else:
        prompt_forumlike = substitute_forumlike(
            normalize_for_generator(prompt),
            shuffle=False,
            infer_first=False,
            left_strip_newline=SELECTOR_LEFT_STRIP_NEWLINE_IN_FORUMLIKE,
        )
        prompt_finalchar = prompt_forumlike[
            last_control_char(
                prompt_forumlike,
                incl_number=False,
                control_seg_config=CONTROL_SEG_CONFIG,
            )[1] :
        ]
        selector_inputs = [prompt_finalchar + c for c in continuations]

    if DO_ALT_TIMESTAMPS:
        for alt_ts in _make_alt_timestamps(v10_timestamp):
            alt_selector_inputs = pd.DataFrame(
                {
                    "selector_input": [
                        join_time_sidechannel(s, alt_ts) for s in selector_inputs
                    ]
                }
            )
            entry_selection_results = predict_select(alt_selector_inputs)
            listkey = f"alt_selection_proba__{alt_ts.replace(' ', '_')}"
            response_data[listkey] = [float(p) for p in entry_selection_results]

    selector_inputs = [
        join_time_sidechannel(s, selector_v10_timestamp) for s in selector_inputs
    ]
    selector_inputs = pd.DataFrame(
        {
            "selector_input": selector_inputs,
            "prompt_finalchar": [f"{A_CHAR}a" for _ in range(len(selector_inputs))],
        }
    )
    if GLOBAL_DEBUG:
        print(f"passing to predict_select: {selector_inputs}")
    selection_results = predict_select(
        selector_inputs,
        override_disable_forumlike=override_disable_forumlike,
    )
    response_data["selection_proba"] = [float(p) for p in selection_results]
    selector_inputs = pd.DataFrame({"selector_input": response_data["continuations"]})
    sentiment_results = predict_sentiment(selector_inputs)
    response_data["sentiment_logit_diffs"] = [float(p) for p in sentiment_results]

    autoreview_inputs = [
        cut_to_new_since_last_frank_post(prompt + final_munge_after_neural(c, delete_title=delete_title))
        for c in continuations
    ]

    autoreview_inputs = [
        join_time_sidechannel(s, selector_v10_timestamp) for s in autoreview_inputs
    ]

    autoreview_inputs = pd.DataFrame(
        {
            "selector_input": autoreview_inputs,
            "prompt_finalchar": ["" for _ in range(len(autoreview_inputs))],
        }
    )
    autoreview_results = predict_autoreview(
        autoreview_inputs,
        debug=False,
    )
    response_data["autoreview_proba"] = [float(p) for p in autoreview_results]

    if GLOBAL_DEBUG:
        print(f"sending back: {response_data}")

    return response_data


def text_post_from_gpt(loop_persistent_data, mood_name=None, ts=None):
    t1 = time.time()

    mood = get_mood_by_name(mood_name)

    if ts is None:
        ts = datetime.now()
    v10_timestamp = timestamp_to_v10_format(ts)
    if DO_FAKE_V10_YEAR_MONTH:
        v10_timestamp = (
            " ".join(v10_timestamp.split(" ")[:2]) + " " + FAKE_V10_YEAR_MONTH
        )

    n_retention = len(loop_persistent_data.retention_stack)

    result_generator = old_bridge_call__textpost(n_retention=n_retention,
                                                 mood=mood,
                                                 v10_timestamp=v10_timestamp)

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
    result["v10_timestamp"] = v10_timestamp
    result["mood"] = mood_name

    delta_t = time.time() - t1
    print(f'text_post_from_gpt: served in {delta_t:.1f}s')

    return result, loop_persistent_data


def old_bridge_call__textpost(
        n_retention,
        mood=None,
        v10_timestamp="",
):
    avoid_if_under = 10
    avoid_half_if_under = 10
    avoid_initial_blockquote = False
    avoid_if_says_frank = False
    random_year_for_generator = True

    best_of = TEXTPOST_N_CANDIDATES_TARGET
    best_of = adjust_best_of(best_of, mood)

    if n_retention is not None:
        best_of = max(1, best_of - n_retention)
        print(f"with {n_retention} on stack, only need {best_of}")

    print(f"n_retention {n_retention}")

    # old serve_textpost

    generator_v10_timestamp, selector_v10_timestamp = generator_and_selector_timestamps(
        random_year_for_generator, v10_timestamp
    )

    continuations, continuation_side_data = basic_n_continuations(
        prompt="",
        N=best_of,
        avoid_if_under=avoid_if_under,
        avoid_half_if_under=avoid_half_if_under,
        use_textpost_prompt=True,
        avoid_initial_blockquote=avoid_initial_blockquote,
        avoid_if_says_frank=avoid_if_says_frank,
        v10_timestamp=generator_v10_timestamp,
    )

    # dreams coldstart curiosity
    # TODO: remove later
    qc = 0
    for c in continuations:
        if "#dreams" in c:
            qc += 1
            print(c, end='\n----------\n')
    print(f'{qc} / {len(continuations)} have dreams')

    response_data = {}
    response_data["continuations"] = [final_munge_after_neural(c) for c in continuations]
    response_data["continuation_side_data"] = continuation_side_data
    response_data["generator_v10_timestamp"] = generator_v10_timestamp
    response_data["selector_v10_timestamp"] = selector_v10_timestamp

    selector_inputs = [c for c in continuations]
    for alt_char in [
        CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"],
        CONTROL_SEG_CONFIG["ORIG_FICTION_CHAR_FORUMLIKE"],
    ]:
        selector_inputs = [
            s.replace(alt_char, CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"])
            for s in selector_inputs
        ]

    selector_inputs = [
        s.replace(generator_v10_timestamp, selector_v10_timestamp)
        for s in selector_inputs
    ]
    selector_inputs = pd.DataFrame(
        {
            "selector_input": selector_inputs,
            "prompt_finalchar": [
                ORIG_POST_CHAR_CHINESE for _ in range(len(selector_inputs))
            ],
        }
    )
    if GLOBAL_DEBUG:
        print(f"passing to predict_select: {selector_inputs}")
    selection_results = predict_select(
        selector_inputs,
        override_disable_forumlike=True,
    )
    response_data["selection_proba"] = [float(p) for p in selection_results]

    sentiment_inputs = pd.DataFrame({"selector_input": response_data["continuations"]})
    sentiment_results = predict_sentiment(sentiment_inputs)
    response_data["sentiment_logit_diffs"] = [float(p) for p in sentiment_results]

    autoreview_inputs = selector_inputs
    autoreview_results = predict_autoreview(
        autoreview_inputs,
        override_disable_forumlike=True,
        debug=False,
    )
    response_data["autoreview_proba"] = [float(p) for p in autoreview_results]

    if GLOBAL_DEBUG:
        print(f"sending back: {response_data}")

    return response_data


def selection_proba_from_gpt2_service(texts: List[str], timestamp: str = None):
    if timestamp is None:
        timestamp = ""

    texts = [join_time_sidechannel(s, timestamp) for s in texts]
    texts = [final_munge_before_neural(s) for s in texts]

    selector_inputs = pd.DataFrame(
        {
            "selector_input": texts,
            "prompt_finalchar": ["" for _ in range(len(texts))],  # unused but necessary
        }
    )
    selection_results = predict_select(
        selector_inputs, override_disable_forumlike=True
    )
    results = [float(p) for p in selection_results]

    return results


def sentiment_logit_diffs_from_gpt2_service(texts: List[str]):
    sentiment_inputs = pd.DataFrame({"selector_input": texts})
    sentiment_results = predict_sentiment(sentiment_inputs)
    results = [float(p) for p in sentiment_results]

    return results


def autoreview_proba_from_gpt2_service(texts: List[str], timestamp: str = None):
    if timestamp is None:
        timestamp = ""

    autoreview_inputs = [cut_to_new_since_last_frank_post(s) for s in texts]

    autoreview_inputs = [join_time_sidechannel(s, timestamp) for s in autoreview_inputs]

    autoreview_inputs = pd.DataFrame(
        {
            "selector_input": autoreview_inputs,
            "prompt_finalchar": ["" for _ in range(len(autoreview_inputs))],
        }
    )
    autoreview_results = predict_autoreview(
        autoreview_inputs,
        debug=False,
    )
    results = [float(p) for p in autoreview_results]

    return results
