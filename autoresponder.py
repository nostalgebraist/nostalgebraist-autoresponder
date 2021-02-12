import sys
import json
import os
import subprocess
import pickle
import re
import time
from string import whitespace
from textwrap import wrap, fill
from functools import partial

# TODO: use the following
# import textwrap

# def optional_fill(s, wrap=True):
#   return textwrap.fill(s) if wrap else s

# def optional_wrap(s, wrap=True):
#   return textwrap.fill(s) if wrap else s

import requests
import numpy as np
import pandas as pd
from IPython.utils.capture import capture_output

import tensorflow as tf

import model
import encoder

from flask import Flask, escape, request, jsonify

from autoresponder_config import *
from autoresponder_static import *
from autoresponder_static_v8 import *

from experimental.generator_model import GeneratorModel, is_repeating_criterion
from selector_model.selector_estimator import SelectorEstimatorFromCkpt

# TODO: move this over later
drivedir = "/content/drive/MyDrive/gpt-2/"
os.chdir("/")

hparams = model.hparams_1558M()

hparams.set_hparam("attn_dropout", 0)
hparams.set_hparam("res_dropout", 0)

start_token = None

if V10:
    CONTROL_SEG_CONFIG = CONTROL_SEG_CONFIGS["V10"]
else:
    CONTROL_SEG_CONFIG = CONTROL_SEG_CONFIGS["V9"]

if FORUMLIKE:
    ORIG_POST_CHAR = CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"]
else:
    ORIG_POST_CHAR = ORIG_POST_CHAR_CHINESE


def load_from_gdrive_with_gs_fallback(
    load_fn, relative_path, gs_command, retries=False, **kwargs
):
    local_gdrive_path = os.path.join(drivedir, relative_path)
    local_gs_path = os.path.join("/", relative_path)
    print(f"local_gdrive_path: {local_gdrive_path}")
    print(f"local_gs_path: {local_gs_path}")

    enclosing_dir = local_gs_path.rpartition("/")[0]
    os.makedirs(enclosing_dir, exist_ok=True)

    enclosing_dir_exists = os.path.exists(enclosing_dir)
    target_exists = os.path.exists(local_gs_path)

    print(f"local_gs: enclosing dir {enclosing_dir} exists?: {enclosing_dir_exists}")
    print(f"local_gs: target {local_gs_path} exists?: {target_exists}")

    if not target_exists:
        try:
            print(f"local_gdrive: trying to load from {local_gdrive_path}...")
            return load_fn(path=local_gdrive_path, retries=False, **kwargs)
        except (OSError, FileNotFoundError, KeyError):
            print(f"local_gdrive failure, falling back to local_gs")
            print(f"downlading from gs...")
            subprocess.check_output(gs_command, shell=True)
    return load_fn(path=local_gs_path, retries=retries, **kwargs)


def load_encoder_only(path, retries=False):  # ignored
    if path.endswith("vocab.bpe"):
        enclosing_dir = path.rpartition("/")[0]
        path = enclosing_dir
    enc = encoder.get_encoder_from_path(path, eot_workaround=EOT_WORKAROUND)
    return enc


enc = load_from_gdrive_with_gs_fallback(
    load_fn=load_encoder_only,
    relative_path=os.path.join("models", model_name, "vocab.bpe"),
    gs_command=gs_command_get_encoder,
)


def load_generator_model(
    path, enc, batch_size, sample_done_criterion, hparams, retries=False
):
    return GeneratorModel.load(
        path, enc, batch_size, sample_done_criterion, hparams, retries=retries
    )


def make_sample_done_criterion(control_seg_config):
    def sample_done_criterion(text, unique_token_frac):
        has_EOT = eot_end_segment in text

        has_control_chars = contains_control_chars(
            text, control_seg_config=control_seg_config
        )

        has_multiple_tag_chars = len([char for char in text if char == T_CHAR]) >= 2

        is_repeating = is_repeating_criterion(unique_token_frac)

        return has_EOT or has_control_chars or has_multiple_tag_chars or is_repeating

    return sample_done_criterion


sample_done_criterion = make_sample_done_criterion(
    control_seg_config=CONTROL_SEG_CONFIG
)

generator_model = load_from_gdrive_with_gs_fallback(
    load_fn=load_generator_model,
    relative_path=os.path.join(model_path),
    gs_command=gs_command_get_model,
    enc=enc,
    batch_size=batch_size,
    sample_done_criterion=sample_done_criterion,
    hparams=hparams,
)


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


def get_prompted_continuation(
    prompt: str,
    verbose=False,
    mirotarg=None,  # TODO: allow vary across batch, add noise inside this fn
):
    if mirotarg is None:
        mirotarg = np.random.choice(MIRO_TARGET_ALL)

    return generator_model.write(prompt, mirotarg=mirotarg, verbose=verbose)


def parse_continuation(continuation: str, verbose=True, wrap=False):
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


def get_textpost_prompt():
    overrides = {}

    roll = np.random.rand()
    if roll < FORUMLIKE_REVIEW_PROB:
        prompt = CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"]
        overrides["v8_timestamp"] = ""
        overrides["v10_timestamp"] = ""
    elif roll < FORUMLIKE_REVIEW_PROB + FORUMLIKE_FIC_PROB:
        prompt = CONTROL_SEG_CONFIG["ORIG_FICTION_CHAR_FORUMLIKE"]
        overrides["v8_timestamp"] = ""
        overrides["v10_timestamp"] = ""
        overrides["tag_string_raw"] = "#original fiction"
    else:
        prompt = CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"]
    print(f"using prompt={repr(prompt)}")

    return prompt, overrides


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
    avoid_if_cut_off=True,
    split_on_control_char=False,
    use_textpost_prompt=False,
    avoid_initial_blockquote=False,
    continue_if_cut_off=False,
    avoid_if_profane=False,
    v8_timestamp="",
    v10_timestamp="",
    max_continue_steps=MAX_CONTINUE_STEPS,
    mirotarg=None,
    forced_tags_string=None,
    write_fic_override=False,
    override_disable_forumlike=False,
    verbose=False,
):
    continuation_side_data = []

    if mirotarg is None:
        mirotarg = np.random.choice(MIRO_TARGET_ALL)

    relevant_timestamp = v10_timestamp if V10 else v8_timestamp

    if use_textpost_prompt:
        prompt, textpost_overrides = get_textpost_prompt()
        v8_timestamp = textpost_overrides.get("v8_timestamp", v8_timestamp)
        v10_timestamp = textpost_overrides.get("v10_timestamp", v10_timestamp)
        relevant_timestamp = v10_timestamp if V10 else v8_timestamp

        if V8:
            ts_string = format_segment_v8_time(
                relevant_timestamp, control_seg_config=CONTROL_SEG_CONFIG
            )
            if CONTROL_SEG_CONFIG["flags"]["add_control_prefix_to_forced_tag_strings"]:
                tag_string = format_segment_v8_tags(
                    textpost_overrides.get("tag_string_raw", ""),
                    control_seg_config=CONTROL_SEG_CONFIG,
                )
            else:
                tag_string = textpost_overrides.get("tag_string_raw", "")
            prompt = globally_format_v8(
                doc_tagless=prompt,
                ts_string=ts_string,
                interlocutor_string=format_segment_v8_interlocutors(""),
                tag_string=tag_string,
                control_seg_config=CONTROL_SEG_CONFIG,
            )
    elif V8:
        prompt = join_time_sidechannel(prompt, relevant_timestamp)

    prompt = finalize_prompt_for_neural(
        prompt,
        override_disable_forumlike=use_textpost_prompt or override_disable_forumlike,
        forced_tags_string=forced_tags_string,
        write_fic_override=write_fic_override,
    )

    if GLOBAL_DEBUG:
        print(f"in basic_n_continuations, using prompt: {repr(prompt)}")
    continuations = []

    while len(continuations) < N:
        print(f"\ncontinuing, have {len(continuations)} of {N}\n")

        this_batch_continuations = get_prompted_continuation(
            prompt,
            verbose=verbose,
            mirotarg=mirotarg,
        )

        for c in this_batch_continuations:
            if contains_control_chars(c, control_seg_config=CONTROL_SEG_CONFIG):
                if split_on_control_char:
                    # min_ix = min([i for i, char in enumerate(c) if char in {Q_CHAR, A_CHAR, ORIG_POST_CHAR, UNAME_CHAR}])
                    min_ix = first_control_char(
                        c, control_seg_config=CONTROL_SEG_CONFIG
                    )[1]
                    csub = c[:min_ix]
                    print(f"splitting on control char:")
                    print(
                        f"\t{len(c)} chars, {len(c.split(' '))} words-->\n\t{len(csub)} chars, {len(csub.split(' '))} words"
                    )
                    c = csub
                else:
                    print(f"rejecting because control char: \n{fill(c)}\n")
                    continue

            roll = np.random.rand()
            if len(c.partition("\n")[2].split(" ")) < avoid_if_under:
                print(f"rejecting because length under {avoid_if_under}: \n{fill(c)}\n")
            elif (
                len(c.partition("\n")[2].split(" ")) < avoid_half_if_under
            ) and roll < 0.5:
                print(
                    f"rejecting because length under {avoid_half_if_under} and roll {roll}: \n{fill(c)}\n"
                )
            elif (not c.endswith(eot_end_segment)) and avoid_if_cut_off:
                print(f"rejecting because cut off: \n{fill(c)}\n")
            elif (
                c.partition("\n")[2].lstrip(" \n").startswith("<blockquote")
            ) and avoid_initial_blockquote:
                print(f"rejecting because initial blockquote: \n{fill(c)}\n")
            elif len([char for char in c if char == T_CHAR]) >= 2:
                print(f"rejecting because multiple T_CHAR: \n{fill(c)}\n")
            elif (
                any([subs in c.lower() for subs in profane_substrings])
                and avoid_if_profane
            ):
                print(f"rejecting because profane: \n{fill(c)}\n")
            elif normalize_for_generator(
                c.partition(T_CHAR)[0].strip(whitespace)
            ) in normalize_for_generator(prompt):
                print(f"rejecting because repeating myself: \n{fill(c)}\n")
            else:
                if len(c.partition("\n")[2].split(" ")) < avoid_half_if_under:
                    print(
                        f"keeping with roll {roll}, although length under {avoid_half_if_under}"
                    )
                continuations.append(c)
                continuation_side_data.append({"mirotarg": mirotarg})

    continuations_ = []
    for continuation in continuations:
        if use_textpost_prompt:
            continuation = prompt + continuation
            if EOT_PREPEND and continuation.startswith("<|endoftext|>"):
                continuation = continuation[len("<|endoftext|>") :]
            if FORUMLIKE and continuation.startswith(ORIG_POST_CHAR_CHINESE):
                continuation = CONTROL_SEG_CONFIG[
                    "ORIG_POST_CHAR_FORUMLIKE"
                ] + continuation.lstrip(ORIG_POST_CHAR_CHINESE)
        continuations_.append(continuation)
    continuations = continuations_

    return continuations, continuation_side_data, prompt_for_neural


def logit_diff_to_pos_sent(x):
    return 1 / (1 + np.exp(-x))


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


def load_selector(path, session, base_hparams, enc, retries=False, **kwargs):
    selector_est = SelectorEstimatorFromCkpt.load(
        path, session=session, base_hparams=base_hparams, enc=enc, **kwargs
    )
    return selector_est


selector_est = load_from_gdrive_with_gs_fallback(
    load_fn=load_selector,
    relative_path=ckpt_select.rpartition("/")[0],  # TODO: redefine ckpt_select
    gs_command=gs_command_get_selector,
    session=generator_model.session,
    base_hparams=hparams,
    enc=enc,
    batch_size=batch_size,
)
selector_est.length = length_select

lr_calib_resp = selector_est.lr_calib_resp_
lr_calib_orig = selector_est.lr_calib_orig_


sentiment_est = load_from_gdrive_with_gs_fallback(
    load_fn=load_selector,
    relative_path=ckpt_sentiment.rpartition("/")[0],  # TODO: redefine ckpt_select
    gs_command=gs_command_get_sentiment,
    session=generator_model.session,
    base_hparams=hparams,
    enc=enc,
    batch_size=batch_size,
)
sentiment_est.length = length_sentiment

lr_calib_sentiment = selector_est.lr_calib_


def predict_select(data, debug=False, override_disable_forumlike=False):
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

    probs = selector_est.predict_proba(data)[:, 1]
    return probs


def predict_sentiment(data, debug=False):
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
        if NORMALIZE:
            text = normalize_for_generator(text)
        text = re.sub(r"\<.*?\>", "", text)  # sentiment-specific

        if EOT_PREPEND:
            if (not SELECTOR_EOT_PREPEND) and text.startswith(EOT_FULL):
                text = text[len(EOT_FULL) :]
            if SELECTOR_EOT_PREPEND and (not text.startswith(EOT_FULL)):
                text = EOT_FULL + text

        selector_input.append(text)
    data.loc[:, "selector_input"] = selector_input

    logits = sentiment_est._predict(data, key="logits")
    logit_diffs = logits[:, 1:] - logits[:, :1]

    return logit_diffs


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


def serve_answer(data):
    print("\n------------\n")
    print("serving answer for\n")
    for k, v in data.items():
        print(f"\t{k}: {v}")
    print("\n------------\n")
    prompt = data["prompt"].rstrip(whitespace)

    if EOT_PREPEND and not V8:
        prompt = "<|endoftext|>" + prompt

    kwargs = data["kwargs"]
    avoid_if_under = kwargs.get("avoid_if_under", 20)
    avoid_half_if_under = kwargs.get("avoid_half_if_under", 40)
    avoid_if_cut_off = kwargs.get("avoid_if_cut_off", True)
    split_on_control_char = kwargs.get("split_on_control_char", False)
    avoid_initial_blockquote = kwargs.get("avoid_initial_blockquote", True)
    avoid_if_profane = kwargs.get("avoid_if_profane", False)

    continue_if_cut_off = kwargs.get("continue_if_cut_off", True)
    if continue_if_cut_off:
        avoid_if_cut_off = False

    selector_cut_to_final_exchange = kwargs.get("selector_cut_to_final_exchange", False)
    forced_tags_string = kwargs.get("forced_tags_string", None)
    write_fic_override = kwargs.get("write_fic_override", False)
    print(f"write_fic_override: {write_fic_override}")

    v8_timestamp = data.get("v8_timestamp", "")
    v10_timestamp = data.get("v10_timestamp", "")
    relevant_timestamp = v10_timestamp if V10 else v8_timestamp

    override_disable_forumlike = False
    if prompt.startswith(CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"]):
        override_disable_forumlike = True

    continuations, continuation_side_data, prompt_for_neural = basic_n_continuations(
        prompt,
        N=kwargs["best_of"],
        avoid_if_under=avoid_if_under,
        avoid_half_if_under=avoid_half_if_under,
        avoid_if_cut_off=avoid_if_cut_off,
        use_textpost_prompt=False,
        split_on_control_char=split_on_control_char,
        avoid_initial_blockquote=avoid_initial_blockquote,
        continue_if_cut_off=continue_if_cut_off,
        avoid_if_profane=avoid_if_profane,
        v8_timestamp=v8_timestamp,
        v10_timestamp=v10_timestamp,
        forced_tags_string=forced_tags_string,
        write_fic_override=write_fic_override,
        override_disable_forumlike=override_disable_forumlike,
    )
    parsed = data.copy()
    parsed["continuations"] = [final_munge_after_neural(c) for c in continuations]
    parsed["mirotarg"] = [cd.get("mirotarg") for cd in continuation_side_data]
    parsed["prompt_for_neural"] = prompt_for_neural

    if SELECTOR_CAN_SEE_PROMPTS:
        if selector_cut_to_final_exchange and not override_disable_forumlike:
            prompt_cut = cut_to_final_exchange_chinese(prompt)
            selector_inputs = [
                prompt_cut + final_munge_after_neural(c) for c in continuations
            ]
        else:
            selector_inputs = [prompt + c for c in continuations]
    else:
        if FORUMLIKE:
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
        else:
            selector_inputs = [A_CHAR + c for c in continuations]

    if DO_ALT_TIMESTAMPS:
        for alt_ts in _make_alt_timestamps(v10_timestamp):
            alt_selector_inputs = pd.DataFrame(
                {
                    "selector_input": [
                        join_time_sidechannel(s, alt_ts) for s in selector_inputs
                    ]
                }
            )
            entry_selection_results = predict_select(alt_selector_inputs, debug=True)
            listkey = f"alt_selection_proba__{alt_ts.replace(' ', '_')}"
            parsed[listkey] = [float(p) for p in entry_selection_results]

    selector_inputs = [
        join_time_sidechannel(s, relevant_timestamp) for s in selector_inputs
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
        debug=True,
        override_disable_forumlike=override_disable_forumlike,
    )
    parsed["selection_proba"] = [float(p) for p in selection_results]
    selector_inputs = pd.DataFrame({"selector_input": parsed["continuations"]})
    sentiment_results = predict_sentiment(selector_inputs, debug=True)
    parsed["sentiment_logit_diffs"] = [float(p) for p in sentiment_results]
    show_note_probas(
        continuations,
        probas=parsed["selection_proba"],
        sentiment_logit_diffs=parsed["sentiment_logit_diffs"],
    )

    if GLOBAL_DEBUG:
        print(f"sending back: {parsed}")

    return parsed


def serve_textpost(data):
    prompt = ""
    kwargs = data["kwargs"]
    avoid_if_under = kwargs.get("avoid_if_under", 20)
    avoid_half_if_under = kwargs.get("avoid_half_if_under", 40)
    avoid_if_cut_off = kwargs.get("avoid_if_cut_off", True)
    split_on_control_char = kwargs.get("split_on_control_char", True)
    avoid_initial_blockquote = kwargs.get("avoid_initial_blockquote", False)

    continue_if_cut_off = kwargs.get("continue_if_cut_off", True)
    if continue_if_cut_off:
        avoid_if_cut_off = False

    try:
        (
            continuations,
            continuation_side_data,
            prompt_for_neural,
        ) = basic_n_continuations(
            prompt,
            N=kwargs["best_of"],
            avoid_if_under=avoid_if_under,
            avoid_half_if_under=avoid_half_if_under,
            avoid_if_cut_off=avoid_if_cut_off,
            split_on_control_char=split_on_control_char,
            use_textpost_prompt=True,
            avoid_initial_blockquote=avoid_initial_blockquote,
            v8_timestamp=data.get("v8_timestamp"),
            v10_timestamp=data.get("v10_timestamp", ""),
            continue_if_cut_off=continue_if_cut_off,
        )
    except Exception as e:
        if EVEN_BETTER_LENGTH:
            raise (e)
        print(f"got {e}, trying without continue_if_cut_off")
        (
            continuations,
            continuation_side_data,
            prompt_for_neural,
        ) = basic_n_continuations(
            prompt,
            N=kwargs["best_of"],
            avoid_if_under=avoid_if_under,
            avoid_half_if_under=avoid_half_if_under,
            avoid_if_cut_off=avoid_if_cut_off,
            split_on_control_char=split_on_control_char,
            use_textpost_prompt=True,
            avoid_initial_blockquote=avoid_initial_blockquote,
            v8_timestamp=data.get("v8_timestamp"),
            v10_timestamp=data.get("v10_timestamp", ""),
            continue_if_cut_off=False,
        )
    parsed = data.copy()
    parsed["continuations"] = [final_munge_after_neural(c) for c in continuations]
    parsed["mirotarg"] = [cd.get("mirotarg") for cd in continuation_side_data]
    parsed["prompt_for_neural"] = prompt_for_neural

    if FORUMLIKE:
        selector_inputs = [c for c in continuations]
        for alt_char in [
            CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"],
            CONTROL_SEG_CONFIG["ORIG_FICTION_CHAR_FORUMLIKE"],
        ]:
            selector_inputs = [
                s.replace(alt_char, CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"])
                for s in selector_inputs
            ]
    else:
        selector_inputs = [A_CHAR + c for c in continuations]
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
        debug=True,
        override_disable_forumlike=True,
    )
    parsed["selection_proba"] = [float(p) for p in selection_results]

    selector_inputs = pd.DataFrame({"selector_input": parsed["continuations"]})
    sentiment_results = predict_sentiment(selector_inputs, debug=True)
    parsed["sentiment_logit_diffs"] = [float(p) for p in sentiment_results]
    show_note_probas(
        continuations,
        probas=parsed["selection_proba"],
        sentiment_logit_diffs=parsed["sentiment_logit_diffs"],
    )

    if GLOBAL_DEBUG:
        print(f"sending back: {parsed}")

    return parsed


def serve_raw_select(data):
    texts = data["texts"]

    if V8:
        vX_timestamp = (
            data.get("v10_timestamp", "") if V10 else data.get("v8_timestamp", "")
        )
        texts = [join_time_sidechannel(s, vX_timestamp) for s in texts]
        texts = [
            s
            if len(find_all_control_chars_chinese(s)) > 0
            else ORIG_POST_CHAR_CHINESE + s
            for s in texts
        ]
        texts = [final_munge_before_neural(s) for s in texts]
    else:
        if FORUMLIKE:
            for alt_char in [
                CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"],
                CONTROL_SEG_CONFIG["ORIG_FICTION_CHAR_FORUMLIKE"],
            ]:
                texts = [
                    s.replace(alt_char, CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"])
                    for s in texts
                ]
        texts = [s if ORIG_POST_CHAR in s else ORIG_POST_CHAR + s for s in texts]
    results = {}

    selector_inputs = texts
    if GLOBAL_DEBUG:
        print(f"passing to predict_select: {selector_inputs}")
    selector_inputs = pd.DataFrame(
        {
            "selector_input": selector_inputs,
            "prompt_finalchar": [
                ORIG_POST_CHAR_CHINESE for _ in range(len(selector_inputs))
            ],
        }
    )
    selection_results = predict_select(
        selector_inputs, debug=True, override_disable_forumlike=True
    )
    results["selection_proba"] = [float(p) for p in selection_results]

    selector_inputs = pd.DataFrame(
        {"selector_input": [final_munge_after_neural(s) for s in texts]}
    )
    sentiment_results = predict_sentiment(selector_inputs, debug=True)
    results["sentiment_logit_diffs"] = [float(p) for p in sentiment_results]
    show_note_probas(
        texts,
        probas=results["selection_proba"],
        sentiment_logit_diffs=results["sentiment_logit_diffs"],
    )

    print(f"texts: {texts}\nresults: {results}\n")

    return results


def poll_no_capture(
    capture_ident=None,
    dummy=False,
    ports=[
        bridge_service_port,
    ],
    routes=[
        "pollgenerator",
    ],
):
    global RESULT_STACK
    global model_name
    if model_name == "1558M":
        raise ValueError("don't use base gpt2 for AR, rob...")

    for port, route in zip(ports, routes):
        r = requests.post(
            f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
            json={"results": RESULT_STACK if not dummy else {}},
        )

        PROMPT_STACK = r.json()

        if (port, route) == (ports[0], routes[0]):
            # delete saved presents for finished jobs
            for k in RESULT_STACK.keys():
                if k not in PROMPT_STACK and "prompt_for_neural" in RESULT_STACK[k]:
                    generator_model.done_writing(k["prompt_for_neural"])

            # clean out already used results
            RESULT_STACK = {k: v for k, v in RESULT_STACK.items() if k in PROMPT_STACK}

        # print(f"got prompt stack: {PROMPT_STACK}")

        for prompt_id, data in PROMPT_STACK.items():
            print("generating...")
            if data["type"] == "answer":
                RESULT_STACK[prompt_id] = serve_answer(data)
            elif data["type"] == "textpost":
                RESULT_STACK[prompt_id] = serve_textpost(data)
            elif data["type"] == "raw_select":
                RESULT_STACK[prompt_id] = serve_raw_select(data)

            sampling_info = {
                "MIRO": MIRO,
                "MIRO_LR": MIRO_LR,
                "MIRO_ONLY_ON_CONTINUE": MIRO_ONLY_ON_CONTINUE,
                "length": length,
                "T": temperature,
                "p": top_p,
                "chop_lowest": chop_lowest,
                "chop_highest": chop_highest,
                "pre_continue_length": pre_continue_length,
                "pre_continue_T": pre_continue_temperature,
                "pre_continue_p": pre_continue_top_p,
            }

            model_info = {
                "model_name": model_name,
                "ckpt_select": ckpt_select,
                "ckpt_sentiment": ckpt_sentiment,
                "hparams_select": {
                    k: v
                    for k, v in selector_est.hparams_select_train_.values().items()
                    if k not in {"dtype", "adapt_layers"}
                },
                "hparams_select_sentiment": {
                    k: v
                    for k, v in sentiment_est.hparams_select_train_.values().items()
                    if k not in {"dtype", "adapt_layers"}
                },
                "sampling_info": sampling_info,
            }
            RESULT_STACK[prompt_id]["model_info"] = model_info

        if len(PROMPT_STACK) > 0:
            r = requests.post(
                f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
                json={"results": RESULT_STACK if not dummy else {}},
            )
            time.sleep(1)


def loop_poll_no_capture(
    period=60,
    capture_ident=None,
    dummy=False,
    ports=[
        bridge_service_port,
    ],
    routes=[
        "pollgenerator",
    ],
):
    global RESULT_STACK
    while True:
        try:
            poll_no_capture(
                capture_ident=capture_ident, dummy=dummy, ports=ports, routes=routes
            )
        except Exception as e:
            print(f"{type(e)}: {e}")
            time.sleep(period * 10)
        if len(RESULT_STACK) == 0 or dummy:
            time.sleep(period)
