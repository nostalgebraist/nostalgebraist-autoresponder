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
import tflex

import model
import sample
import encoder
from load_dataset import load_dataset, Sampler

from flask import Flask, escape, request, jsonify

from autoresponder_config import *
from autoresponder_static import *
from autoresponder_static_v8 import *

drivedir = "/content/drive/MyDrive/nostalgebraist-autoresponder/"
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

    try:
        print(f"trying to load from {local_gdrive_path}...")
        return load_fn(path=local_gdrive_path, retries=False, **kwargs)
    except (OSError, FileNotFoundError, KeyError):
        print(f"falling back to {local_gs_path}...")

        enclosing_dir = local_gs_path.rpartition("/")[0]
        os.makedirs(enclosing_dir, exist_ok=True)

        enclosing_dir_exists = os.path.exists(enclosing_dir)
        target_exists = os.path.exists(local_gs_path)

        print(f"enclosing dir {enclosing_dir} exists?: {enclosing_dir_exists}")
        print(f"target {local_gs_path} exists?: {target_exists}")

        if not target_exists:
            print(f"downlading from gs...")
            subprocess.check_output(gs_command, shell=True)
        try:
            return load_fn(path=local_gs_path, retries=retries, **kwargs)
        except:
            print(f"downlading from gs...")
            subprocess.check_output(gs_command, shell=True)
            return load_fn(path=local_gs_path, retries=retries, **kwargs)


def load_encoder_only(path, retries=False):  # ignored
    if path.endswith("vocab.bpe"):
        enclosing_dir = path.rpartition("/")[0]
        path = enclosing_dir
    enc = encoder.get_encoder_from_path(path, eot_workaround=EOT_WORKAROUND)
    return enc


def load_data_sampler(path, retries=False):  # ignored
    chunks = load_dataset(enc, path, 50000)
    data_sampler = Sampler(chunks)
    return data_sampler


enc = load_from_gdrive_with_gs_fallback(
    load_fn=load_encoder_only,
    relative_path=os.path.join("models", model_name, "vocab.bpe"),
    gs_command=gs_command_get_encoder,
)


data_sampler = load_from_gdrive_with_gs_fallback(
    load_fn=load_data_sampler,
    relative_path=dataset,
    gs_command=gs_command_get_dataset,
)


def make_session(reset=True):
    if reset:
        tf.reset_default_graph()
    sess = tflex.Session()

    with sess.as_default():
        context = tf.placeholder(tf.int32, [batch_size, None])
        sample_pasts = tf.placeholder(
            tf.float32, model.past_shape(hparams=hparams, batch_size=batch_size)
        )

        mirostat_target = tf.placeholder(shape=[], dtype=tf.float32)
        mirostat_lr = tf.placeholder(shape=[], dtype=tf.float32)
        mirostat_mu_from_past = tf.placeholder(
            tf.float32,
            [
                batch_size,
            ],
        )

        # TODO: DRY
        output_with_presents = sample.sample_sequence(
            stop_at_EOT=True,
            better_length=better_length,
            eot_workaround=EOT_WORKAROUND,
            enc=enc,
            hparams=hparams,
            length=length,
            start_token=start_token,
            context=context,
            batch_size=batch_size,
            temperature=pre_continue_temperature,
            top_k=pre_continue_top_k,
            top_p=pre_continue_top_p,
            middle_p=pre_continue_middle_p,
            chop_lowest=pre_continue_chop_lowest,
            chop_highest=pre_continue_chop_highest,
            return_presents=True,
            pasts=sample_pasts,
            mirostat=pre_continue_mirostat,
            mirostat_surprise_target=mirostat_target,
            mirostat_lr=mirostat_lr,
            mirostat_trunc=MIRO_TRUNC,
            mirostat_v2=MIRO_V2,
            disable_prints=True,
        )
        # TODO: DRY
        continue_output = sample.sample_sequence(
            stop_at_EOT=True,
            better_length=better_length,
            eot_workaround=EOT_WORKAROUND,
            enc=enc,
            hparams=hparams,
            length=length,
            start_token=start_token,
            context=context,
            batch_size=batch_size,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            middle_p=middle_p,
            chop_lowest=chop_lowest,
            chop_highest=chop_highest,
            return_presents=True,
            pasts=sample_pasts,
            mirostat=MIRO,
            mirostat_surprise_target=mirostat_target,
            mirostat_mu_init=mirostat_mu_from_past,
            mirostat_lr=mirostat_lr,
            mirostat_trunc=MIRO_TRUNC,
            mirostat_v2=MIRO_V2,
            disable_prints=True,
        )
        manual_presents_op = model.model(hparams=hparams, X=context)["present"]

    return (
        sess,
        context,
        sample_pasts,
        output_with_presents,
        continue_output,
        manual_presents_op,
        mirostat_target,
        mirostat_lr,
        mirostat_mu_from_past,
    )


(
    sess,
    context,
    sample_pasts,
    output_with_presents,
    continue_output,
    manual_presents_op,
    mirostat_target,
    mirostat_lr,
    mirostat_mu_from_past,
) = make_session()


def load_ckpt_with_retries(path, session, retries=True):
    enclosing_dir = path.rpartition("/")[0]
    ckpt = tflex.latest_checkpoint(enclosing_dir)
    if ckpt is None:
        raise FileNotFoundError
    saver = tflex.Saver()

    load_done = False
    while not load_done:
        try:
            with session.as_default():
                print(f"restoring checkpoint: {ckpt}")
                saver.restore(session, ckpt)
                load_done = True
        except Exception as e:
            if retries:
                print(f"encountered {e}, retrying...")
            else:
                raise e


load_from_gdrive_with_gs_fallback(
    load_fn=load_ckpt_with_retries,
    relative_path=os.path.join("models", model_path),
    gs_command=gs_command_get_model,
    session=sess,
)


def get_prompted_continuation(
    prompt: str,
    continue_if_cut_off=False,
    max_continue_steps=MAX_CONTINUE_STEPS,
    max_continue_tokens=MAX_CONTINUE_TOKENS,
    verbose=False,
    override_disable_forumlike=False,
    mirotarg=None,  # TODO: allow vary across batch, add noise inside this fn
    mirolr=MIRO_LR,
    forced_tags_string=None,
    write_fic_override=False,
    startup_presents=None,
):
    if mirotarg is None:
        mirotarg = np.random.choice(MIRO_TARGET_ALL)
    mu_init_scale = 1.0 if MIRO_V2 else 2.0

    if GLOBAL_DEBUG:
        print(f"in get_prompted_continuation, got prompt: {repr(prompt)}")
    raw_text = final_munge_before_neural(
        prompt,
        override_disable_forumlike=override_disable_forumlike,
        left_strip_newline=SELECTOR_LEFT_STRIP_NEWLINE_IN_FORUMLIKE,
        forced_tags_string=forced_tags_string,
        write_fic_override=write_fic_override,
    )
    raw_text = raw_text.replace(EOT_FULL, "")
    if EOT_PREPEND:
        raw_text = EOT_FULL + raw_text
    if GLOBAL_DEBUG:
        print(f"get_prompted_continuation, using prompt (munged): {repr(raw_text)}")
    context_tokens = enc.encode(raw_text)

    if better_length:
        max_context_size = length - required_continuation_room
    else:
        max_context_size = max_ctx_fits_on_gpu - length
    if len(context_tokens) > max_context_size:
        orig_len = len(context_tokens)
        context_tokens = context_tokens[-(max_context_size):]
        print(
            f"truncated {orig_len} to {len(context_tokens)}, max_context_size={max_context_size}"
        )
    else:
        print(
            f"{len(context_tokens)} tokens can fit in max_context_size {max_context_size}"
        )

    token_start_ix = len(context_tokens)

    batch_context_tokens = [context_tokens for _ in range(batch_size)]
    continuations = [[raw_text] for _ in batch_context_tokens]
    continuations_tokens = [[context_tokens] for _ in range(batch_size)]
    is_repeating = [False for _ in batch_context_tokens]
    is_not_finished = [True for _ in batch_context_tokens]
    generated = 0
    tokens_generated = 0
    this_batch_continue_steps = 0

    first_step_with_miro = 1 if MIRO_ONLY_ON_CONTINUE else 0

    done = False
    recompute_presents = False
    if startup_presents is None:
        print("computing startup presents")
        startup_presents = sess.run(
            manual_presents_op,
            feed_dict={context: [bct[:-1] for bct in batch_context_tokens]},
        )
    presents = startup_presents

    miromu = None
    mirosurprises, miroks = None, None
    while not done:
        recompute_presents = (token_start_ix >= max_context_size) or (presents is None)
        with sess.as_default():
            if recompute_presents:
                print("recomputing presents")
                presents = sess.run(
                    manual_presents_op,
                    feed_dict={context: [bct[:-1] for bct in batch_context_tokens]},
                )
                if this_batch_continue_steps >= first_step_with_miro:
                    if miromu is None:
                        miromu = mu_init_scale * mirotarg * np.ones((batch_size,))
                    print(f"miromu on entry: {miromu}")
                    sample_output_dict = sess.run(
                        continue_output,  # continue_output_no_presents,
                        feed_dict={
                            context: batch_context_tokens,
                            mirostat_target: mirotarg,
                            mirostat_lr: mirolr,
                            mirostat_mu_from_past: miromu,
                            sample_pasts: presents,  # !
                        },
                    )
                else:
                    sample_output_dict = sess.run(
                        output_with_presents,  # output,
                        feed_dict={
                            context: batch_context_tokens,
                            mirostat_target: mirotarg,
                            mirostat_lr: mirolr,
                            sample_pasts: presents,  # !
                        },
                    )
            else:
                print("using saved presents")
                if this_batch_continue_steps >= first_step_with_miro:
                    if miromu is None:
                        miromu = mu_init_scale * mirotarg * np.ones((batch_size,))
                    print(f"miromu on entry: {miromu}")
                    sample_output_dict = sess.run(
                        continue_output,
                        feed_dict={
                            context: batch_context_tokens,
                            sample_pasts: presents,
                            mirostat_target: mirotarg,
                            mirostat_lr: mirolr,
                            mirostat_mu_from_past: miromu,
                        },
                    )
                else:
                    sample_output_dict = sess.run(
                        output_with_presents,
                        feed_dict={
                            context: batch_context_tokens,
                            sample_pasts: presents,
                            mirostat_target: mirotarg,
                            mirostat_lr: mirolr,
                        },
                    )
        sample_output_dict["tokens"] = sample_output_dict["tokens"][:, token_start_ix:]
        sample_output_dict["presents"] = sample_output_dict["presents"][
            ..., -(max_context_size - 1) :, :
        ]
        out, presents = sample_output_dict["tokens"], sample_output_dict["presents"]

        if mirosurprises is None or (this_batch_continue_steps == first_step_with_miro):
            mirosurprises = sample_output_dict["mirostat_surprises"]
            miroks = sample_output_dict["mirostat_ks"]
        else:
            mirosurprises = np.concatenate(
                [mirosurprises, sample_output_dict["mirostat_surprises"]], axis=1
            )
            miroks = np.concatenate([miroks, sample_output_dict["mirostat_ks"]], axis=1)

        print(f"miromu before setting: {miromu}")
        if this_batch_continue_steps >= first_step_with_miro:
            miromu = sample_output_dict["mirostat_mus"][:, -1]
            print(f"miromu after setting: {miromu}")

        miroks = np.clip(miroks, a_min=None, a_max=hparams.n_vocab)

        miro_avg_surprises = np.mean(mirosurprises, axis=1)
        miro_median_ks = np.median(miroks, axis=1)
        miro_mean_ks = np.mean(miroks, axis=1)

        tokens_generated += len(out[0])
        for i in range(batch_size):
            generated += 1
            text = enc.decode(out[i])

            continuations[i].append(text)
            continuations_tokens[i].append(out[i])

            if (len(set(out[i])) >= 0.2 * len(out[i])) and not is_repeating[i]:
                is_repeating[i] = False
            else:
                print(f"{i} is repeating")
                is_repeating[i] = True

        if continue_if_cut_off:
            next_prompts = ["".join(subtexts) for subtexts in continuations]
            batch_context_tokens = [
                np.concatenate(ct)[-(max_context_size):] for ct in continuations_tokens
            ]

            bct_lens = [len(bct) for bct in batch_context_tokens]
            token_start_ix = min(bct_lens)
            while not all([bctl == token_start_ix for bctl in bct_lens]):
                print(
                    f"weirdness: not all elements of batch_context_tokens have same length"
                )
                for subtexts, nep, bct in zip(
                    continuations, next_prompts, batch_context_tokens
                ):
                    st_lens = [len(enc.encode(st)) for st in subtexts]
                    full_len = len(enc.encode("".join(subtexts)))
                    nep_len = len(enc.encode(nep))
                    bct_len = len(bct)
                    print(
                        f"st_lens={st_lens} | full_len={full_len} | nep_len={nep_len} | bct_len={bct_len}"
                    )
                batch_context_tokens = [
                    bct[:token_start_ix] for bct in batch_context_tokens
                ]
                bct_lens = [len(bct) for bct in batch_context_tokens]
                token_start_ix = min(bct_lens)

            next_prompts_contonly = [
                "".join(subtexts[1:]) for subtexts in continuations
            ]
            is_not_finished = [
                (
                    (eot_end_segment not in c)
                    and (
                        not contains_control_chars(
                            c, control_seg_config=CONTROL_SEG_CONFIG
                        )
                    )
                    and (len([char for char in c if char == T_CHAR]) < 2)
                    and not rep
                )
                for c, rep in zip(next_prompts_contonly, is_repeating)
            ]
            not_finished = [
                c for c, is_nf in zip(next_prompts_contonly, is_not_finished) if is_nf
            ]
            n_not_finished = len(not_finished)
            more_needed = n_not_finished > 0
            more_permitted = (this_batch_continue_steps < max_continue_steps) and (
                tokens_generated < max_continue_tokens
            )

            show_miro_logs = MIRO and (
                (not MIRO_ONLY_ON_CONTINUE)
                or this_batch_continue_steps >= first_step_with_miro
            )

            if show_miro_logs:
                for i in range(batch_size):
                    if i == 0:
                        print("\n")
                    finished_mark = "[ ]" if is_not_finished[i] else "[x]"
                    print(
                        f"{finished_mark} {i}: targeting surprise {mirotarg:.3f}, avg surprise {miro_avg_surprises[i]:.3f}, median k {miro_median_ks[i]:.1f}, mean k {miro_mean_ks[i]:.1f}"
                    )

                    if this_batch_continue_steps == first_step_with_miro:
                        print(
                            [
                                (j, enc.decode([tok]), mk, f"{ms:.3f}", f"{mmu:.3f}")
                                for j, (tok, mk, ms, mmu) in enumerate(
                                    zip(
                                        out[i],
                                        miroks[i, 1:],
                                        mirosurprises[i].tolist()[1:],
                                        sample_output_dict["mirostat_mus"][i].tolist()[
                                            1:
                                        ],
                                    )
                                )
                            ]
                        )
                    if i == batch_size - 1:
                        print()

            done = (not more_needed) or (not more_permitted)
            if not done:
                print("continuing within batch:")
                print(f"\t{n_not_finished}/{len(next_prompts)} unfinished")
                print(
                    f"\t{this_batch_continue_steps}/{max_continue_steps} continue steps used"
                )
                print(
                    f"\t{tokens_generated}/{max_continue_tokens} continue tokens generated"
                )
                print(
                    f"\tcontext tokens sizes: {[len(ct) for ct in batch_context_tokens]}"
                )

                if verbose:
                    print("Using prompts:")
                    for nep in not_finished:
                        print("\t" + "\n\t".join(wrap(nep, width=90)) + "\n")

                this_batch_continue_steps += 1
        else:
            done = True

    # cleanup
    continuations_ = []
    for subtexts, rep in zip(continuations, is_repeating):
        text = "".join(subtexts[1:])  # don't return prompt as part of these
        if rep:
            if GLOBAL_DEBUG:
                print(f"skipping because repeating:\n\n{repr(text)}\n\n")
            continue
        if not text.endswith(eot_end_segment) and eot_end_segment in text:
            text = text.split(eot_end_segment)[0] + eot_end_segment
        continuations_.append(text)

    return continuations_, startup_presents


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

    post = post.lstrip(
        ORIG_POST_CHAR
    )  # TODO: fix this in get_prompted_continuation_with_length_proportional_sampling
    parsed = {"post": post, "tags": tags}
    return parsed


def get_prompt_from_dataset(dataset):
    # TODO: deprecate this, it's no longer used for
    # anything non-trivial
    overrides = {}

    global data_sampler

    segment = "会"
    segments = []
    # while segment[0]=="会": # V3
    if FORUMLIKE:
        roll = np.random.rand()
        if roll < FORUMLIKE_REVIEW_PROB:
            look_for = CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"]
            overrides["v8_timestamp"] = ""
            overrides["v10_timestamp"] = ""
        elif roll < FORUMLIKE_REVIEW_PROB + FORUMLIKE_FIC_PROB:
            look_for = CONTROL_SEG_CONFIG["ORIG_FICTION_CHAR_FORUMLIKE"]
            overrides["v8_timestamp"] = ""
            overrides["v10_timestamp"] = ""
            overrides["tag_string_raw"] = "#original fiction"
        else:
            look_for = CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"]
        print(f"using look_for={repr(look_for)}")
    else:
        look_for = "翰"

    if V10:
        prompt = look_for
        return prompt, overrides

    while segment[: len(look_for)] != look_for:  # V4
        while len(segments) == 0:
            segments = enc.decode(data_sampler.sample(1024)).split("<|endoftext|>")[1:]
        segment = segments.pop()

    n_extra_tokens = 1
    if V8:
        n_extra_tokens = 0
    if EOT_WORKAROUND:
        if EOT_PREPEND:
            segment = "<|endoftext|>" + segment
            n_extra_tokens += 1
        context_tokens = enc.encode(segment)[
            : len(enc.encode(look_for)) + n_extra_tokens
        ]
        print(f"using context_tokens {context_tokens} = {enc.decode(context_tokens)}")
    else:
        context_tokens = enc.encode("endoftext|>" + segment)[
            : len(enc.encode("endoftext|>")) + 3
        ]
    prompt = enc.decode(context_tokens)

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
    prompt_from_dataset=False,
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

    if prompt_from_dataset:
        prompt, textpost_overrides = get_prompt_from_dataset(dataset)
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

    if GLOBAL_DEBUG:
        print(f"in basic_n_continuations, using prompt: {repr(prompt)}")
    continuations = []
    startup_presents = None
    while len(continuations) < N:
        print(f"\ncontinuing, have {len(continuations)} of {N}\n")

        this_batch_continuations, startup_presents = get_prompted_continuation(
            prompt,
            continue_if_cut_off=continue_if_cut_off,
            max_continue_steps=max_continue_steps,
            verbose=verbose,
            override_disable_forumlike=prompt_from_dataset
            or override_disable_forumlike,
            mirotarg=mirotarg,
            forced_tags_string=forced_tags_string,
            write_fic_override=write_fic_override,
            startup_presents=startup_presents,
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
        if prompt_from_dataset:
            continuation = prompt + continuation
            if EOT_PREPEND and continuation.startswith("<|endoftext|>"):
                continuation = continuation[len("<|endoftext|>") :]
            if FORUMLIKE and continuation.startswith(ORIG_POST_CHAR_CHINESE):
                continuation = CONTROL_SEG_CONFIG[
                    "ORIG_POST_CHAR_FORUMLIKE"
                ] + continuation.lstrip(ORIG_POST_CHAR_CHINESE)
        continuations_.append(continuation)
    continuations = continuations_

    return continuations, continuation_side_data


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


from model import *

SELECTION_CHAR = "<|endoftext|>"
SELECTION_TOK = enc.encode(SELECTION_CHAR)[-1]


def extract_selection_ix(tokens, extract_from):
    mask = tf.equal(tf.dtypes.cast(tokens, tf.int32), SELECTION_TOK)
    extracted_ragged = tf.ragged.boolean_mask(extract_from, mask)

    row_lengths = extracted_ragged.row_lengths()
    row_ixs = row_lengths - 1
    selection_ix = tf.stack(
        [tf.range(0, batch_size_for_h, dtype=tf.int64), row_ixs],
        axis=1,
    )

    extracted = tf.gather_nd(
        extracted_ragged.to_tensor(),
        selection_ix,
    )

    return {"extracted": extracted, "selection_ix": selection_ix}


def extract_selection_ix_position(tokens):
    return extract_selection_ix(tokens, tf.sort(tf.argsort(tokens)))


def model_activations(
    hparams,
    X,
    hparams_select,
    layer_nums: list,
    norm_layers_after: bool = False,
    add_position_emb_later_layers=False,
    past=None,
    past_select=None,
    scope="model",
    reuse=tf.AUTO_REUSE,
):
    activations = []
    h_names = []

    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, reuse=reuse, dtype=dtype):
        results = {}
        batch, sequence = shape_list(X)

        wpe = get_variable("wpe") or tf.get_variable(
            "wpe",
            [hparams.n_ctx, hparams.n_embd],
            initializer=tf.random_normal_initializer(stddev=0.01, dtype=dtype),
        )
        wte = get_variable("wte") or tf.get_variable(
            "wte",
            [hparams.n_vocab, hparams.n_embd],
            initializer=tf.random_normal_initializer(stddev=0.02, dtype=dtype),
        )
        past_length = 0 if past is None else tf.shape(past)[-2]
        position_emb = tf.gather(wpe, positions_for(X, past_length))
        h = tf.gather(wte, X) + position_emb

        # Transformer
        presents = []
        pasts = (
            tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        )
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present, present_adapt = block(
                h, "h%d" % layer, past=past, past_adapt=past, hparams=hparams
            )
            presents.append(present)
            if layer in layer_nums:
                h_name = f"h{layer}"
                print(f"{h_name} found")
                h_names.append(h_name)
                if add_position_emb_later_layers:
                    print(f"adding position emb at {h_name}")
                h_to_use = h + position_emb if add_position_emb_later_layers else h
                activations.append(h_to_use)

        results["present"] = tf.stack(presents, axis=1)
        h = norm(h, "ln_f", hparams=hparams)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch * sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results["logits"] = logits

        # activations
        if norm_layers_after:
            activations = [
                norm(act, f"ln_after_{act_name}", hparams=hparams_select)
                for act_name, act in zip(h_names, activations)
            ]

        results["activations"] = list(zip(h_names, activations))

        return results


def get_initializer(hparams, scope):
    initializer = tf.random_normal_initializer
    if hparams.get("orth_init"):
        print(f"orth init in scope {scope}")
        initializer = tf.compat.v1.orthogonal_initializer
    return initializer


def norm_orig(x, scope, *, axis=-1, epsilon=1e-5, hparams=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    dtype = hparams.dtype if hparams else tf.float32
    n_state = x.shape[-1].value
    with tf.variable_scope(scope, dtype=dtype):
        g = get_variable("g") or tf.get_variable(
            "g", [n_state], initializer=tf.constant_initializer(1, dtype=dtype)
        )
        b = get_variable("b") or tf.get_variable(
            "b", [n_state], initializer=tf.constant_initializer(0, dtype=dtype)
        )
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x * g + b
        return x


def norm(x, scope, *, axis=-1, epsilon=1e-5, hparams=None):
    if hparams.get("scalenorm", False):
        dtype = hparams.dtype if hparams else tf.float32
        n_state = x.shape[-1].value
        with tf.variable_scope(scope, dtype=dtype):
            """https://arxiv.org/pdf/1910.05895.pdf"""
            g = get_variable("g") or tf.get_variable(
                "g", [1], initializer=tf.constant_initializer(1, dtype=dtype)
            )
            s = tf.reduce_mean(tf.square(x), axis=axis, keepdims=True)
            x = g * (x / tf.rsqrt(s + epsilon))
            # x = g*(x/tf.norm(x, ord=2, axis=axis, keepdims=True))
            return x
    else:
        return norm_orig(x, scope, axis=axis, epsilon=epsilon, hparams=hparams)


def conv1d(x, scope, nf, *, w_init_stdev=0.02, hparams=None):
    dtype = hparams.dtype if hparams else tf.float32

    initializer = get_initializer(hparams, scope)
    with tf.variable_scope(scope, dtype=dtype):
        *start, nx = shape_list(x)
        w = get_variable("w") or tf.get_variable(
            "w", [1, nx, nf], initializer=initializer(w_init_stdev, dtype=dtype)
        )
        b = get_variable("b") or tf.get_variable(
            "b", [nf], initializer=tf.constant_initializer(0, dtype=dtype)
        )
        c = tf.reshape(
            tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b,
            start + [nf],
        )
        return c


# this is a copy/paste -- we need to redefine "attn" so the "conv1d" defn'd above
# is used


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert (
            past.shape.ndims == 5
        )  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(65500 if w.dtype != tf.float32 else 1e10, w.dtype) * (1 - b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        w = dropout(w, hparams.attn_dropout)
        a = tf.matmul(w, v)
        return a

    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        c = conv1d(x, "c_attn", n_state * 3, hparams=hparams)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, "c_proj", n_state, hparams=hparams)
        a = dropout(a, hparams.res_dropout)
        return a, present


def attn_only_block(x, scope, *, past, hparams, do_input_norm=True):
    dtype = hparams.dtype if hparams else tf.float32
    do_resid = hparams.do_resid if hparams else True
    print(f"do_resid: {do_resid}")
    print(f"do_input_norm: {do_input_norm}")
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value

        if do_input_norm:
            norm_in_fn = norm if hparams.get("scalenorm_in", False) else norm_orig
            x_attn_in = norm_in_fn(x, "ln_1", hparams=hparams)
        else:
            x_attn_in = x

        a, present = attn(x_attn_in, "attn", nx, past=past, hparams=hparams)
        if do_resid:
            x = x + a
        else:
            x = a

        return x, present


def mlp_no_proj(x, scope, n_state, *, hparams, is_expansion=False):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, "c_fc", n_state, w_init_stdev=0.02, hparams=hparams))
        h = dropout(h, hparams.res_dropout)
        return h


def mlp_acti_dropout(
    x, scope, n_state, *, hparams, w_init_stdev=1, n_final=None, dropout_final=True
):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        if n_final is None:
            n_final = nx
        h = gelu(conv1d(x, "c_fc", n_state, hparams=hparams, w_init_stdev=w_init_stdev))
        h = dropout(h, hparams.acti_dropout)
        h2 = conv1d(h, "c_proj", n_final, hparams=hparams, w_init_stdev=w_init_stdev)
        if dropout_final:
            h2 = dropout(h2, hparams.res_dropout)
        return h2


def selector(
    hparams,
    X,
    hparams_select,
    layer_nums: list,
    scope="model",
    select_scope="select",
    reuse=tf.AUTO_REUSE,
    norm_layers_after: bool = False,
    use_mlp: bool = True,
    resid_mlp: bool = True,
    direct_mlp: bool = False,
    mlp_proj=True,
    mlp_ratio=1,
    use_length_channel=False,
    use_length_channel_v2=False,
    add_position_emb_later_layers=False,
    add_prompt_cont_embs=False,
    prompt_end_ntoks=None,
    norm_final_output=True,
):
    results = {}

    activations_ = model_activations(
        hparams=hparams,
        hparams_select=hparams_select,
        X=X,
        layer_nums=layer_nums,
        norm_layers_after=norm_layers_after,
        add_position_emb_later_layers=add_position_emb_later_layers,
        scope=scope,
        reuse=reuse,
    )["activations"]

    if add_prompt_cont_embs:
        with tf.variable_scope(select_scope, reuse=reuse):
            initializer = get_initializer(hparams_select, select_scope)
            prompt_cont_embs = tf.get_variable(
                "select_prompt_cont_embs",
                [2, hparams.n_embd],
                initializer=initializer(0.01, dtype=hparams.dtype),
            )

            positions = positions_for(X, 0)
            tiled_prompt_end_ntoks = tf.reshape(prompt_end_ntoks, (-1, 1))
            is_cont = tf.where(
                positions > tiled_prompt_end_ntoks,
                tf.ones_like(positions),
                tf.zeros_like(positions),
            )

            prompt_cont_embs_add = tf.gather(prompt_cont_embs, is_cont)

            activations = []
            for act_name, act in activations_:
                activations.append((act_name, act + prompt_cont_embs_add))
    else:
        activations = activations_

    hs_select = []
    with tf.variable_scope(select_scope, reuse=reuse, dtype=hparams_select.dtype):
        for act_name, act in activations:
            h_select, _ = attn_only_block(
                act,
                f"h_select_{act_name}",
                hparams=hparams_select,
                past=None,
                do_input_norm=(not norm_layers_after),
            )
            if norm_final_output:
                h_select = norm(
                    h_select,
                    f"ln_2_select_{act_name}",
                    hparams=hparams_select,
                )
            hs_select.append(h_select)

            h_select_in = tf.concat(hs_select, axis=-1)

            h_select_in_at_selection_ix = extract_selection_ix(X, h_select_in)[
                "extracted"
            ]
            selection_ix_position = tf.cast(
                tf.reshape(extract_selection_ix_position(X)["extracted"], [-1, 1]),
                tf.float32,
            )

        if use_mlp:
            nx = h_select_in_at_selection_ix.shape[-1].value
            if mlp_proj:
                n_final = 2 if direct_mlp else None
                m = mlp_acti_dropout(
                    h_select_in_at_selection_ix,
                    "select_mlp",
                    int(mlp_ratio * nx),
                    hparams=hparams_select,
                    n_final=n_final,
                    dropout_final=(not direct_mlp),
                )
            else:
                m = mlp_no_proj(
                    h_select_in_at_selection_ix,
                    "select_mlp",
                    nx,
                    hparams=hparams_select,
                )
            if direct_mlp:
                pass
            elif resid_mlp:
                h_select_in_at_selection_ix = m + h_select_in_at_selection_ix
            else:
                h_select_in_at_selection_ix = m

        w_select = get_variable("w_select")
        if w_select is None:
            initializer = get_initializer(hparams_select, scope)
            w_select = tf.get_variable(
                "w_select",
                [len(layer_nums) * hparams.n_embd, 2],
                initializer=initializer(0.02, dtype=hparams.dtype),
            )

        b_select = get_variable("b_select")
        if b_select is None:
            b_select = tf.get_variable(
                "b_select",
                [2],
                initializer=tf.constant_initializer(0, dtype=hparams.dtype),
            )

        select_logits = tf.matmul(h_select_in_at_selection_ix, w_select) + b_select

        if direct_mlp and use_mlp:
            select_logits = select_logits + m

        if use_length_channel:
            if use_length_channel_v2:
                w_select_length_linear = get_variable("w_select_length_linear")
                if w_select_length_linear is None:
                    w_select_length_linear = tf.get_variable(
                        "w_select_length_linear",
                        [1, 2],
                        initializer=tf.constant_initializer(0.0, dtype=hparams.dtype),
                    )

                w_select_length_log = get_variable("w_select_length_log")
                if w_select_length_log is None:
                    w_select_length_log = tf.get_variable(
                        "w_select_length_log",
                        [1, 2],
                        initializer=tf.constant_initializer(0.0, dtype=hparams.dtype),
                    )

                select_logits = select_logits + tf.matmul(
                    selection_ix_position, w_select_length_linear
                )
                select_logits = select_logits + tf.matmul(
                    tf.log(selection_ix_position), w_select_length_log
                )

            else:
                w_select_length = get_variable("w_select_length")
                if w_select_length is None:
                    w_select_length = tf.get_variable(
                        "w_select_length",
                        [1, 2],
                        initializer=tf.constant_initializer(0.0, dtype=hparams.dtype),
                    )
                select_logits = select_logits * (
                    1 + tf.matmul(1 + tf.log(selection_ix_position), w_select_length)
                )

    results["logits_select"] = select_logits

    return results


batch_size_for_h = batch_size


def load_variables_with_retries(
    path, var_list, session, multi_calib=False, retries=True
):
    done = False
    tries = 0
    while not done:
        try:
            print(f"loading from {path}")
            tflex.load_variables(path, session=sess, var_list=var_list)
            done = True
        except Exception as e:
            if not retries:
                raise e
            if tries > 5:
                break
            print(f"encountered {e}, retrying...")
            tries += 1

    display(var_list)

    if multi_calib:
        with open(path.rpartition("/")[0] + "/lr_calib_resp.pkl", "rb") as f:
            lr_calib_resp = pickle.load(f)
        with open(path.rpartition("/")[0] + "/lr_calib_orig.pkl", "rb") as f:
            lr_calib_orig = pickle.load(f)
    else:
        with open(path.rpartition("/")[0] + "/lr_calib.pkl", "rb") as f:
            lr_calib = pickle.load(f)
            lr_calib_resp = lr_calib
            lr_calib_orig = lr_calib
    return lr_calib_resp, lr_calib_orig


def load_selector_metadata(path, retries=False):  # ignored
    with open(path, "r") as f:
        selector_metadata = json.load(f)
    return selector_metadata


if SELECT_VIA_GENERATOR:
    metadata_filename = ckpt_select.rpartition("/")[0] + "/metadata.json"
    selector_metadata = load_from_gdrive_with_gs_fallback(
        load_fn=load_selector_metadata,
        relative_path=metadata_filename,
        gs_command=gs_command_get_selector_metadata,
    )
    select_scope = selector_metadata["select_scope"]

    hparams_select = HParams(
        n_vocab=hparams.n_vocab,
        n_ctx=hparams.n_ctx,
        n_embd=hparams.n_embd,
        n_head=selector_metadata["n_head"],
        n_layer=hparams.n_layer,
        res_dropout=0,
        attn_dropout=0,
        acti_dropout=0,
        dtype=tf.float32,
        do_resid=do_resid,
        orth_init=True,
    )

    with sess.as_default():
        context_for_h = tf.placeholder(tf.int32, [batch_size_for_h, None])
        prompt_end_ntoks = tf.placeholder(
            tf.int32, [batch_size_for_h], name="select_prompt_end_ntoks"
        )

        selection_step = selector(
            select_scope=select_scope,
            hparams=hparams,
            hparams_select=hparams_select,
            X=context_for_h,
            layer_nums=layer_nums,
            norm_layers_after=norm_layers_after,
            use_mlp=use_mlp,
            resid_mlp=resid_mlp,
            direct_mlp=direct_mlp,
            mlp_proj=mlp_proj,
            mlp_ratio=mlp_ratio,
            use_length_channel=use_length_channel,
            use_length_channel_v2=use_length_channel_v2,
            add_position_emb_later_layers=add_position_emb_later_layers,
            add_prompt_cont_embs=add_prompt_cont_embs,
            prompt_end_ntoks=prompt_end_ntoks,
            norm_final_output=norm_final_output,
        )

        select_logits = selection_step["logits_select"]

    old_names = {}
    var_list = [
        var
        for var in tf.trainable_variables()
        if select_scope in var.name and not any([on in var.name for on in old_names])
    ]

    lr_calib_resp, lr_calib_orig = load_from_gdrive_with_gs_fallback(
        load_fn=partial(
            load_variables_with_retries,
            var_list=var_list,
            multi_calib=MULTI_LR_CALIB,
        ),
        relative_path=ckpt_select,
        gs_command=gs_command_get_selector,
        session=sess,
    )


if SENTIMENT_VIA_GENERATOR:
    sentiment_metadata_filename = ckpt_sentiment.rpartition("/")[0] + "/metadata.json"
    sentiment_metadata = load_from_gdrive_with_gs_fallback(
        load_fn=load_selector_metadata,
        relative_path=sentiment_metadata_filename,
        gs_command=gs_command_get_sentiment_metadata,
    )
    sentiment_select_scope = sentiment_metadata["select_scope"]

    hparams_select_sentiment = HParams(
        n_vocab=hparams.n_vocab,
        n_ctx=hparams.n_ctx,
        n_embd=hparams.n_embd,
        n_head=sentiment_metadata["n_head"],
        n_layer=hparams.n_layer,
        res_dropout=0,
        attn_dropout=0,
        acti_dropout=0,
        dtype=tf.float32,
        do_resid=do_resid,
        orth_init=True,
    )

    with sess.as_default():
        selection_step_sentiment = selector(
            select_scope=sentiment_select_scope,
            hparams=hparams,
            hparams_select=hparams_select_sentiment,
            X=context_for_h,
            layer_nums=layer_nums_sentiment,
            norm_layers_after=norm_layers_after,
            use_mlp=use_mlp_sentiment,
            resid_mlp=resid_mlp,
            mlp_proj=mlp_proj,
            use_length_channel=use_length_channel_sentiment,
            use_length_channel_v2=use_length_channel_v2_sentiment,
            norm_final_output=norm_final_output_sentiment,
        )

        sentiment_logits = selection_step_sentiment["logits_select"]

    var_list = [
        var for var in tf.trainable_variables() if sentiment_select_scope in var.name
    ]

    lr_calib_sentiment, _ = load_from_gdrive_with_gs_fallback(
        load_fn=partial(
            load_variables_with_retries,
            var_list=var_list,
            multi_calib=False,
        ),
        relative_path=ckpt_sentiment,
        gs_command=gs_command_get_sentiment,
        session=sess,
    )


if SELECT_VIA_GENERATOR:
    import scipy.special

    def single_batch_predict_select(
        data_batch,
        lr_calib,
        threshold=0.5,
        debug=False,
        truncate_at_right=TRUNCATE_AT_RIGHT,
        length_=length_select,
        override_disable_forumlike=False,
    ):
        if len(data_batch) != batch_size_for_h:
            raise ValueError("badlength")
        batch_context = []
        if add_prompt_cont_embs:
            batch_prompt_end_ntoks = data_batch.prompt_end_ntoks.values
        for text in data_batch.selector_inputs:
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

            if truncate_at_right:
                batch_context.append(
                    enc.encode(text)[-(length_ - 1) :] + [SELECTION_TOK]
                )
            else:
                batch_context.append(
                    enc.encode(text)[: (length_ - 1)] + [SELECTION_TOK]
                )

            if debug:
                print(
                    f"in single_batch_predict_select, predicting on:\n{enc.decode(batch_context[-1])}\n"
                )
        max_tokens = max([len(toks) for toks in batch_context])
        batch_context_ = [
            toks + [0 for _ in range(max_tokens - len(toks))] for toks in batch_context
        ]
        batch_context = batch_context_

        feed_dict = {}
        feed_dict[context_for_h] = batch_context

        if add_prompt_cont_embs:
            shift = max(0, max_tokens - length)
            batch_prompt_end_ntoks = batch_prompt_end_ntoks - shift
            batch_prompt_end_ntoks[batch_prompt_end_ntoks < 0] = 0

            feed_dict[prompt_end_ntoks] = batch_prompt_end_ntoks

        with sess.as_default():
            logits = sess.run(select_logits, feed_dict=feed_dict)

        if SELECTOR_LR_CALIB_INPUT == "logits":
            probs = lr_calib.predict_proba(logits)[:, 1]
        elif SELECTOR_LR_CALIB_INPUT == "logit_diff":
            probs = lr_calib.predict_proba(logits[:, 1:] - logits[:, :1])[:, 1]
        results = {"logits": logits, "probs": probs, "preds": probs > threshold}
        return results

    def predict_select(
        data,
        lr_calib,
        threshold=0.5,
        debug=False,
        length_=length_select,
        override_disable_forumlike=False,
    ):
        batches = []

        for i in range(0, len(data), batch_size_for_h):
            data_batch = data.iloc[i : i + batch_size_for_h]

            n_needed = len(data_batch)
            if n_needed < batch_size_for_h:
                data_batch = pd.concat(
                    [data_batch]
                    + (batch_size_for_h - n_needed) * [data_batch.iloc[-1:, :]],
                    ignore_index=True,
                )
            # while len(batch) != batch_size_for_h:
            #   batch = batch + [batch[-1] for _ in range(batch_size_for_h - len(batch))]
            batches.append(data_batch)

        batch_results = [
            single_batch_predict_select(
                data_batch,
                lr_calib,
                threshold=threshold,
                debug=debug,
                length_=length_,
                override_disable_forumlike=override_disable_forumlike,
            )
            for data_batch in batches
        ]

        result_keys = batch_results[0].keys()
        results = {
            k: np.concatenate([br[k] for br in batch_results])[: len(data)]
            for k in result_keys
        }

        return results


if SENTIMENT_VIA_GENERATOR:

    def single_batch_predict_sentiment(
        text_batch, length_=length_sentiment, debug=False
    ):
        if len(text_batch) != batch_size_for_h:
            raise ValueError("badlength")
        batch_context = []
        for text in text_batch:
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
            # if FORUMLIKE:
            #   text = substitute_forumlike(text, shuffle=False, infer_first=False)
            text = re.sub(r"\<.*?\>", "", text)  # sentiment-specific

            if EOT_PREPEND:
                # do this after the regex so it sticks
                if (not SELECTOR_EOT_PREPEND) and text.startswith(EOT_FULL):
                    text = text[len(EOT_FULL) :]
                if SELECTOR_EOT_PREPEND and (not text.startswith(EOT_FULL)):
                    text = EOT_FULL + text

            batch_context.append(enc.encode(text)[:length_] + [SELECTION_TOK])
            if debug:
                print(
                    f"in single_batch_predict_sentiment, predicting on:\n{enc.decode(batch_context[-1])}\n"
                )
        max_tokens = max([len(toks) for toks in batch_context])
        batch_context_ = [
            toks + [0 for _ in range(max_tokens - len(toks))] for toks in batch_context
        ]
        batch_context = batch_context_

        with sess.as_default():
            logits = sess.run(
                sentiment_logits, feed_dict={context_for_h.name: batch_context}
            )

        # probs = scipy.special.softmax(logits, axis=1)[:, 1]
        # results = {"logits": logits, "probs": probs, "logit_diffs": logits[:, 1] - logits[:, 0]}
        logit_diffs_raw = logits[:, 1:] - logits[:, :1]
        logit_diffs_calib = lr_calib_sentiment.predict(logit_diffs_raw)

        print(f"logit_diffs_raw avg={logit_diffs_raw.mean():.2f}")
        print(f"logit_diffs_calib avg={logit_diffs_calib.mean():.2f}")
        print(f"diff avg={logit_diffs_calib.mean()-logit_diffs_raw.mean():.2f}")

        results = {"logit_diffs": logit_diffs_calib, "logit_diffs_raw": logit_diffs_raw}
        return results

    def predict_sentiment(data, length_=length_sentiment, debug=False):
        texts = data.selector_inputs.values.tolist()
        batches = []

        for i in range(0, len(texts), batch_size_for_h):
            batch = texts[i : i + batch_size_for_h]

            while len(batch) != batch_size_for_h:
                batch = batch + [
                    batch[-1] for _ in range(batch_size_for_h - len(batch))
                ]
            batches.append(batch)

        batch_results = [
            single_batch_predict_sentiment(batch, length_=length_, debug=debug)
            for batch in batches
        ]

        result_keys = batch_results[0].keys()
        results = {
            k: np.concatenate([br[k] for br in batch_results])[: len(texts)]
            for k in result_keys
        }

        return results


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

    if kwargs.get("V5"):
        continuations, continuation_side_data = basic_n_continuations(
            prompt,
            N=kwargs["best_of"],
            avoid_if_under=avoid_if_under,
            avoid_half_if_under=avoid_half_if_under,
            avoid_if_cut_off=avoid_if_cut_off,
            prompt_from_dataset=kwargs.get("prompt_from_dataset"),
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

        if SELECT_VIA_GENERATOR:
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
                            "selector_inputs": [
                                join_time_sidechannel(s, alt_ts)
                                for s in selector_inputs
                            ]
                        }
                    )
                    entry_selection_results = predict_select(
                        alt_selector_inputs, lr_calib_resp, debug=True
                    )
                    listkey = f"alt_selection_proba__{alt_ts.replace(' ', '_')}"
                    parsed[listkey] = [
                        float(p) for p in entry_selection_results["probs"]
                    ]

            selector_inputs = [
                join_time_sidechannel(s, relevant_timestamp) for s in selector_inputs
            ]
            selector_inputs = pd.DataFrame({"selector_inputs": selector_inputs})
            if GLOBAL_DEBUG:
                print(f"passing to predict_select: {selector_inputs}")
            selection_results = predict_select(
                selector_inputs,
                lr_calib_resp,
                debug=True,
                override_disable_forumlike=override_disable_forumlike,
            )
            parsed["selection_proba"] = [float(p) for p in selection_results["probs"]]
            if SENTIMENT_VIA_GENERATOR:
                selector_inputs = pd.DataFrame(
                    {"selector_inputs": parsed["continuations"]}
                )
                sentiment_results = predict_sentiment(selector_inputs, debug=True)
                parsed["sentiment_logit_diffs"] = [
                    float(p) for p in sentiment_results["logit_diffs"]
                ]
                show_note_probas(
                    continuations,
                    probas=parsed["selection_proba"],
                    sentiment_logit_diffs=parsed["sentiment_logit_diffs"],
                )
            else:
                show_note_probas(continuations, probas=parsed["selection_proba"])
    else:
        kwargs = {k: v for k, v in kwargs.items() if k != "V5"}
        continuations = get_prompted_continuation_with_length_proportional_sampling(
            prompt, **kwargs
        )
        continuation = continuations[0]

        parsed = parse_continuation(continuation)

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

    if kwargs.get("V5"):
        try:
            continuations, continuation_side_data = basic_n_continuations(
                prompt,
                N=kwargs["best_of"],
                avoid_if_under=avoid_if_under,
                avoid_half_if_under=avoid_half_if_under,
                avoid_if_cut_off=avoid_if_cut_off,
                split_on_control_char=split_on_control_char,
                prompt_from_dataset=kwargs.get("prompt_from_dataset"),
                avoid_initial_blockquote=avoid_initial_blockquote,
                v8_timestamp=data.get("v8_timestamp"),
                v10_timestamp=data.get("v10_timestamp", ""),
                continue_if_cut_off=continue_if_cut_off,
            )
        except Exception as e:
            if EVEN_BETTER_LENGTH:
                raise (e)
            print(f"got {e}, trying without continue_if_cut_off")
            continuations, continuation_side_data = basic_n_continuations(
                prompt,
                N=kwargs["best_of"],
                avoid_if_under=avoid_if_under,
                avoid_half_if_under=avoid_half_if_under,
                avoid_if_cut_off=avoid_if_cut_off,
                split_on_control_char=split_on_control_char,
                prompt_from_dataset=kwargs.get("prompt_from_dataset"),
                avoid_initial_blockquote=avoid_initial_blockquote,
                v8_timestamp=data.get("v8_timestamp"),
                v10_timestamp=data.get("v10_timestamp", ""),
                continue_if_cut_off=False,
            )
        parsed = data.copy()
        parsed["continuations"] = [final_munge_after_neural(c) for c in continuations]
        parsed["mirotarg"] = [cd.get("mirotarg") for cd in continuation_side_data]

        if SELECT_VIA_GENERATOR:
            if FORUMLIKE:
                selector_inputs = [c for c in continuations]
                for alt_char in [
                    CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"],
                    CONTROL_SEG_CONFIG["ORIG_FICTION_CHAR_FORUMLIKE"],
                ]:
                    selector_inputs = [
                        s.replace(
                            alt_char, CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"]
                        )
                        for s in selector_inputs
                    ]
            else:
                selector_inputs = [A_CHAR + c for c in continuations]
            selector_inputs = pd.DataFrame({"selector_inputs": selector_inputs})
            if GLOBAL_DEBUG:
                print(f"passing to predict_select: {selector_inputs}")
            selection_results = predict_select(
                selector_inputs,
                lr_calib_orig,
                debug=True,
                override_disable_forumlike=True,
            )
            parsed["selection_proba"] = [float(p) for p in selection_results["probs"]]

            if SENTIMENT_VIA_GENERATOR:
                selector_inputs = pd.DataFrame(
                    {"selector_inputs": parsed["continuations"]}
                )
                sentiment_results = predict_sentiment(selector_inputs, debug=True)
                parsed["sentiment_logit_diffs"] = [
                    float(p) for p in sentiment_results["logit_diffs"]
                ]
                show_note_probas(
                    continuations,
                    probas=parsed["selection_proba"],
                    sentiment_logit_diffs=parsed["sentiment_logit_diffs"],
                )
            else:
                show_note_probas(continuations, probas=parsed["selection_proba"])
    else:
        kwargs = {k: v for k, v in kwargs.items() if k != "V5"}
        continuations = get_prompted_continuation_with_retries_for_length(
            prompt, **kwargs
        )
        continuation = continuations[0]

        parsed = parse_continuation(continuation)

    if GLOBAL_DEBUG:
        print(f"sending back: {parsed}")

    return parsed


def serve_raw_select(data):
    texts = data["texts"]

    # texts = [s.lstrip("翰") for s in texts]
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

    if SELECT_VIA_GENERATOR:
        selector_inputs = texts
        if GLOBAL_DEBUG:
            print(f"passing to predict_select: {selector_inputs}")
        selector_inputs = pd.DataFrame({"selector_inputs": selector_inputs})
        selection_results = predict_select(
            selector_inputs, lr_calib_orig, debug=True, override_disable_forumlike=True
        )
        results["selection_proba"] = [float(p) for p in selection_results["probs"]]

        if SENTIMENT_VIA_GENERATOR:
            selector_inputs = pd.DataFrame(
                {"selector_inputs": [final_munge_after_neural(s) for s in texts]}
            )
            sentiment_results = predict_sentiment(selector_inputs, debug=True)
            results["sentiment_logit_diffs"] = [
                float(p) for p in sentiment_results["logit_diffs"]
            ]
            show_note_probas(
                texts,
                probas=results["selection_proba"],
                sentiment_logit_diffs=results["sentiment_logit_diffs"],
            )
        else:
            show_note_probas(texts, probas=results["selection_proba"])

        print(f"texts: {texts}\nresults: {results}\n")

    return results


# TODO: DRY
def poll(
    capture_ident=None,
    dummy=False,
    ports=[
        bridge_service_port,
    ],
    routes=[
        "pollgenerator",
    ],
):
    with capture_output(True, True, False) as cap:
        global RESULT_STACK
        global model_name
        if model_name == "1558M":
            raise ValueError("don't use base gpt2 for AR, rob...")

        for port, route in zip(ports, routes):
            r = requests.post(
                f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/pollgenerator",
                json={"results": RESULT_STACK if not dummy else {}},
            )

            PROMPT_STACK = r.json()

            if (port, route) == (ports[0], routes[0]):
                RESULT_STACK = {
                    k: v for k, v in RESULT_STACK.items() if k in PROMPT_STACK
                }  # clean out already used results

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
                        for k, v in hparams_select.values().items()
                        if k not in {"dtype", "adapt_layers"}
                    },
                    "hparams_select_sentiment": {
                        k: v
                        for k, v in hparams_select_sentiment.values().items()
                        if k not in {"dtype", "adapt_layers"}
                    },
                    "sampling_info": sampling_info,
                }
                RESULT_STACK[prompt_id]["model_info"] = model_info

            # print("done generating for this poll")

            if len(PROMPT_STACK) > 0:
                r = requests.post(
                    f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/pollgenerator",
                    json={"results": RESULT_STACK if not dummy else {}},
                )
                time.sleep(1)

                if capture_ident is not None:
                    capture_stdout_fn = f"AR_logs/{capture_ident}.txt"
                    with open(
                        os.path.join(drivedir, capture_stdout_fn), "a", encoding="utf-8"
                    ) as f:
                        f.write(cap.stdout)
                    if len(cap.stderr) > 0:
                        capture_stderr_fn = f"AR_logs/{capture_ident}_stderr.txt"
                        with open(
                            os.path.join(drivedir, capture_stderr_fn),
                            "a",
                            encoding="utf-8",
                        ) as f:
                            f.write(cap.stderr)
    if capture_ident is not None:
        print(cap.stdout, end="")
        print(cap.stderr, end="")


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
            f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/pollgenerator",
            json={"results": RESULT_STACK if not dummy else {}},
        )

        PROMPT_STACK = r.json()

        RESULT_STACK = {
            k: v for k, v in RESULT_STACK.items() if k in PROMPT_STACK
        }  # clean out already used results

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
                    for k, v in hparams_select.values().items()
                    if k not in {"dtype", "adapt_layers"}
                },
                "hparams_select_sentiment": {
                    k: v
                    for k, v in hparams_select_sentiment.values().items()
                    if k not in {"dtype", "adapt_layers"}
                },
                "sampling_info": sampling_info,
            }
            RESULT_STACK[prompt_id]["model_info"] = model_info

        # print("done generating for this poll")

        if len(PROMPT_STACK) > 0:
            r = requests.post(
                f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/pollgenerator",
                json={"results": RESULT_STACK if not dummy else {}},
            )
            time.sleep(1)


def loop_poll(
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
            poll(capture_ident=capture_ident, dummy=dummy)
        except Exception as e:
            print(f"{type(e)}: {e}")
            time.sleep(period * 10)
        if len(RESULT_STACK) == 0 or dummy:
            time.sleep(period)


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
            poll_no_capture(capture_ident=capture_ident, dummy=dummy)
        except Exception as e:
            print(f"{type(e)}: {e}")
            time.sleep(period * 10)
        if len(RESULT_STACK) == 0 or dummy:
            time.sleep(period)
