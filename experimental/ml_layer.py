import os
import subprocess
import time

import requests
import numpy as np

import model
import encoder

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

if V10_1:
    CONTROL_SEG_CONFIG = CONTROL_SEG_CONFIGS["V10_1"]
elif V10:
    CONTROL_SEG_CONFIG = CONTROL_SEG_CONFIGS["V10"]
else:
    CONTROL_SEG_CONFIG = CONTROL_SEG_CONFIGS["V9"]

if FORUMLIKE:
    ORIG_POST_CHAR = CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"]
else:
    ORIG_POST_CHAR = ORIG_POST_CHAR_CHINESE

CLOSED_REQUESTS = {}


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

autoreviewer_est = load_from_gdrive_with_gs_fallback(
    load_fn=load_selector,
    relative_path=ckpt_autoreviewer.rpartition("/")[0],  # TODO: redefine ckpt_select
    gs_command=gs_command_get_autoreviewer,
    session=generator_model.session,
    base_hparams=hparams,
    enc=enc,
    batch_size=batch_size,
)


def poll(
    dummy=False,
    ports=[
        bridge_service_port,
    ],
    routes=[
        "pollml",
    ],
):
    global CLOSED_REQUESTS

    for port, route in zip(ports, routes):
        r = requests.get(
            f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
        )

        PROMPT_STACK = {prompt_id: data for prompt_id, data in r.json().items()}

        RESULT_STACK = {}

        for prompt_id, data in PROMPT_STACK.items():
            if prompt_id in CLOSED_REQUESTS:
                RESULT_STACK[prompt_id] = CLOSED_REQUESTS[prompt_id]
                continue

            requested_model = None
            if data["model"] == "generator":
                requested_model = generator_model
            elif data["model"] == "selector":
                requested_model = selector_est
            elif data["model"] == "sentiment":
                requested_model = sentiment_est
            elif data["model"] == "autoreviewer":
                requested_model = autoreviewer_est
            else:
                raise ValueError(f"requested_model: {data.get('model')}")

            requested_method = data["method"]
            if not hasattr(requested_model, requested_method):
                raise ValueError(
                    f"requested_model {requested_model} has no method {requested_method}"
                )

            requested_args, requested_kwargs = data.get("args", []), data.get(
                "kwargs", {}
            )

            result = getattr(requested_model, requested_method)(
                *requested_args, **requested_kwargs
            )

            if isinstance(result, np.ndarray):
                result = result.tolist()

            if requested_method in {"done_writing"}:
                continue

            RESULT_STACK[prompt_id] = {"result": result}

            sampling_info = {
                "MIRO": MIRO,
                "MIRO_V2": MIRO_V2,
                "MIRO_TRUNC": MIRO_TRUNC,  # unused in miro v2
                "MIRO_LR": MIRO_LR,
                "MIRO_ONLY_ON_CONTINUE": MIRO_ONLY_ON_CONTINUE,
                "BREAKRUNS": BREAKRUNS,
                "BREAKRUNS_TAU": BREAKRUNS_TAU,
                "length": length,
                "T": temperature,
                "p": top_p,
                "chop_lowest": chop_lowest,
                "chop_highest": chop_highest,
                "pre_continue_length": pre_continue_length,
                "pre_continue_T": pre_continue_temperature,
                "pre_continue_p": pre_continue_top_p,
                "pre_continue_chop_lowest": pre_continue_chop_lowest,
                "pre_continue_chop_highest": pre_continue_chop_highest,
            }

            model_info = {
                "model_name": model_name,
                "ckpt_select": ckpt_select,
                "ckpt_sentiment": ckpt_sentiment,
                "ckpt_autoreviewer": ckpt_autoreviewer,
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

        if len(RESULT_STACK) > 0:
            requests.post(
                f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
                json=RESULT_STACK if not dummy else {},
            )

        open_request_ids = set()
        for prompt_id in PROMPT_STACK:
            if PROMPT_STACK[prompt_id].get("repeat_until_done_signal", False):
                open_request_ids.add(prompt_id)
            elif prompt_id in RESULT_STACK:
                CLOSED_REQUESTS[prompt_id] = RESULT_STACK[prompt_id]

        return open_request_ids


def loop_poll(
    period=60,
    dummy=False,
    ports=[
        bridge_service_port,
    ],
    routes=[
        "pollml",
    ],
):
    open_request_ids = set()
    while True:
        try:
            open_request_ids = poll(dummy=dummy, ports=ports, routes=routes)
        except Exception as e:
            print(f"{type(e)}: {e}")
            time.sleep(period * 10)
        if len(open_request_ids) == 0 or dummy:
            time.sleep(period)
        else:
            time.sleep(0.2)
