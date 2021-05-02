import os
import subprocess
import time
from functools import partial

import requests
import numpy as np
from transformers import AutoTokenizer
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig

from autoresponder_config import *
from autoresponder_static import *
from autoresponder_static_v8 import *

from experimental.generator_model import (
    is_repeating_criterion,
)
from experimental.gptneo_generator_model import GPTNeoGeneratorModel, GPT_NEO_DEFAULT_SAMPLING_PARAMS
from selector_model.selector_estimator_neo import NostARHeadEstimator

from modeling_gpt_neo import GPTNeoForCausalLM, GPTNeoModel
GPTNeoModel.init_weights = lambda *args, **kwargs: None
GPTNeoForCausalLM.init_weights = lambda *args, **kwargs: None

from util.util import typed_namedtuple_to_dict


# TODO: move this over later
drivedir = "/content/drive/MyDrive/gpt_neo/"
os.chdir("/")

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


def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2", max_model_len=2048)
    tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
    return tokenizer


tokenizer = load_tokenizer()


def load_generator_model(
    path,
    tokenizer,
    batch_size,
    sampling_params=GPT_NEO_DEFAULT_SAMPLING_PARAMS,
    device="cuda:0",
    retries=False,
):
    state_dict = torch.load(
        os.path.join(path, "pytorch_model.bin"),
        map_location=torch.device(device),
    )

    transformers_model = GPTNeoForCausalLM.from_pretrained(
        None,
        state_dict=state_dict,
        config=GPTNeoConfig.from_pretrained(path),
    )
    transformers_model = transformers_model.to(device)

    return GPTNeoGeneratorModel.load(
        transformers_model=transformers_model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
        sampling_params=sampling_params,
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
    tokenizer=tokenizer,
    batch_size=batch_size,
    device=device,
    sampling_params=GPT_NEO_DEFAULT_SAMPLING_PARAMS,
)


def load_selector(path, base_model, tokenizer, retries=False, **kwargs):
    selector_est = NostARHeadEstimator.load(
        path, base_model=base_model, tokenizer=tokenizer, **kwargs
    )
    return selector_est


selector_est = load_from_gdrive_with_gs_fallback(
    load_fn=partial(load_selector, base_model=generator_model.transformers_model),
    relative_path=ckpt_select.rpartition("/")[0],  # TODO: redefine ckpt_select
    gs_command=gs_command_get_selector,
    session=generator_model.session,
    base_hparams=hparams,
    enc=enc,
    batch_size=batch_size,
)
selector_est.length = length_select

lr_calib_resp = selector_est.lr_calib_
lr_calib_orig = selector_est.lr_calib_


# sentiment_est = load_from_gdrive_with_gs_fallback(
#     load_fn=load_selector,
#     relative_path=ckpt_sentiment.rpartition("/")[0],  # TODO: redefine ckpt_select
#     gs_command=gs_command_get_sentiment,
#     session=generator_model.session,
#     base_hparams=hparams,
#     enc=enc,
#     batch_size=batch_size,
# )
# sentiment_est.length = length_sentiment
#
# lr_calib_sentiment = selector_est.lr_calib_
#
# autoreviewer_est = load_from_gdrive_with_gs_fallback(
#     load_fn=load_selector,
#     relative_path=ckpt_autoreviewer.rpartition("/")[0],  # TODO: redefine ckpt_select
#     gs_command=gs_command_get_autoreviewer,
#     session=generator_model.session,
#     base_hparams=hparams,
#     enc=enc,
#     batch_size=batch_size,
# )


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
                continue
                # requested_model = sentiment_est
            elif data["model"] == "autoreviewer":
                continue
                # requested_model = autoreviewer_est
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
                "MIRO": False,
                "MIRO_V2": False,
                "MIRO_TRUNC": MIRO_TRUNC,  # unused in miro v2
                "MIRO_LR": MIRO_LR,
                "USE_FIRST_STEP": False,
                "BREAKRUNS": False,
                "BREAKRUNS_TAU": BREAKRUNS_TAU,
                "BREAKRUNS_DECAY": BREAKRUNS_DECAY,
                "length": GPT_NEO_MAX_LENGTH,
                "T": GPT_NEO_T,
                "p": GPT_NEO_TOP_P,
                "chop_lowest": chop_lowest,
                "chop_highest": chop_highest,
                "first_step_length": first_step_length,
                "first_step_T": first_step_temperature,
                "first_step_p": first_step_top_p,
                "first_step_chop_lowest": first_step_chop_lowest,
                "first_step_chop_highest": first_step_chop_highest,
            }

            model_info = {
                "model_name": model_name,
                "ckpt_select": ckpt_select,
                "ckpt_sentiment": ckpt_sentiment,
                "ckpt_autoreviewer": ckpt_autoreviewer,
                "hparams_select": typed_namedtuple_to_dict(selector_est.params),
                "hparams_select_sentiment": {},  # typed_namedtuple_to_dict(sentiment_est.params),
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
