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

from experimental.generator_model import SamplingParams, SamplingConfig, DEFAULT_SAMPLING_CONFIG, GeneratorModel, is_repeating_criterion
from selector_model.selector_estimator import SelectorEstimatorFromCkpt

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


enc = encoder.get_encoder_from_path("/models/autoresponder_v10_1/", eot_workaround=EOT_WORKAROUND)


def load_generator_model(
    path, enc, batch_size, sampling_config, sample_done_criterion, hparams, retries=False
):
    return GeneratorModel.load(
        path, enc, batch_size, sample_done_criterion, sampling_config, hparams, retries=retries
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


generator_model = GeneratorModel.load(
    path='/models/autoresponder_v10_1/model-141.hdf5',
    enc=enc,
    batch_size=batch_size,
    sampling_config=DEFAULT_SAMPLING_CONFIG,
    sample_done_criterion=sample_done_criterion,
    hparams=hparams,
)


def load_selector(path, session, base_hparams, enc, retries=False, **kwargs):
    selector_est = SelectorEstimatorFromCkpt.load(
        path, session=session, base_hparams=base_hparams, enc=enc, **kwargs
    )
    return selector_est


selector_est = SelectorEstimatorFromCkpt.load(
    path="/selector",
    session=generator_model.session,
    base_hparams=hparams,
    enc=enc,
    batch_size=batch_size,
)
selector_est.length = length_select

lr_calib_resp = selector_est.lr_calib_resp_
lr_calib_orig = selector_est.lr_calib_orig_


sentiment_est = SelectorEstimatorFromCkpt.load(
    path="/sentiment",
    session=generator_model.session,
    base_hparams=hparams,
    enc=enc,
    batch_size=batch_size,
)
sentiment_est.length = length_sentiment

lr_calib_sentiment = selector_est.lr_calib_

autoreviewer_est = SelectorEstimatorFromCkpt.load(
    path="/draft_autoreviewer",
    session=generator_model.session,
    base_hparams=hparams,
    enc=enc,
    batch_size=batch_size,
)


def handle_request(data, lambda_uid: str=None):
    requested_model = None

    if 'request_id' not in data:
        raise ValueError('no request id')
    if 'bridge_id' not in data:
        raise ValueError('no bridge id')
    request_id = data['request_id']
    bridge_id = data['bridge_id']

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

    response_data = {"result": result, 'request_id': request_id, 'bridge_id': bridge_id}

    sampling_info = {
        "MIRO": MIRO,
        "MIRO_V2": MIRO_V2,
        "MIRO_TRUNC": MIRO_TRUNC,  # unused in miro v2
        "MIRO_LR": MIRO_LR,
        "USE_FIRST_STEP": USE_FIRST_STEP,
        "BREAKRUNS": BREAKRUNS,
        "BREAKRUNS_TAU": BREAKRUNS_TAU,
        "BREAKRUNS_DECAY": BREAKRUNS_DECAY,
        "length": length,
        "T": temperature,
        "p": top_p,
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
    response_data["model_info"] = model_info

    response_data["lambda_uid"] = lambda_uid

    return response_data
