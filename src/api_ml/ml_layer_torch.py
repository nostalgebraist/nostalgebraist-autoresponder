import sys
import time

import requests
import numpy as np
from transformers import AutoTokenizer
from transformer_utils.util.tfm_utils import get_local_path_from_huggingface_cdn

from config.autoresponder_config import *
from tumblr_to_text.classic.autoresponder_static_v8 import *

from ml.generator_model_torch import GeneratorModelTorch, GPT_NEO_DEFAULT_SAMPLING_PARAMS, is_repeating_criterion
from classifier_heads.head_estimator import NostARHeadEstimator
from ml.load_gptj import load_gpt_j_split_ckpt

from util.util import typed_namedtuple_to_dict, collect_and_show, show_gpu

# TODO: move this over later
drivedir = "/content/drive/MyDrive/gpt_neo/"
os.chdir("/")

start_token = None

if V11_2:
    CONTROL_SEG_CONFIG = CONTROL_SEG_CONFIGS["V10_2"]
elif V10_1:
    CONTROL_SEG_CONFIG = CONTROL_SEG_CONFIGS["V10_1"]
elif V10:
    CONTROL_SEG_CONFIG = CONTROL_SEG_CONFIGS["V10"]
else:
    CONTROL_SEG_CONFIG = CONTROL_SEG_CONFIGS["V9"]

ORIG_POST_CHAR = CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"]

CLOSED_REQUESTS = {}


def load_from_gdrive_with_gs_fallback(
    load_fn, relative_path, gs_command, retries=False, **kwargs
):
    local_gdrive_path = os.path.join(drivedir, relative_path)
    local_gs_path = os.path.join("/", relative_path)
    print(f"local_gdrive_path: {local_gdrive_path}")
    print(f"local_gs_path: {local_gs_path}")

    os.makedirs(local_gs_path, exist_ok=True)

    enclosing_dir_exists = os.path.exists(local_gs_path)
    target_exists = len(os.listdir(local_gs_path)) > 0

    print(f"local_gs: enclosing dir {local_gs_path} exists?: {enclosing_dir_exists}")
    print(f"local_gs: enclosing {local_gs_path} non-empty?: {target_exists}")

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
    tokenizer = AutoTokenizer.from_pretrained("gpt2", max_model_len=2048 if V11 else 1024)
    tokenizer.add_special_tokens({'pad_token': '<|padding|>'})
    return tokenizer


tokenizer = load_tokenizer()


def load_generator_model(
    path,
    tokenizer,
    batch_size,
    device='cuda:0',
    sampling_params=GPT_NEO_DEFAULT_SAMPLING_PARAMS,
    retries=False,
):
    transformers_model = load_gpt_j_split_ckpt(path)

    return GeneratorModelTorch.load(
        transformers_model=transformers_model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
        sampling_params=sampling_params,
    )


def load_selector(path, base_model, tokenizer, retries=False, **kwargs):
    selector_est = NostARHeadEstimator.load(
        path, base_model=base_model, tokenizer=tokenizer, inference_batch_size=head_inference_batch_size,
        device='cpu',
        **kwargs
    )
    return selector_est


def make_sample_done_criterion(control_seg_config):
    def sample_done_criterion(text, unique_token_frac):
        has_EOT = EOT in text

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

# MODELS
generator_model, selector_est, sentiment_est, autoreviewer_est = None, None, None, None

# GENERATOR: download if necessary
generator_path = model_name

if not os.path.exists(generator_path):
    model_tar_path = get_local_path_from_huggingface_cdn(
        'nostalgebraist/nostalgebraist-autoresponder-6_1b', 'model.tar.gz'
    )
    subprocess.run(f"tar -xf {model_tar_path} && rm {model_tar_path}", shell=True)

# HEADS: download if necessary
head_paths = [ckpt_select, ckpt_sentiment, ckpt_autoreviewer]
needs_head_download = not all(os.path.exists(path) for path in head_paths)
heads_tar_path = ""

if needs_head_download:
    heads_tar_path = get_local_path_from_huggingface_cdn(
        'nostalgebraist/nostalgebraist-autoresponder-6_1b', 'heads.tar.gz'
    )

if "selector" in MODELS_SERVED and not os.path.exists(ckpt_select):
    subprocess.run(f"tar -xvf {heads_tar_path} {ckpt_select}", shell=True)

if "sentiment" in MODELS_SERVED and not os.path.exists(ckpt_sentiment):
    subprocess.run(f"tar -xvf {heads_tar_path} {ckpt_sentiment}", shell=True)

if "autoreviewer" in MODELS_SERVED and not os.path.exists(ckpt_autoreviewer):
    subprocess.run(f"tar -xvf {heads_tar_path} {ckpt_autoreviewer}", shell=True)

if needs_head_download:
    subprocess.run(f"rm {heads_tar_path}", shell=True)

# MODELS: load
generator_model = load_generator_model(
    path=generator_path,
    tokenizer=tokenizer,
    batch_size=batch_size,
    device='cuda:0',
    sampling_params=GPT_NEO_DEFAULT_SAMPLING_PARAMS,
)

if "selector" in MODELS_SERVED:
    selector_est = load_selector(ckpt_select, base_model=generator_model.transformers_model, tokenizer=tokenizer)
    selector_est.length = length_select

if "sentiment" in MODELS_SERVED:
    sentiment_est = load_selector(ckpt_sentiment, base_model=generator_model.transformers_model, tokenizer=tokenizer)
    sentiment_est.length = length_sentiment

if "autoreviewer" in MODELS_SERVED:
    autoreviewer_est = load_selector(ckpt_autoreviewer, base_model=generator_model.transformers_model, tokenizer=tokenizer)

DEPRECATED_KWARGS = {"mirotarg"}


def poll(
    dummy=False,
    ports=[
        bridge_service_port,
    ],
    routes=[
        "pollml",
    ],
    show_memory=True,
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

            if data["model"] not in MODELS_SERVED:
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

            if GPU_TYPE != "big" and requested_method == "write":
                # can't handle long pasts calc yet
                prompt = requested_kwargs.get("prompt")
                if not prompt and len(requested_args) > 0:
                    prompt = requested_args[0]

                ntok = len(generator_model.tokenizer.encode(prompt))
                print(f"prompt length: {ntok}")

            for name in DEPRECATED_KWARGS:
                if name in requested_kwargs:
                    print(f"skipping deprecated param {name}")
                    del requested_kwargs[name]

            result = getattr(requested_model, requested_method)(
                *requested_args, **requested_kwargs
            )

            if isinstance(result, np.ndarray):
                result = result.tolist()

            RESULT_STACK[prompt_id] = {"result": result}

            sampling_info = {
                "MIRO": False,
                "MIRO_V2": False,
                "MIRO_TRUNC": MIRO_TRUNC,  # unused in miro v2
                "MIRO_LR": MIRO_LR,
                "USE_FIRST_STEP": False,
                "BREAKRUNS": BREAKRUNS,
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

            hparams_select, hparams_select_sentiment = None, None

            if "selector" in MODELS_SERVED:
                hparams_select = typed_namedtuple_to_dict(selector_est.params)

            if "sentiment" in MODELS_SERVED:
                hparams_select_sentiment = typed_namedtuple_to_dict(sentiment_est.params)

            model_info = {
                "model_name": model_name,
                "ckpt_select": ckpt_select,
                "ckpt_sentiment": ckpt_sentiment,
                "ckpt_autoreviewer": ckpt_autoreviewer,
                "hparams_select": hparams_select,
                "hparams_select_sentiment": hparams_select_sentiment,
                "sampling_info": sampling_info,
            }
            RESULT_STACK[prompt_id]["model_info"] = model_info

            # print(f"sending back:")
            # print(repr(RESULT_STACK[prompt_id]))

        if len(RESULT_STACK) > 0:
            requests.post(
                f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
                json=RESULT_STACK if not dummy else {},
            )

            collect_and_show()
            if show_memory:
                show_gpu()

        open_request_ids = set()
        for prompt_id in PROMPT_STACK:
            if PROMPT_STACK[prompt_id].get("repeat_until_done_signal", False):
                open_request_ids.add(prompt_id)
            elif prompt_id in RESULT_STACK:
                CLOSED_REQUESTS[prompt_id] = RESULT_STACK[prompt_id]

        return open_request_ids


def loop_poll(
    period=1,
    dummy=False,
    ports=[
        bridge_service_port,
    ],
    routes=[
        "pollml",
    ],
    show_memory=True,
):
    open_request_ids = set()
    while True:
        try:
            open_request_ids = poll(dummy=dummy, ports=ports, routes=routes, show_memory=show_memory)
        except Exception as e:
            raise e
            # print(f"{type(e)}: {e}")
            # time.sleep(period * 10)
        if len(open_request_ids) == 0 or dummy:
            time.sleep(period)
        else:
            time.sleep(0.2)


if __name__ == "__main__":
    sys.exit(loop_poll())
