import sys
import time

import requests
import numpy as np
import torch
from transformers import AutoTokenizer
from transformer_utils.util.tfm_utils import get_local_path_from_huggingface_cdn

import magma

from config.autoresponder_config import *
from tumblr_to_text.classic.autoresponder_static_v8 import *

from ml.generator_model_torch import GeneratorModelTorch, GPT_NEO_DEFAULT_SAMPLING_PARAMS, is_repeating_criterion
from classifier_heads.head_estimator import NostARHeadEstimator
from ml.load_gptj import load_gpt_j_split_ckpt, load_gpt_j_split_ckpt_state_dict, quick_init_gptj
from ml.kv_cache import setup_kv_buffer

import ml.captioning

from util.util import typed_namedtuple_to_dict, collect_and_show, show_gpu

import config.bot_config_singleton
bot_specific_constants = config.bot_config_singleton.bot_specific_constants

bridge_service_port = bot_specific_constants.bridge_service_port
BRIDGE_SERVICE_REMOTE_HOST = bot_specific_constants.BRIDGE_SERVICE_REMOTE_HOST


def caption_image(self, path_or_url, **kwargs):
    return ml.captioning.caption_image(
        path_or_url=path_or_url,
        magma_wrapper=self,
        adapters_device=captioning_adapters_device,
        **kwargs
    )

magma.Magma.caption_image = caption_image

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
    use_captioner=False,
    captioner_path="",
    use_kv_buffer=True,
):
    if use_captioner:
        sd = load_gpt_j_split_ckpt_state_dict(path)

        magma_config_path = os.path.join(captioner_path, 'config.yml')

        magma_wrapper = magma.Magma.from_split_checkpoint(
            path=captioner_path,
            config_path=magma_config_path,
            lm_path_or_state_dict=sd,
            gptj_init_fn=quick_init_gptj,
            device=device,
        )

        magma_wrapper.detach_adapters()

        if not LATE_TRANSFER_TO_GPU:
            for k in magma_wrapper.adapter_map:
                magma_wrapper.adapter_map[k].to(device=captioning_adapters_device)

            magma_wrapper.image_prefix.to(device=captioning_adapters_device)

        if use_kv_buffer:
            setup_kv_buffer(magma_wrapper, batch_size=batch_size, max_sequence_length=max_feed_size_with_cache)

        transformers_model = magma_wrapper.lm
    else:
        transformers_model = load_gpt_j_split_ckpt(path)
        magma_wrapper = None

    generator_model = GeneratorModelTorch.load(
        transformers_model=transformers_model,
        tokenizer=tokenizer,
        batch_size=batch_size,
        device=device,
        sampling_params=sampling_params,
    )

    return generator_model, magma_wrapper


def load_selector(path, base_model, tokenizer, retries=False, **kwargs):
    selector_est = NostARHeadEstimator.load(
        path,
        base_model=base_model,
        tokenizer=tokenizer,
        inference_batch_size=head_inference_batch_size,
        use_amp_inference=autocast_recommended,
        device=head_load_device,
        blocks_inference_device_attn=head_inference_blocks_device_attn,
        blocks_inference_device_mlp=head_inference_blocks_device_mlp,
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

t_start = time.time()

# GENERATOR: download if necessary
generator_path = model_name

if not os.path.exists(generator_path):
    model_tar_name = 'model.tar.gz' if HF_FILES_GZIPPED else 'model.tar'
    model_tar_path = get_local_path_from_huggingface_cdn(
        HF_REPO_NAME, model_tar_name
    )
    subprocess.run(f"tar -xf {model_tar_path} && rm {model_tar_path}", shell=True)

# HEADS: download if necessary
head_paths = [ckpt_select, ckpt_sentiment, ckpt_autoreviewer, ckpt_captioner]
needs_head_download = not all(os.path.exists(path) for path in head_paths)
heads_tar_path = ""

if needs_head_download:
    heads_tar_name = 'heads.tar.gz' if HF_FILES_GZIPPED else 'heads.tar'
    heads_tar_path = get_local_path_from_huggingface_cdn(
        HF_REPO_NAME, heads_tar_name
    )

if "selector" in MODELS_SERVED and not os.path.exists(ckpt_select):
    subprocess.run(f"tar -xvf {heads_tar_path} {ckpt_select}", shell=True)

if "sentiment" in MODELS_SERVED and not os.path.exists(ckpt_sentiment):
    subprocess.run(f"tar -xvf {heads_tar_path} {ckpt_sentiment}", shell=True)

if "autoreviewer" in MODELS_SERVED and not os.path.exists(ckpt_autoreviewer):
    subprocess.run(f"tar -xvf {heads_tar_path} {ckpt_autoreviewer}", shell=True)

if "captioner" in MODELS_SERVED and not os.path.exists(ckpt_captioner):
    subprocess.run(f"tar -xvf {heads_tar_path} {ckpt_captioner}", shell=True)

if needs_head_download:
    subprocess.run(f"rm {heads_tar_path}", shell=True)

t_file = time.time()
print(f"downloaded in {t_file - t_start}s")

# MODELS: load
generator_model, magma_wrapper = load_generator_model(
    path=generator_path,
    tokenizer=tokenizer,
    batch_size=batch_size,
    device='cpu' if LATE_TRANSFER_TO_GPU else 'cuda:0',
    sampling_params=GPT_NEO_DEFAULT_SAMPLING_PARAMS,
    use_captioner="captioner" in MODELS_SERVED,
    captioner_path=os.path.abspath(ckpt_captioner),
    use_kv_buffer=USE_KV_BUFFER,
)

if "selector" in MODELS_SERVED:
    selector_est = load_selector(ckpt_select, base_model=generator_model.transformers_model, tokenizer=tokenizer)
    selector_est.length = length_select

if "sentiment" in MODELS_SERVED:
    sentiment_est = load_selector(ckpt_sentiment, base_model=generator_model.transformers_model, tokenizer=tokenizer)
    sentiment_est.length = length_sentiment

if "autoreviewer" in MODELS_SERVED:
    autoreviewer_est = load_selector(ckpt_autoreviewer, base_model=generator_model.transformers_model, tokenizer=tokenizer)
    autoreviewer_est.length = length_autoreview

DEPRECATED_KWARGS = {"mirotarg"}

t_ready = time.time()
print(f"ready in {t_ready - t_start}s (model load: {t_ready - t_file}s)")


def activate(magma_wrapper):
    magma_wrapper.lm.cuda()

    magma_wrapper.detach_adapters()

    for k in magma_wrapper.adapter_map:
        magma_wrapper.adapter_map[k].cuda()

    magma_wrapper.image_prefix.cuda()


def is_active():
    return magma_wrapper.lm.device == 'cuda:0'


if os.environ.get('EARLY_ACTIVATE', False):
    activate()


def poll(
    dummy=False,
    ports=[
        bridge_service_port,
    ],
    routes=[
        "pollml",
    ],
    show_memory=True,
    multirequest_sequence_in_process=False,
):
    global CLOSED_REQUESTS

    for port, route in zip(ports, routes):
        r = requests.get(
            f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
        )

        PROMPT_STACK = {prompt_id: data for prompt_id, data in r.json().items()}

        RESULT_STACK = {}

        last_requested_model_name = None

        for prompt_id, data in PROMPT_STACK.items():
            if prompt_id in CLOSED_REQUESTS:
                RESULT_STACK[prompt_id] = CLOSED_REQUESTS[prompt_id]
                continue

            if data["model"] not in MODELS_SERVED:
                continue

            requested_model = None
            if data["model"] == "generator":
                requested_model = generator_model
                multirequest_sequence_in_process = True
            elif data["model"] == "selector":
                requested_model = selector_est
                multirequest_sequence_in_process = True
            elif data["model"] == "sentiment":
                requested_model = sentiment_est
            elif data["model"] == "autoreviewer":
                requested_model = autoreviewer_est
                multirequest_sequence_in_process = False
            elif data["model"] == "captioner":
                requested_model = magma_wrapper
                multirequest_sequence_in_process = True
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

            if not is_active():
                wait_secs = 10
                print(f"Waiting {wait_secs}s before activating...")
                time.sleep(wait_secs)

                activate()
                print(f"activate done. is_active: {is_active()}")

            # keep magma activated over strings of captioning requests
            if data["model"] == "captioner":
                requested_kwargs['deactivate_when_done'] = False
            elif len(magma_wrapper.adapter_map) == 0:
                # need magma decativated, but adapters are attached
                ml.captioning.deactivate_magma(
                    magma_wrapper,
                    adapters_device=captioning_adapters_device,
                )

            for name in DEPRECATED_KWARGS:
                if name in requested_kwargs:
                    print(f"skipping deprecated param {name}")
                    del requested_kwargs[name]

            with torch.inference_mode():
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
                "length": generator_model.max_feed_size_with_cache,
                "T": GPT_NEO_T,
                "p": GPT_NEO_TOP_P,
                "chop_lowest": chop_lowest,
                "chop_highest": chop_highest,
                "first_step_length": first_step_length,
                "first_step_T": first_step_temperature,
                "first_step_p": first_step_top_p,
                "first_step_chop_lowest": first_step_chop_lowest,
                "first_step_chop_highest": first_step_chop_highest,
                "TYPICAL_SAMPLING": TYPICAL_SAMPLING,
                "TYPICAL_SAMPLING_MASS": TYPICAL_SAMPLING_MASS,
                "TYPICAL_SAMPLING_MIN_TOKENS_TO_KEEP": TYPICAL_SAMPLING_MIN_TOKENS_TO_KEEP
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

        almostdone_in_flight = False
        open_request_ids = set()
        for prompt_id in PROMPT_STACK:
            if PROMPT_STACK[prompt_id].get("repeat_until_done_signal", False):
                open_request_ids.add(prompt_id)
                if PROMPT_STACK[prompt_id].get("almost_done", False):
                    almostdone_in_flight = True
            elif prompt_id in RESULT_STACK:
                CLOSED_REQUESTS[prompt_id] = RESULT_STACK[prompt_id]

        return open_request_ids, almostdone_in_flight, multirequest_sequence_in_process


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
    n_loops=None,
    use_almostdone=True,
    multirequest_sequence_in_process=False,
):
    loop_counter = 0
    open_request_ids = set()

    def _should_stop(loop_counter, open_request_ids):
        if n_loops is not None:
            return (loop_counter >= n_loops) and (open_request_ids == set())
        return False

    while not _should_stop(loop_counter, open_request_ids):
        open_request_ids, almostdone_in_flight, multirequest_sequence_in_process = poll(
            dummy=dummy, ports=ports, routes=routes, show_memory=show_memory,
            multirequest_sequence_in_process=multirequest_sequence_in_process
        )
        if multirequest_sequence_in_process:
            time.sleep(0.1)
        elif len(open_request_ids) == 0 or dummy:
            time.sleep(period)
        elif use_almostdone and almostdone_in_flight:
            time.sleep(2)
        else:
            time.sleep(0.2)
        loop_counter += 1


if __name__ == "__main__":
    sys.exit(loop_poll())
