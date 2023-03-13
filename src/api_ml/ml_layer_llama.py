import sys
import time
from io import StringIO
from functools import partial

import requests
import numpy as np
import torch
import bitsandbytes as bnb

import llama.load
import llama.generation

from config.autoresponder_config import *
from tumblr_to_text.classic.autoresponder_static_v8 import *

from ml.generator_model_torch import GPT_NEO_DEFAULT_SAMPLING_PARAMS

from util.util import typed_namedtuple_to_dict, hardcore_collect_and_show, show_gpu

BRIDGE_SERVICE_REMOTE_HOST, bridge_service_port = None, None

try:
    import config.bot_config_singleton
    bot_specific_constants = config.bot_config_singleton.bot_specific_constants

    bridge_service_port = bot_specific_constants.bridge_service_port
    BRIDGE_SERVICE_REMOTE_HOST = bot_specific_constants.BRIDGE_SERVICE_REMOTE_HOST
except FileNotFoundError:
    print("No config file found. Running in local mode.")

GENERATOR_METHODS_SERVED = "only_write"
MODELS_SERVED = {"generator"}

CONTROL_SEG_CONFIG = CONTROL_SEG_CONFIGS["V10_2"]

ORIG_POST_CHAR = CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"]

CLOSED_REQUESTS = {}


def no_init(loading_code):
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm, bnb.nn.Linear8bitLt]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy

    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]

    return result


class GeneratorModelLlama:
    def __init__(
        self,
        load_kwargs=dict(),
        generate_kwargs=dict(),
        required_continuation_room=required_continuation_room,
        lora_path=LLAMA_PATH_LORA,
    ):
        lora_premerged = lora_path is None
        load_kwargs_defaults=dict(
            ckpt_dir=LLAMA_PATH_CKPT,
            tokenizer_path=LLAMA_PATH_ENC,
            local_rank=0,
            world_size=1,
            use_cache=True,
            max_batch_size=1,
            n_ctx=LLAMA_N_CTX,
            use_xformers=False, # minimal lift with caching, can skip triton compile
            use_lora=not lora_premerged,
            lora_r=48,
            quantize_frozen=False, # 7B fits in fp16
            freeze_layers_below_n=32 if lora_premerged else 0,
            lowmem=True,
            lowmem_cpu_ratio=0,
        )
        load_kwargs_ = dict()
        load_kwargs_.update(load_kwargs_defaults)
        load_kwargs_.update(load_kwargs)
        load_kwargs = load_kwargs_

        self.load_kwargs = load_kwargs

        generate_kwargs_defaults=dict(
            max_gen_len=load_kwargs['n_ctx'],
            stop_at_eos=True,
            temperature=1.0,
            top_p=0.95, 
            breakruns=True, 
            breakruns_tau=0.02,
        )
        generate_kwargs_ = dict()
        generate_kwargs_.update(generate_kwargs_defaults)
        generate_kwargs_.update(generate_kwargs)
        generate_kwargs = generate_kwargs_

        self.n_ctx = load_kwargs['n_ctx']
        self.batch_size = load_kwargs['max_batch_size']
        if self.batch_size > 1:
            raise ValueError(self.batch_size)

        self.generate_kwargs = generate_kwargs
        self.required_continuation_room = required_continuation_room

        self.gen_model = no_init(partial(llama.load.load, **load_kwargs))

        hardcore_collect_and_show()

        if not lora_premerged:
            for n, p in self.gen_model.model.named_parameters():
                if 'lora' in n:
                    p.data = p.data.float()
            
            sd = torch.load(lora_path, map_location='cpu')
            
            llama.load.load_state_dict_meta(self.gen_model.model, sd, 'cpu')
            
            self.gen_model.model.merge_lora_into_base()

            hardcore_collect_and_show()

        self.gen_model.model.requires_grad_(False)
        self.gen_model.model.cuda()

    def write_random_prompt(self, prompts: list, probs: list, verbose=False):
        prompt = np.random.choice(prompts, p=np.array(probs) / sum(probs))
        return self.write(prompt=prompt, verbose=verbose)

    @property
    def max_context_size(self):
        return self.n_ctx - self.required_continuation_room

    def write(self, prompt: str, verbose=False, max_length_per_feed=None):
        done = False

        # calling code adds this sometimes
        prompt = prompt.replace("<|endoftext|>", "")
        
        tokens = [self.gen_model.tokenizer.encode(prompt, bos=True, eos=False)] * self.batch_size

        continuation_tokens = [[]] * len(tokens)

        while not done:
            generate_kwargs_ = dict()
            generate_kwargs_.update(self.generate_kwargs)

            if max_length_per_feed:
                generate_kwargs_['max_gen_len'] = max_length_per_feed

            tokens = [t[-self.max_context_size:] for t in tokens]
            prompt_lens = [len(t) for t in tokens]

            next_tokens, stop_reason = self.gen_model.generate(
                tokens,
                return_stop_reason=True,
                decode=False,
                **generate_kwargs_
            )

            for pl, nt, ct in zip(prompt_lens, next_tokens, continuation_tokens):
                ct.extend(nt[pl:])

            done = stop_reason == 'eos'

            tokens = next_tokens

        continuations = [self.gen_model.tokenizer.decode(t) for t in continuation_tokens]

        return {
            "continuations": continuations_,
            "side_data": {
                "prompt_for_neural": prompt,
            },
        }


generator_model = GeneratorModelLlama()

model_name = generator_model.load_kwargs['ckpt_dir']

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
                multirequest_sequence_in_process = False
                continue

            requested_model = generator_model
            multirequest_sequence_in_process = True

            requested_method = data["method"]

            if data["model"] == "generator":
                if GENERATOR_METHODS_SERVED == 'all_except_write' and method in {'write', 'write_random_prompt'}:
                    continue
                if GENERATOR_METHODS_SERVED == 'only_write' and method not in {'write', 'write_random_prompt'}:
                    continue

            requested_args, requested_kwargs = data.get("args", []), data.get(
                "kwargs", {}
            )
            for name in DEPRECATED_KWARGS:
                if name in requested_kwargs:
                    del requested_kwargs[name]

            with torch.inference_mode():
                with torch.no_grad():
                    result = getattr(requested_model, requested_method)(
                        *requested_args, **requested_kwargs
                    )

            if isinstance(result, np.ndarray):
                result = result.tolist()

            RESULT_STACK[prompt_id] = {"result": result}

            sampling_info = {
                "BREAKRUNS": generator_model.generate_kwargs['breakruns'],
                "BREAKRUNS_TAU": generator_model.generate_kwargs['breakruns_tau'],
                "BREAKRUNS_DECAY": 0,
                "length": generator_model.n_ctx,
                "T": generator_model.generate_kwargs['temperature'],
                "p": generator_model.generate_kwargs['top_p'],
            }

            hparams_select, hparams_select_sentiment = None, None

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

            hardcore_collect_and_show()
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
