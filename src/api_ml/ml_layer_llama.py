import sys
import time
from io import StringIO
from functools import partial
from typing import List

import requests
import numpy as np
import torch
import bitsandbytes as bnb

import llama.load
import llama.generation
import llama.tokenizer

from config.autoresponder_config import *
from tumblr_to_text.classic.autoresponder_static_v8 import *

from ml.split_checkpoint import SplitCheckpoint

from util.util import collect_and_show, show_gpu

BRIDGE_SERVICE_REMOTE_HOST, bridge_service_port = None, None

try:
    import config.bot_config_singleton
    bot_specific_constants = config.bot_config_singleton.bot_specific_constants

    bridge_service_port = bot_specific_constants.bridge_service_port
    BRIDGE_SERVICE_REMOTE_HOST = bot_specific_constants.BRIDGE_SERVICE_REMOTE_HOST
except FileNotFoundError:
    print("No config file found. Running in local mode.")

GENERATOR_METHODS_SERVED = GENERATOR_METHODS_SERVED_LLAMA
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


def collect_and_show_cache_clear():
    collect_and_show()
    torch.cuda.empty_cache()


def set_bnb_threshold(p):
    def _set_bnb_threshold(m):
        if hasattr(m, 'state') and hasattr(m.state, 'threshold'):
            m.state.threshold = p
    return set_bnb_threshold


def set_bnb_thresholds(model, p, pw2):
    model.apply(set_bnb_threshold(p))
    for l in model.layers:
        if hasattr(l.feed_forward.w2, 'state'):
            l.feed_forward.w2.state.threshold = pw2


class LlamaAvoidUnkCaptionLogitsProcessor:
    def __init__(self,
                 device='cuda:0'
                 ):
        self.unk_prefix = torch.as_tensor(
            [25512, 13]).to(device)  # ['===', '\n']
        self.prefix_length = self.unk_prefix.shape[0]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq_length = input_ids.shape[1]

        if seq_length < self.prefix_length:
            return scores

        with torch.no_grad():
            for i in range(len(input_ids)):
                if (input_ids[i, -self.prefix_length:] == self.unk_prefix.to(input_ids.device)).all():
                    scores[i, 26690] = scores[i, :].min() - 10000.  # 'unknown'

        return scores


def make_preserve_tokens(token_strings, enc):
    tokens = [
        enc.encode(ss, 0, 0)[-1]
        for s in token_strings
        for ss in [s]
    ]
    return sorted(set(tokens))


class RepetitionPenaltyLogitsProcessor:
    """
    TODO: develop a version of repetition penalty that respects the translation invariance of logits
    """

    def __init__(self, penalty: float,
                 preserve_tokens=None,
                 device='cuda'):
        if not isinstance(penalty, float) or not (penalty > 0):
            raise ValueError(
                f"`penalty` has to be a strictly positive float, but is {penalty}")

        self.penalty = penalty
        self.preserve_tokens = preserve_tokens
        if self.preserve_tokens is not None:
            self.preserve_tokens = torch.as_tensor(
                self.preserve_tokens)[None, :].to(device=device)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.preserve_tokens is not None:
            keep = scores[:, self.preserve_tokens[0]]

        score = torch.gather(scores, 1, input_ids)

        # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
        score = torch.where(score < 0, score * self.penalty,
                            score / self.penalty)

        scores.scatter_(1, input_ids, score)

        scores.scatter_(1, self.preserve_tokens, keep)

        return scores


class GeneratorModelLlama:
    def __init__(
        self,
        load_kwargs=dict(),
        generate_kwargs=dict(),
        required_continuation_room=required_continuation_room,
        max_continue_tokens=MAX_CONTINUE_TOKENS,
        lora_path=LLAMA_PATH_LORA,
        require_xformers=True,
    ):
        use_xformers = False
        try:
            import xformers.ops
            import triton.language
            use_xformers = True
        except Exception as e:
            if require_xformers:
                raise e
            
        checkpoint = None
        if LLAMA_SPLIT_CKPT:
            checkpoint = SplitCheckpoint(LLAMA_PATH_CKPT)

        lora_premerged = lora_path is None
        load_kwargs_defaults=dict(
            checkpoint=checkpoint,
            ckpt_dir=LLAMA_PATH_CKPT,
            tokenizer_path=LLAMA_PATH_ENC,
            local_rank=0,
            world_size=1,
            use_cache=True,
            max_batch_size=1,
            n_ctx=LLAMA_N_CTX,
            use_xformers=use_xformers,
            use_lora=not lora_premerged,
            lora_r=48,
            quantize_frozen=LLAMA_QUANTIZE,
            quantize_cache=LLAMA_QUANTIZE_CACHE,
            quantize_cache_above=LLAMA_QUANTIZE_CACHE_ABOVE,
            quantize_cache_after_token=LLAMA_QUANTIZE_CACHE_AFTER_TOKEN,
            freeze_layers_below_n=40 if lora_premerged else 0,
            lowmem=True,
            lowmem_cpu_ratio=0,
            fp32_logits=False,
        )
        load_kwargs_ = dict()
        load_kwargs_.update(load_kwargs_defaults)
        load_kwargs_.update(load_kwargs)
        load_kwargs = load_kwargs_

        self.load_kwargs = load_kwargs

        enc = llama.tokenizer.Tokenizer(LLAMA_PATH_ENC)

        extra_logits_processors = [LlamaAvoidUnkCaptionLogitsProcessor()]

        if LLAMA_REP_PENALTY > 0.:
            extra_logits_processors = [
                RepetitionPenaltyLogitsProcessor(
                    LLAMA_REP_PENALTY,
                    # preserve_tokens=None,
                    preserve_tokens=make_preserve_tokens(LLAMA_PRESERVE_TOKENS, enc),
                )
            ] + extra_logits_processors

        generate_kwargs_defaults=dict(
            max_gen_len=load_kwargs['n_ctx'],
            stop_at_eos=True,
            temperature=LLAMA_TEMPERATURE,
            top_p=0.95, 
            breakruns=LLAMA_BREAKRUNS, 
            breakruns_tau=LLAMA_BREAKRUNS_TAU,
            allow_xformers=use_xformers,
            all_xformers=use_xformers,
            extra_logits_processors=extra_logits_processors,
            cache_build_size=LLAMA_CACHE_BUILD_SIZE,
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
        self.max_continue_tokens = max_continue_tokens

        self.gen_model = no_init(partial(llama.load.load, **load_kwargs))

        collect_and_show_cache_clear()

        if not lora_premerged:
            for n, p in self.gen_model.model.named_parameters():
                if 'lora' in n:
                    p.data = p.data.float()
            
            sd = torch.load(lora_path, map_location='cpu')
            
            llama.load.load_state_dict_meta(self.gen_model.model, sd, 'cpu')
            
            self.gen_model.model.merge_lora_into_base()

            collect_and_show_cache_clear()

        set_bnb_thresholds(
            self.gen_model.model,p=LLAMA_CUSTOM_LOAD_KWARGS.get('quantize_threshold', 6),
            pw2=LLAMA_W2_THRESHOLD
        )

        self.gen_model.model.requires_grad_(False)
        self.gen_model.model.cuda()

        self.eos_token = self.gen_model.tokenizer.encode("", bos=False, eos=True)[0]

    def write_random_prompt(self, prompts: list, probs: list, verbose=False):
        prompt = np.random.choice(prompts, p=np.array(probs) / sum(probs))
        return self.write(prompt=prompt, verbose=verbose)

    @property
    def max_context_size(self):
        return self.n_ctx - self.required_continuation_room

    def write(self, prompt: str, verbose=False, max_length_per_feed=None):
        max_length_per_feed = max_length_per_feed or self.max_continue_tokens
        done = False

        prompt_orig = prompt

        # calling code adds this sometimes
        prompt = prompt.replace("<|endoftext|>", "")
        
        tokens = [[self.eos_token] + self.gen_model.tokenizer.encode(prompt, bos=False, eos=False)] * self.batch_size

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
                progress_bar=verbose,
                progress_bar_show_text=verbose,
                **generate_kwargs_
            )

            for pl, nt, ct in zip(prompt_lens, next_tokens, continuation_tokens):
                ct.extend(nt[pl:])

            more_permitted = len(continuation_tokens[0]) < self.max_continue_tokens
            done = (stop_reason == 'eos') or (not more_permitted)

            tokens = next_tokens

        continuations = [self.gen_model.tokenizer.decode(t) for t in continuation_tokens]

        return {
            "continuations": continuations,
            "side_data": {
                "prompt_for_neural": prompt_orig,
            },
        }
    
    @torch.no_grad()
    def get_next_logits(self, tokens: list, to_numpy=True):
        tokens = [t[-self.max_context_size:] for t in tokens]

        tokens = torch.as_tensor(tokens, device='cuda')
        
        cache_build_size = self.generate_kwargs.get('cache_build_size')
        prev_pos = 0
        cur_pos = tokens.shape[1]
        if cache_build_size is not None:
            while cur_pos - prev_pos > cache_build_size:
                self.gen_model.model(
                    tokens[:, prev_pos:prev_pos + cache_build_size], prev_pos)
                prev_pos = prev_pos + cache_build_size
        logits = self.gen_model.model(tokens[:, prev_pos:cur_pos], prev_pos)[0, -1]

        if to_numpy:
            logits = logits.cpu().numpy()
        return logits

    def get_next_probs(self, tokens: list, forbidden_tokens: List[int] = None, to_numpy=True):
        logits = self.get_next_logits(tokens, to_numpy=False)
        if forbidden_tokens is not None:
            logits[forbidden_tokens] = -1000.
        probs = torch.softmax(logits, dim=-1)
        if to_numpy:
            probs = probs.cpu().numpy()
        return probs
    
    def get_prob_delta_over_ref(self, text: str, text_ref: str, token_str: str,
                                forbidden_strings: List[str],
                                ):
        if token_str in forbidden_strings:
            return 0.
        
        token_str = token_str.lstrip(' ')

        token = self.gen_model.tokenizer.encode(token_str, 0, 0)[0]

        forbidden_tokens = [self.gen_model.tokenizer.encode(
            s, 0, 0)[0] for s in forbidden_strings]
        
        text_ref = ' \n\n' + text_ref

        text_ref_tokens = [self.gen_model.tokenizer.encode(text_ref, 0, 0)[1:]]
        text_tokens = [self.gen_model.tokenizer.encode(text, 0, 0)]

        prob_ref = self.get_next_probs(text_ref_tokens, forbidden_tokens=[], to_numpy=True)[token]
        prob = self.get_next_probs(text_tokens, forbidden_tokens=forbidden_tokens, to_numpy=True)[token]

        delta = np.log(prob + 1e-5) - np.log(prob_ref + 1e-5)

        print(f"text {repr(text)},  text_ref {(repr(text_ref))}, token_str {token_str}, forbidden_strings {forbidden_strings}")
        print(f"token {repr(token)}, text_tokens[0][-8:] {text_tokens[0][-8:]}, text_ref_tokens[0][-8:] {text_ref_tokens[0][-8:]}")
        print(f"delta {delta}, prob {prob}, prob_ref {prob_ref}")
        print()

        if np.isnan(delta) or np.isinf(delta):
            delta = 0.

        delta = float(delta)
        return delta

    def get_prob_delta_over_ref_multi(self, text: List[str], text_ref: List[str], token_str: str,
                                      forbidden_strings: List[List[str]]):
        return [self.get_prob_delta_over_ref(t, tr, token_str, fs)
                for t, tr, fs in zip(text, text_ref, forbidden_strings)]



generator_model = GeneratorModelLlama(load_kwargs=LLAMA_CUSTOM_LOAD_KWARGS)

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

    UNSERVABLE_REQUESTS = set()

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
                UNSERVABLE_REQUESTS.add(prompt_id)
                multirequest_sequence_in_process = False
                continue

            requested_model = generator_model
            multirequest_sequence_in_process = True

            requested_method = data["method"]

            if data["model"] == "generator":
                if GENERATOR_METHODS_SERVED == 'all_except_write' and requested_method in {'write', 'write_random_prompt'}:
                    multirequest_sequence_in_process = False
                    UNSERVABLE_REQUESTS.add(prompt_id)
                    continue
                if GENERATOR_METHODS_SERVED == 'only_write' and requested_method not in {'write', 'write_random_prompt'}:
                    multirequest_sequence_in_process = False
                    UNSERVABLE_REQUESTS.add(prompt_id)
                    continue
                if GENERATOR_METHODS_SERVED == 'only_write_prob_delt' and requested_method not in {
                    'write', 'write_random_prompt', 'get_prob_delta_over_ref_multi'
                }:
                    multirequest_sequence_in_process = False
                    UNSERVABLE_REQUESTS.add(prompt_id)
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

        if len(RESULT_STACK) > 0:
            requests.post(
                f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
                json=RESULT_STACK if not dummy else {},
            )

            if dummy:
                print(f"would have sent:")
                print(repr(RESULT_STACK[prompt_id]))

            collect_and_show_cache_clear()
            if show_memory:
                show_gpu()

        almostdone_in_flight = False
        open_request_ids = set()
        for prompt_id in PROMPT_STACK:
            if prompt_id in UNSERVABLE_REQUESTS:
                continue
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
        if use_almostdone and almostdone_in_flight:
            time.sleep(2)
        elif multirequest_sequence_in_process and len(MODELS_SERVED) > 1:
            time.sleep(0.1)
        elif len(open_request_ids) == 0 or dummy:
            time.sleep(period)
        else:
            time.sleep(0.2)
        loop_counter += 1


if __name__ == "__main__":
    sys.exit(loop_poll())
