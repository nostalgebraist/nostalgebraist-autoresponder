from typing import List

import numpy as np
import torch

from config.autoresponder_config import *  # TODO: move elsewhere?
from ml.sampling_params import SamplingParams, DEFAULT_SAMPLING_CONFIG

from transformers import LogitsProcessorList
from ml.sample_torch import BreakrunsLogitsProcessor, TypicalLogitsWarper, AvoidUnkCaptionLogitsProcessor

from util.util import copy_and_update_config, collect_and_show

GPT_NEO_DEFAULT_SAMPLING_PARAMS = copy_and_update_config(
    SamplingParams,
    DEFAULT_SAMPLING_CONFIG.params,
    temperature=GPT_NEO_T,
    top_p=GPT_NEO_TOP_P,
    top_k=GPT_NEO_TOP_K,
    avoid_unk_caption=AVOID_UNK_CAPTION,
    breakruns_off_within_images=BREAKRUNS_OFF_WITHIN_IMAGES,
)


def is_repeating_criterion(unique_token_frac):
    return unique_token_frac < 0.2


def hardcore_collect_and_show():
    collect_and_show()
    torch.cuda.empty_cache()


def override_disable_logits_processors(*args, **kwargs) -> LogitsProcessorList:
    return LogitsProcessorList()


def make_override_get_breakruns(base_temperature, tau, tokenizer=None, debug=False,
                                disable_trigger=None,
                                enable_trigger=None,
                                avoid_unk_caption=True,
                                ):
    def _override_get_breakruns(*args, **kwargs) -> LogitsProcessorList:
        processors = [
            BreakrunsLogitsProcessor(
                base_temperature=base_temperature,
                tau=tau,
                tokenizer=tokenizer,
                debug=debug,
                disable_trigger=disable_trigger,
                enable_trigger=enable_trigger,
            )
        ]
        if avoid_unk_caption:
            avoider = AvoidUnkCaptionLogitsProcessor()
            processors = [avoider] + processors
        return LogitsProcessorList(processors)
    return _override_get_breakruns


def make_override_get_typical_sampling(mass, min_tokens_to_keep):
    def _override_get_typical_sampling(*args, **kwargs) -> LogitsProcessorList:
        return LogitsProcessorList([
            TypicalLogitsWarper(
                mass=mass,
                min_tokens_to_keep=min_tokens_to_keep,
            )
        ])
    return _override_get_typical_sampling


class GeneratorModelTorch:
    def __init__(
        self,
        transformers_model,
        tokenizer,
        batch_size,
        device="cuda:0",
        sampling_params: SamplingParams = GPT_NEO_DEFAULT_SAMPLING_PARAMS,
        max_continue_tokens = MAX_CONTINUE_TOKENS,
        max_feed_size_with_cache = max_feed_size_with_cache,
        max_feed_size_no_cache = max_feed_size_no_cache,
        required_continuation_room = required_continuation_room,
    ):
        self.transformers_model = transformers_model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.sampling_params = sampling_params
        self.max_continue_tokens = max_continue_tokens
        self.max_feed_size_with_cache = max_feed_size_with_cache
        self.max_feed_size_no_cache = max_feed_size_no_cache
        self.required_continuation_room = required_continuation_room

        self.transformers_model = self.transformers_model.to(device)

        if self.sampling_params.breakruns:
            disable_trigger, enable_trigger = None, None
            if self.sampling_params.breakruns_off_within_images:
                disable_trigger = (1421, 18604, 198, 1421, 18604, 198,)
                enable_trigger = (1421, 18604, 198,)

            msg = f'using breakruns, base T={self.sampling_params.temperature}, tau={self.sampling_params.breakruns_tau}'
            msg += f', avoid_unk_caption={self.sampling_params.avoid_unk_caption}'
            msg += f', disable_trigger={disable_trigger}, enable_trigger={enable_trigger}'

            print(msg)
            breakruns_override = make_override_get_breakruns(
                base_temperature=self.sampling_params.temperature,
                tau=self.sampling_params.breakruns_tau,
                tokenizer=self.tokenizer if BREAKRUNS_DEBUG else None,
                debug=BREAKRUNS_DEBUG,
                avoid_unk_caption=self.sampling_params.avoid_unk_caption,
                disable_trigger=disable_trigger,
                enable_trigger=enable_trigger,
            )
            self.transformers_model._get_logits_processor = breakruns_override
        elif self.sampling_params.typical_sampling:
            print(f'using typical sampling, mass={self.sampling_params.typical_sampling_mass}, min_tokens_to_keep={self.sampling_params.typical_sampling_min_tokens_to_keep}')
            typical_sampling_override = make_override_get_typical_sampling(
                mass=self.sampling_params.typical_sampling_mass,
                min_tokens_to_keep=self.sampling_params.typical_sampling_min_tokens_to_keep,
            )
            self.transformers_model._get_logits_processor = override_disable_logits_processors
            self.transformers_model._get_logits_warper = typical_sampling_override

    @property
    def max_context_size(self):
        return self.max_feed_size_with_cache - self.required_continuation_room

    def set_past(self, past_key_values):
        for block, layer_past in zip(self.transformers_model.transformer.h, past_key_values):
            block.attn.attention.set_past(layer_past)

    def clear_past(self):
        for block in self.transformers_model.transformer.h:
            block.attn.attention.clear_past()

    def shift_past(self, offset):
        for block in self.transformers_model.transformer.h:
            block.attn.attention.shift_past(offset)

    def collect_past(self):
        past = []
        for block in self.transformers_model.transformer.h:
            pk = block.attn.attention.bufk[:, :, :block.attn.attention.seqlen, :]
            pv = block.attn.attention.bufv[:, :, :block.attn.attention.seqlen, :]
            layer_past = (pk, pv)
            past.append(layer_past)
        return tuple(past)

    @torch.no_grad()
    def compute_kv_cache(self, input_ids):
        input_ids = input_ids[:, -self.max_feed_size_with_cache:]

        full_len = input_ids.shape[1]
        # if full_len <= self.max_feed_size_no_cache:
        #     return input_ids, None

        print(f"Computing kv cache for length {full_len}")

        input_ids_no_cache = input_ids[:, :self.max_feed_size_no_cache]

        self.clear_past()

        if full_len <= self.max_feed_size_no_cache:
            presents = self.transformers_model(
                input_ids=input_ids_no_cache[:, :-1],
                use_cache=True,
            ).past_key_values
            self.set_past(presents)
        else:
            presents = self.transformers_model(
                input_ids=input_ids_no_cache,
                use_cache=True,
            ).past_key_values
            self.set_past(presents)

        for ix in range(self.max_feed_size_no_cache, full_len-1):
            presents = self.transformers_model(
                input_ids=input_ids[:, ix:ix+1],
                past_key_values=presents,
                use_cache=True,
            ).past_key_values

        print(f"Done computing kv cache for length {full_len}")

        return input_ids, presents

    def write_random_prompt(self, prompts: list, probs: list, verbose=False):
        prompt = np.random.choice(prompts, p=np.array(probs) / sum(probs))
        return self.write(prompt=prompt, verbose=verbose)

    @torch.no_grad()
    def write(self, prompt: str, verbose=False, max_length_per_feed=None):
        batch_pr = [prompt for _ in range(self.batch_size)]
        batch_pr_tokens = self.tokenizer(
            batch_pr,
        )["input_ids"]

        batch_pr_tokens = [toks[-self.max_context_size:] for toks in batch_pr_tokens]

        continuations_tokens = batch_pr_tokens
        n_orig_prompt_tokens = len(continuations_tokens[0])
        done = False

        input_ids = None
        past = None

        while not done:
            if input_ids is None:
                input_ids = self.tokenizer(
                    batch_pr,
                )["input_ids"]
            input_ids = [toks[-self.max_context_size:] for toks in input_ids]
            prompt_end_ix = len(input_ids[0])

            input_ids_th = torch.as_tensor(input_ids).to(self.device)

            if past is None:
                input_ids_th, past = self.compute_kv_cache(input_ids_th)

            if max_length_per_feed is not None:
                max_length_for_transformers_call = min(
                    self.max_feed_size_with_cache, max_length_per_feed + prompt_end_ix
                )
            else:
                max_length_for_transformers_call = min(self.max_feed_size_with_cache, self.max_continue_tokens + prompt_end_ix)

            out = self.transformers_model.generate(
                input_ids=input_ids_th,
                do_sample=True,
                use_cache=True,
                top_p=self.sampling_params.top_p,
                temperature=1 if self.sampling_params.breakruns else self.sampling_params.temperature,
                max_length=max_length_for_transformers_call,
                pad_token_id=self.tokenizer.pad_token_id,
                past=past,
            )
            hardcore_collect_and_show()

            input_ids = []
            dones = []
            for i, o in enumerate(out):
                # record the tokens
                extras = o[prompt_end_ix:].cpu().numpy()
                nonpads = [t for t in extras if t != self.tokenizer.pad_token_id]
                continuations_tokens[i].extend(nonpads)

                # is this one done?
                final_token = nonpads[-1]
                more_needed = final_token != self.tokenizer.eos_token_id

                n_continuations_tokens = (
                    len(continuations_tokens[i]) - n_orig_prompt_tokens
                )
                more_permitted = n_continuations_tokens < self.max_continue_tokens

                this_done = (not more_needed) or (not more_permitted)
                dones.append(this_done)

                print(f"this_done: {this_done}")
                print(f"\tmore_needed={more_needed} <-- final_token={final_token}")
                print(
                    f"\tmore_permitted={more_permitted} <-- n_continuations_tokens={n_continuations_tokens}, len(continuations_tokens[i])={len(continuations_tokens[i])}, n_orig_prompt_tokens={n_orig_prompt_tokens}"
                )

                if not this_done:
                    # construct next prompt
                    self.shift_past(self.required_continuation_room)
                    past = self.collect_past()

                    next_prompt_tokens = continuations_tokens[i][-self.max_context_size:]
                    print(f"next_prompt_tokens: {len(next_prompt_tokens)}")
                    input_ids.append(next_prompt_tokens)

            del out
            del input_ids_th
            hardcore_collect_and_show()

            done = all(dones)

        continuations_ = [
            self.tokenizer.decode(o[n_orig_prompt_tokens:])
            for o in continuations_tokens
        ]
        mirotarg = None  # for back compat
        miro_traces = {
            "surprise": [],
            "mu": [],
            "k": [],
            "tok": [],
        }  # for back compat

        return {
            "continuations": continuations_,
            "side_data": {
                "prompt_for_neural": prompt,
                "mirotarg": mirotarg,
                "miro_traces": miro_traces,
            },
        }

    def tok2str(t):
        if isinstance(t, int):
            return tokenizer.decode([t])
        return [tokenizer.decode([tok]) for tok in t]

    @torch.no_grad()
    def get_next_logits(self, text: str, to_numpy=True):
        input_ids = self.tokenizer([text])["input_ids"]
        input_ids = [input_ids[0][-self.max_context_size:]]
        input_ids_th = torch.as_tensor(input_ids).to(self.device)

        input_ids_th, past = self.compute_kv_cache(input_ids_th)

        logits = self.transformers_model(
            input_ids_th[:, -1:],
            past_key_values=past
        )['logits'][0, -1, :]

        if to_numpy:
            logits = logits.cpu().numpy()

        return logits

    def get_next_probs(self, text: str, forbidden_tokens: List[int] = None, to_numpy=True):
        logits = self.get_next_logits(text, to_numpy=False)

        if forbidden_tokens:
            logits[forbidden_tokens] = 0

        probs = torch.softmax(logits, dim=-1)

        if to_numpy:
            probs = probs.cpu().numpy()

        return probs

    def get_prob_delta_over_ref(self, text: str, text_ref: str, token_str: str,
                                forbidden_strings: List[str]):
        if token_str in forbidden_strings:
            return 0.

        token = self.tokenizer.encode(token_str)[0]

        forbidden_tokens = [self.tokenizer.encode(s)[0] for s in forbidden_strings]

        prob_ref = self.get_next_probs(text_ref, forbidden_tokens=[], to_numpy=True)[token]
        prob = self.get_next_probs(text, forbidden_tokens=forbidden_tokens, to_numpy=True)[token]

        delta = prob - prob_ref
        delta = float(delta)
        return delta

    def get_prob_delta_over_ref_multi(self, text: List[str], text_ref: List[str], token_str: str,
                                      forbidden_strings: List[List[str]]):
        return [self.get_prob_delta_over_ref(t, tr, token_str, fs)
                for t, tr, fs in zip(text, text_ref, forbidden_strings)]

    @staticmethod
    def load(
        transformers_model,
        tokenizer,
        batch_size,
        device="cuda:0",
        sampling_params: SamplingParams = GPT_NEO_DEFAULT_SAMPLING_PARAMS,
    ) -> "GeneratorModelTorch":

        model = GeneratorModelTorch(
            transformers_model=transformers_model,
            tokenizer=tokenizer,
            batch_size=batch_size,
            device=device,
            sampling_params=sampling_params,
        )
        return model
