from typing import List

import numpy as np
import torch

from config.autoresponder_config import *  # TODO: move elsewhere?
from ml.sampling_params import SamplingParams, DEFAULT_SAMPLING_CONFIG

from transformers import LogitsProcessorList
from ml.sample_torch import BreakrunsLogitsProcessor

from util.util import copy_and_update_config, collect_and_show

GPT_NEO_DEFAULT_SAMPLING_PARAMS = copy_and_update_config(
    SamplingParams,
    DEFAULT_SAMPLING_CONFIG.params,
    temperature=GPT_NEO_T,
    top_p=GPT_NEO_TOP_P,
    top_k=GPT_NEO_TOP_K,
)


def is_repeating_criterion(unique_token_frac):
    return unique_token_frac < 0.2


def hardcore_collect_and_show():
    collect_and_show()
    torch.cuda.empty_cache()


def make_override_get_breakruns(base_temperature, tau, tokenizer=None, debug=False):
    def _override_get_breakruns(*args, **kwargs) -> LogitsProcessorList:
        return LogitsProcessorList([
            BreakrunsLogitsProcessor(
                base_temperature=base_temperature,
                tau=tau,
                tokenizer=tokenizer,
                debug=debug
            )
        ])
    return _override_get_breakruns


class GeneratorModelTorch:
    def __init__(
        self,
        transformers_model,
        tokenizer,
        batch_size,
        device="cuda:0",
        sampling_params: SamplingParams = GPT_NEO_DEFAULT_SAMPLING_PARAMS,
    ):
        self.transformers_model = transformers_model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.sampling_params = sampling_params

        self.transformers_model = self.transformers_model.to(device)

        if self.sampling_params.breakruns:
            breakruns_override = make_override_get_breakruns(
                base_temperature=self.sampling_params.temperature,
                tau=self.sampling_params.breakruns_tau,
                tokenizer=self.tokenizer if BREAKRUNS_DEBUG else None,
                debug=BREAKRUNS_DEBUG)
            self.transformers_model._get_logits_processor = breakruns_override

    def write_random_prompt(self, prompts: list, probs: list, verbose=False):
        prompt = np.random.choice(prompts, p=np.array(probs) / sum(probs))
        return self.write(prompt=prompt, verbose=verbose)

    def write(self, prompt: str, verbose=False, max_length_per_feed=None):
        max_context_size = GPT_NEO_MAX_LENGTH - required_continuation_room

        batch_pr = [prompt for _ in range(self.batch_size)]
        batch_pr_tokens = self.tokenizer(
            batch_pr,
        )["input_ids"]

        batch_pr_tokens = [toks[-max_context_size:] for toks in batch_pr_tokens]

        continuations_tokens = batch_pr_tokens
        n_orig_prompt_tokens = len(continuations_tokens[0])
        done = False

        while not done:
            input_ids = self.tokenizer(
                batch_pr,
            )["input_ids"]
            input_ids = [toks[-max_context_size:] for toks in input_ids]
            prompt_end_ix = len(input_ids[0])

            input_ids_th = torch.as_tensor(input_ids).to(self.device)

            if max_length_per_feed is not None:
                max_length_for_transformers_call = min(
                    GPT_NEO_MAX_LENGTH, max_length_per_feed + prompt_end_ix
                )
            else:
                max_length_for_transformers_call = GPT_NEO_MAX_LENGTH

            out = self.transformers_model.generate(
                input_ids=input_ids_th,
                do_sample=True,
                use_cache=True,
                top_p=self.sampling_params.top_p,
                temperature=1 if self.sampling_params.breakruns else self.sampling_params.temperature,
                max_length=max_length_for_transformers_call,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            hardcore_collect_and_show()

            next_prompts = []
            dones = []
            for i, o in enumerate(out):
                # record the tokens
                extras = o[prompt_end_ix:].cpu().numpy()
                nonpads = [t for t in extras if t != self.tokenizer.pad_token_id]
                continuations_tokens[i].extend(nonpads)

                # construct next prompt
                next_prompt_tokens = continuations_tokens[i][-max_context_size:]
                next_prompt = self.tokenizer.decode(next_prompt_tokens)
                next_prompts.append(next_prompt)
                print(f"next_prompt_tokens: {len(next_prompt_tokens)}")

                # is this one done?
                final_token = nonpads[-1]
                more_needed = final_token != self.tokenizer.eos_token_id

                n_continuations_tokens = (
                    len(continuations_tokens[i]) - n_orig_prompt_tokens
                )
                more_permitted = n_continuations_tokens < MAX_CONTINUE_TOKENS

                this_done = (not more_needed) or (not more_permitted)
                dones.append(this_done)

                print(f"this_done: {this_done}")
                print(f"\tmore_needed={more_needed} <-- final_token={final_token}")
                print(
                    f"\tmore_permitted={more_permitted} <-- n_continuations_tokens={n_continuations_tokens}, len(continuations_tokens[i])={len(continuations_tokens[i])}, n_orig_prompt_tokens={n_orig_prompt_tokens}"
                )

            del out
            del input_ids_th
            hardcore_collect_and_show()

            batch_pr = next_prompts
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

    def get_next_logits(self, text: str, to_numpy=True):
        input_ids = self.tokenizer([text])["input_ids"]
        input_ids_th = torch.as_tensor(input_ids).to(self.device)

        with torch.no_grad():
            logits = self.transformers_model(input_ids_th)['logits'][0, -1, :]

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

    def get_prob_delta_over_ref(self, text: str, text_ref: str, token_str: str, forbidden_tokens_str: List[str]):
        token = self.tokenizer.encode(token_str)[0]
        forbidden_tokens = [self.tokenizer.encode(s)[0] for s in forbidden_tokens_str]

        prob_ref = self.get_next_probs(text_ref, forbidden_tokens=[], to_numpy=True)[token]
        prob = self.get_next_probs(text, forbidden_tokens=forbidden_tokens, to_numpy=True)[token]

        return prob - prob_ref

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
