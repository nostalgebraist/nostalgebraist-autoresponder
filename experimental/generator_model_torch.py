import torch
import numpy as np

from autoresponder_config import *  # TODO: move elsewhere?
from experimental.sampling_params import SamplingParams, DEFAULT_SAMPLING_CONFIG
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

    def _get_past_using_padding(self, input_ids, pad_to_mult=256):
        true_len = input_ids.shape[1]

        pad_to_len = pad_to_mult * (true_len // pad_to_mult + 1)

        pad_len = pad_to_len - true_len

        pads = torch.zeros(
            input_ids.shape[0], pad_len, dtype=input_ids.dtype, device=input_ids.device
        )

        input_ids_padded = torch.cat([input_ids, pads], dim=1)

        pkv = self.transformers_model(input_ids=input_ids_padded)["past_key_values"]

        pkv_clipped = tuple(tuple(y[..., : true_len - 1, :] for y in x) for x in pkv)

        return pkv_clipped

    def write_random_prompt(self, prompts: list, probs: list, verbose=False):
        prompt = np.random.choice(prompts, p=np.array(probs) / sum(probs))
        return self.write(prompt=prompt, verbose=verbose)

    def write(self, prompt: str, verbose=False, max_length_per_feed=None):
        max_context_size = GPT_NEO_MAX_LENGTH - required_continuation_room

        batch_pr = [prompt for _ in range(self.batch_size)]
        batch_pr_tokens = self.tokenizer(
            batch_pr, truncation=True, max_length=max_context_size
        )["input_ids"]

        continuations_tokens = batch_pr_tokens
        n_orig_prompt_tokens = len(continuations_tokens[0])
        done = False

        while not done:
            input_ids = self.tokenizer(
                batch_pr, truncation=True, max_length=max_context_size
            )["input_ids"]
            prompt_end_ix = len(input_ids[0])

            input_ids_th = torch.as_tensor(input_ids).to(self.device)

            past_key_values = self._get_past_using_padding(input_ids_th)

            if max_length_per_feed is not None:
                max_length_for_transformers_call = min(
                    GPT_NEO_MAX_LENGTH, max_length_per_feed + prompt_end_ix
                )
            else:
                max_length_for_transformers_call = GPT_NEO_MAX_LENGTH

            out = self.transformers_model.generate(
                input_ids=input_ids_th[:, -1:],
                past_key_values=past_key_values,
                do_sample=True,
                top_p=self.sampling_params.top_p,
                temperature=self.sampling_params.temperature,
                top_k=self.sampling_params.top_k,
                max_length=max_length_for_transformers_call,
                pad_token_id=self.tokenizer.pad_token_id,
            )

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
            collect_and_show()

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

    def done_writing(self, prompt: str):
        pass

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
