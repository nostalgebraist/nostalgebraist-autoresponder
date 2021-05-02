import weakref

import torch
import numpy as np

from autoresponder_config import *  # TODO: move elsewhere?
from experimental.sampling_params import SamplingParams, DEFAULT_SAMPLING_CONFIG

GPT_NEO_DEFAULT_SAMPLING_PARAMS = copy_and_update_config(
    SamplingParams,
    DEFAULT_SAMPLING_CONFIG.params,
    temperature=GPT_NEO_T,
    top_p=GPT_NEO_TOP_P,
    top_k=GPT_NEO_TOP_K
)


class GPTNeoGeneratorModel:
    def __init__(
        self,
        transformers_model,
        tokenizer,
        batch_size,
        device="cuda:0",
        sampling_params: SamplingParams = GPT_NEO_DEFAULT_SAMPLING_PARAMS,
    ):
        self._transformers_model = weakref.ref(transformers_model)
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.device = device
        self.sampling_params = sampling_params

        self.transformers_model = self.transformers_model.to(device)

    @property
    def transformers_model(self):
        return self._transformers_model()

    def write_random_prompt(
        self, prompts: list, probs: list, verbose=False
    ):
        prompt = np.random.choice(prompts, p=np.array(probs) / sum(probs))
        return self.write(prompt=prompt, verbose=verbose)

    def write(self, prompt: str, verbose=False):
        batch_pr = [pr for _ in range(self.batch_size)]

        input_ids = self.tokenizer(batch_pr)['input_ids']
        prompt_end_ix = len(input_ids)[0]

        input_ids_th = torch.as_tensor(input_ids).to(self.device)

        out = self.transformers_model.generate(
            input_ids=input_ids_th,
            do_sample=True,
            top_p=self.sampling_params.top_p,
            temperature=self.sampling_params.temperature,
            top_k=self.sampling_params.top_k,
            max_length=GPT_NEO_MAX_LENGTH,
        )

        continuations_ = [tokenizer.decode(o[prompt_end_ix:]) for o in out]
        mirotarg = None  # for back compat
        miro_traces = {
            "surprise": [],
            "mu": [],
            "k": [],
            'tok': [],
        }  # for back compat

        return {
            "continuations": continuations_,
            "side_data": {
                "prompt_for_neural": prompt,
                "mirotarg": mirotarg,
                "miro_traces": miro_traces,
            },
        }
