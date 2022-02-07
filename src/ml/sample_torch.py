import torch
from transformers import LogitsProcessor, LogitsWarper


class BreakrunsLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 base_temperature: float,
                 tau: float,
                 debug=True,
                 tokenizer=None
                 ):
        self.breakruns_counter = None
        self.last_logits = None
        self.base_temperature = base_temperature
        self.tau = tau
        self.debug = debug
        self.tokenizer = tokenizer

    def _dprint(self, msg, fillers={}, **kwargs):
        if self.debug:
            print(msg.format(**fillers), **kwargs)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        seq_length = input_ids.shape[1]
        if seq_length < 1:
            self._dprint("BREAKRUNS: empty sequence, no op")
            return scores

        if self.breakruns_counter is None:
            self._dprint("BREAKRUNS: init counter")
            self.breakruns_counter = torch.zeros((), device=input_ids.device)

        if self.last_logits is None:
            self._dprint("BREAKRUNS: init logits, no op")
            self.last_logits = scores

            return scores

        # check if last was top
        was_top = (input_ids[:, -1] == self.last_logits.argmax(dim=1)).to(torch.long)

        self.breakruns_counter = was_top * (self.breakruns_counter + 1)

        if self.debug:
            sampled_str = repr(self.tokenizer.decode(input_ids[:, -1].item()))
            actual_top_str = repr(self.tokenizer.decode([self.last_logits.argmax(dim=1).item()]))
            print(f"was_top?: {was_top} | sampled {sampled_str} actual_top {actual_top_str} | self.breakruns_counter: {self.breakruns_counter}")

        eff_temperature = self.base_temperature + (self.breakruns_counter * self.tau)
        self._dprint("eff_temperature: {et}", fillers={"et": eff_temperature})

        self.last_logits = scores

        return scores / eff_temperature


# taken from https://github.com/cimeister/typical-sampling/blob/typical-pr/src/transformers/generation_logits_process.py
# implements method from https://arxiv.org/abs/2202.00666
class TypicalLogitsWarper(LogitsWarper):
    def __init__(self, mass: float = 0.9, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):

        self.filter_value = filter_value
        self.mass = mass
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        # calculate entropy
        normalized = torch.nn.functional.log_softmax(scores, dim=-1)
        p = torch.exp(normalized)
        ent = -(normalized * p).nansum(-1, keepdim=True)

        # shift and sort
        shifted_scores = torch.abs((-normalized) - ent)
        sorted_scores, sorted_indices = torch.sort(shifted_scores, descending=False)
        sorted_logits = scores.gather(-1, sorted_indices)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative mass above the threshold
        last_ind = (cumulative_probs < self.mass).sum(dim=1)
        last_ind[last_ind < 0] = 0
        sorted_indices_to_remove = sorted_scores > sorted_scores.gather(1, last_ind.view(-1, 1))
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores
