import torch
from transformers import LogitsProcessor


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
