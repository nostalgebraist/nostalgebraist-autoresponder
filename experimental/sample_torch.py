import weakref
import torch
from transformers import LogitsProcessor


class BreakrunsLogitsProcessor(LogitsProcessor):
    def __init__(self,
                 base_temperature: float,
                 tau: float
                 ):
        self.breakruns_counter = None
        self.last_logits = None
        self.base_temperature = base_temperature
        self.tau = tau

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.breakruns_counter is None:
            self.breakruns_counter = torch.as_tensor(1, device=input_ids.device)
            
        if self.last_logits is None:
            self.last_logits = scores
