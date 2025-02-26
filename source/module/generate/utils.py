import os
import torch
import numpy as np

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList

class EOSReachedCriteria(StoppingCriteria):
    # Use this when EOS is not a single id, but a sequence of ids, e.g. for a custom EOS text.
    def __init__(
        self, 
        tokenizer: AutoTokenizer, 
        eos_text: str
    ):
        self.tokenizer = tokenizer
        self.eos_text = eos_text
        assert len(self.tokenizer.encode(eos_text)) < 10, (
            "EOS text can't be longer then 10 tokens. It makes stopping_criteria check slow."
        )

    def __call__(
        self, 
        input_ids: torch.LongTensor, 
        scores: torch.FloatTensor, 
        **kwargs
    ) -> bool:
        decoded_text = self.tokenizer.decode(input_ids[0][-10:])
        condition1 = decoded_text.endswith(self.eos_text)
        condition2 = decoded_text.strip().endswith(self.eos_text.strip())
        
        return condition1 or condition2