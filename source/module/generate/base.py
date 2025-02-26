from abc import ABC, abstractmethod
from typing import List, Any, Optional, Literal
from dataclasses import dataclass
import torch
from tqdm import tqdm

@dataclass
class BaseGeneratorConfig:
    batch_size: Optional[int] = 4 # 
    # tensor_parallel_size: Optional[int] = None

class BaseGenerator(ABC):
    
    def __init__(
        self, 
        cfg: BaseGeneratorConfig
    ):
        self.cfg = cfg
    
    @abstractmethod
    def _generate(
        self, 
        input_texts: List[str]
    ) -> List[str]:
        pass
    
    @abstractmethod
    def _score(
        self, 
        input_texts: List[str], 
        output_texts: List[str], 
        method: Literal['perplexity_score'] = 'perplexity_score'
    ) -> Any:
        pass
    
    def generate(
        self, 
        input_texts: List[str],
        verbose: Optional[bool] = False
    ) -> List[str]:
        
        assert type(input_texts) == list, (
            f"`input_texts` must be type of list. But {type(input_texts)}"
        )
        
        outputs = []
        total_batches = (len(input_texts) + self.cfg.batch_size - 1) // self.cfg.batch_size  # 전체 배치 수 계산
        
        pbar = tqdm(
            range(0, len(input_texts), self.cfg.batch_size), 
            desc="Generating...", 
            total=total_batches
        ) if verbose else range(0, len(input_texts), self.cfg.batch_size)

        for i in pbar:
            batched_input_texts = input_texts[i:i + self.cfg.batch_size]
            outputs.extend(self._generate(batched_input_texts))
                
        return outputs
        
    def score(
        self, 
        input_texts: List[str], 
        output_texts: List[str], 
        method: Literal['perplexity_score'] = 'perplexity_score'
    ) -> List[float]:
        
        assert type(input_texts) == list, (
            f"`input_texts` must be type of list. But {type(input_texts)}"
        )
        assert type(output_texts) == list, (
            f"`input_texts` must be type of list. But {type(output_texts)}"
        )
        assert len(input_texts) == len(output_texts), (
            f"Length of `input_texts` and `output_texts` must be same. But {len(input_texts)} != {len(output_texts)}"
        )
        
        outputs = []
        total_batches = (len(input_texts) + self.cfg.batch_size - 1) // self.cfg.batch_size  # 전체 배치 수 계산

        pbar = tqdm(
            range(0, len(input_texts), self.cfg.batch_size), 
            desc="Scoring...", 
            total=total_batches
        )
        
        for i in pbar:
            batched_input_texts = input_texts[i:i + self.cfg.batch_size]
            batched_output_texts = output_texts[i:i + self.cfg.batch_size]
            outputs.extend(self._score(batched_input_texts, batched_output_texts, method))
                
        return outputs
    
    def __call__(
        self,
        input_texts: List[str],
    ) -> List[str]:
        
        return self.generate(input_texts)