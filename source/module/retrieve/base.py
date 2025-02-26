from abc import ABC, abstractmethod
from typing import List, Any, Optional, Literal
from dataclasses import dataclass
from numpy.typing import ArrayLike
import torch
from tqdm import tqdm


@dataclass
class BaseRetrieverConfig:
    batch_size: Optional[int] = 4
    training_strategy: Optional[Literal['query_only', 'both']] = 'query_only'
    use_fp16: Optional[bool] = True
    return_tensors: Literal['pt', 'np'] = 'pt'


class BaseRetriever(ABC):
    
    def __init__(
        self,
        cfg: BaseRetrieverConfig,
    ):

        assert (cfg.training_strategy is not None) and (cfg.return_tensors != 'np'), (
            "Return must be pytorch tensor to use at train."
        )
        
        self.cfg = None
        self.query_model = None
        self.query_tokenizer = None
        self.passage_model = None
        self.passage_tokenizer = None
    
    @abstractmethod
    def _embed_queries(
        self,
        input_texts: List[str],
    ) -> Any:
        pass
    
    @abstractmethod
    def _embed_passages(
        self,
        input_texts: List[str],
    ) -> Any:
        pass
    
    def embed(
        self,
        input_texts: List[str],
        input_type: Optional[Literal['query', 'passage']] = 'query',
        verbose: Optional[bool] = False
    ) -> Any:
        
        assert type(input_texts) == list, (
            f"`input_texts` must be type of list. But {type(input_texts)}"
        )
        
        outputs = []
        total_batches = (len(input_texts) + self.cfg.batch_size - 1) // self.cfg.batch_size  # 전체 배치 수 계산
        
        if input_type == 'query':
            fn = self._embed_queries
        else:
            fn = self._embed_passages
        
        need_grad = False
        if ((input_type == 'query' and self.cfg.training_strategy == 'query_only') |
            (self.cfg.training_strategy == 'both')):
            need_grad = True
            
        pbar = tqdm(
            range(0, len(input_texts), self.cfg.batch_size), 
            desc="Retrieval...", 
            total=total_batches
        ) if verbose else range(0, len(input_texts), self.cfg.batch_size)
        
        for i in pbar:
            batched_input_texts = input_texts[i:i + self.cfg.batch_size]
            if need_grad:
                outputs.append(fn(batched_input_texts))
            else:
                with torch.no_grad():
                    outputs.append(fn(batched_input_texts))
        
        outputs = torch.concat(outputs, dim=0)
        
        return outputs
    
    def __call__(
        self,
        input_texts: List[str],
        input_type: Optional[Literal['query', 'passage']] = 'query'
    ) -> List[str]:
        
        return self.embed(input_texts, input_type)