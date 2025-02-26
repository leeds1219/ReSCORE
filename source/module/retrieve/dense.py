import os
import torch
import transformers
from transformers import BertModel, XLMRobertaModel, AutoModel, AutoTokenizer
from dataclasses import dataclass
from typing import Optional, Literal, List, Dict, Any, Union
from numpy.typing import ArrayLike

from .base import BaseRetrieverConfig, BaseRetriever

import copy


@dataclass
class DenseRetrieverConfig(BaseRetrieverConfig):
    query_model_name_or_path: Optional[
        Union[
            Literal[
                'facebook/mcontriever',
                'facebook/contriever',
                'facebook/mcontriever-msmarco',
                'facebook/contriever-msmarco'
            ],
            str
        ]
    ] = 'facebook/contriever-msmarco'
    passage_model_name_or_path: Optional[
        Union[
            Literal[
                'facebook/mcontriever',
                'facebook/contriever',
                'facebook/mcontriever-msmarco',
                'facebook/contriever-msmarco'
            ],
            str
        ]
    ] = None
    pooling: Optional[Literal['average', 'cls']] = 'average'
    max_length: Optional[int] = 512
    normalize: Optional[bool] = False


class DenseRetriever(BaseRetriever):
    
    def __init__(
        self,
        cfg: DenseRetrieverConfig = DenseRetrieverConfig()
    ):
        self.cfg = cfg
        
        assert self.cfg.query_model_name_or_path != None, (
            "You must specify query model name."
        )
        
        self.query_model = AutoModel.from_pretrained(
            self.cfg.query_model_name_or_path
        )
        self.query_tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.query_model_name_or_path
        )
        self.query_model = self.query_model.cuda()
        if self.cfg.passage_model_name_or_path != None:
            self.query_model = AutoModel.from_pretrained(
                self.cfg.passage_model_name_or_path
            )
            self.query_tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.passage_model_name_or_path
            )
            self.passage_model = self.passage_model.cuda()
        else:
            self.passage_model = copy.deepcopy(self.query_model)
            self.passage_tokenizer = self.query_tokenizer
        
            
        if self.cfg.training_strategy == 'both':
            self.query_model = self.query_model.train()
            self.passage_model = self.passage_model.train()
            
        elif self.cfg.training_strategy == 'query_only':
            self.query_model = self.query_model.train()
            self.passage_model = self.passage_model.eval()
            
            if self.cfg.use_fp16:
                self.passage_model = self.passage_model.half()
            
        else:
            self.query_model = self.query_model.eval()
            self.passage_model = self.passage_model.eval()
            
            if self.cfg.use_fp16:
                self.query_model = self.query_model.half()
                self.passage_model = self.passage_model.half()

    def _embed_passages(
        self,
        input_texts: List[str],
    ) -> Any:
        
        model_inputs = self.passage_tokenizer.batch_encode_plus(
            input_texts,
            return_tensors="pt",
            max_length=self.cfg.max_length,
            padding=True,
            truncation=True,
        )
        model_inputs = {
            k:v.cuda() 
            for k, v in model_inputs.items()
        }
        model_outputs = self.passage_model(**model_inputs)
        
        embeddings = pooling(
            token_embeddings=model_outputs[0],
            mask=model_inputs['attention_mask'],
            pooling=self.cfg.pooling,
            normalize=self.cfg.normalize
        )
        
        return embeddings
    
    def _embed_queries(
        self,
        input_texts: List[str],
    ) -> Any:
        
        model_inputs = self.query_tokenizer.batch_encode_plus(
            input_texts,
            return_tensors="pt",
            max_length=self.cfg.max_length,
            padding=True,
            truncation=True,
        )
        model_inputs = {
            k:v.cuda() 
            for k, v in model_inputs.items()
        }
        model_outputs = self.query_model(**model_inputs)
        
        embeddings = pooling(
            token_embeddings=model_outputs[0],
            mask=model_inputs['attention_mask'],
            pooling=self.cfg.pooling,
            normalize=self.cfg.normalize
        )
        
        return embeddings
        
    
def pooling(
    token_embeddings, 
    mask,
    pooling: Optional[Literal['average', 'cls']] = 'average',
    normalize: Optional[bool] = False
):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
    
    if pooling == "average":
        emb = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    elif pooling == "cls":
        emb = token_embeddings[:, 0]
        
    if normalize:
        emb = torch.nn.functional.normalize(emb, dim=-1)
        
    return emb


if __name__ == '__main__':
    cfg = DenseRetrieverConfig(
        batch_size=2,
        training_strategy='query_only',
        use_fp16=True,
        query_model_name_or_path='facebook/contriever-msmarco',
    )
    
    retriever = DenseRetriever(
        cfg=cfg
    )
    
    sample_inputs = [
        'Transformer-based GPTs have become extremely popular in 2024. What is the most popular deep learning architecture in 2024?',
        'Diffusion-based DALL-E have become extremely popular in 2024. What is the most popular deep learning architecture in 2024?',
        'What is the most popular deep learning architecture in 2024?'
    ]
    
    print('\nQUERY EMBEDDING TEST\n')
    
    q_embed = retriever.embed(
        sample_inputs,
        input_type='query'
    )
    print(q_embed.shape)
    
    print('\nPASSAGE EMBEDDING TEST\n')
    
    p_embed = retriever.embed(
        sample_inputs,
        input_type='passage'
    )
    print(p_embed.shape)