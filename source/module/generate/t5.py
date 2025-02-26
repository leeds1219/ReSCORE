import os
import torch
import numpy as np
import spacy
import gc
from typing import List, Literal, Optional, Any
from dataclasses import dataclass
import spacy
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from math import exp
from .utils import EOSReachedCriteria
from .base import BaseGenerator, BaseGeneratorConfig


PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
FORCE_RESET = bool(int(os.getenv("FORCE_RESET", "0")))

@dataclass
class T5GeneratorConfig(BaseGeneratorConfig):
    # Base Setting
    model_name: Optional[
        Literal[
            'google/flan-t5-base',
            'google/flan-t5-large',
            'google/flan-t5-xl',
            'google/flan-t5-xxl',
        ]
    ] = 'google/flan-t5-xl'
    max_total_tokens: Optional[int] = 6000
    max_new_tokens: Optional[int] = 200
    min_new_tokens: Optional[int] = 1
    # Wheather to use sampling
    do_sample: Optional[bool] = False
    # If use Sampling
    temperature: Optional[float] = 1.0
    top_k: Optional[float] = 50
    top_p: Optional[float] = 1.0
    num_return_sequences: Optional[int] = 1
    # If use Greedy decoding
    repetition_penalty: Optional[float] = None
    length_penalty: Optional[float] = None
    # Tokenizer
    truncation: Optional[bool] = True
    padding: Optional[bool] = True
    # Etc
    eos_text: Optional[str] = None
    vocab_size: Optional[int] = 32128
    

class T5Generator(BaseGenerator):
    
    def __init__(
        self,
        cfg: T5GeneratorConfig = T5GeneratorConfig()
    ):
        super().__init__(cfg)
        if torch.cuda.device_count() == 2:
            hf_device_map = {"shared": 1, "encoder": 0, "decoder": 1, "lm_head": 1}
        else:
            hf_device_map = "auto"
            
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.cfg.model_name, 
            device_map=hf_device_map
        )
        self.model = self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.model_name, 
        )
        self.pad_token_initialized = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': "<PAD>"})
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.pad_token_initialized = True
            
        if self.cfg.eos_text:
            self.stopping_criteria_list = EOSReachedCriteria(
                tokenizer=self.tokenizer,
                eos_text=self.cfg.eos_text
            )
        else:
            self.stopping_criteria_list = None
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='none')
    
    @torch.no_grad()
    def _generate(
        self,
        input_texts = List[str],
    ):
        model_inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=self.cfg.max_total_tokens,
            truncation=self.cfg.truncation,
            padding=self.cfg.padding
        )
        model_inputs = {
            k:v.cuda() 
            for k, v in model_inputs.items()
        }
        generated_texts = self.tokenizer.batch_decode(
            self.model.generate(
                **model_inputs,
                max_new_tokens=self.cfg.max_new_tokens,
                min_length=self.cfg.min_new_tokens,
                do_sample=self.cfg.do_sample,
                temperature=self.cfg.temperature,
                top_k=self.cfg.top_k,
                top_p=self.cfg.top_p,
                num_return_sequences=self.cfg.num_return_sequences,
                repetition_penalty=self.cfg.repetition_penalty,
                length_penalty=self.cfg.length_penalty,
                stopping_criteria=self.stopping_criteria_list,
                output_scores=False, 
            ), skip_special_tokens=True
        )
        
        outputs = generated_texts
        
        return outputs
    
    @torch.no_grad()
    def _score(
        self, 
        input_texts: List[str], 
        output_texts: List[str],
        method: Literal['perplexity_score'] = 'perplexity_score'
    ):  
        model_inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            max_length=self.cfg.max_total_tokens,
            truncation=self.cfg.truncation,
            padding=self.cfg.padding
        )
        model_inputs = {
            k:v.cuda() 
            for k, v in model_inputs.items()
        }

        model_outputs = self.tokenizer(
            output_texts,
            return_tensors="pt",
            max_length=self.cfg.max_new_tokens,
            truncation=self.cfg.truncation,
            padding=self.cfg.padding
        )
        model_outputs = {
            k:v.cuda() 
            for k, v in model_outputs.items()
        }

        labels = model_outputs["input_ids"]
        labels[labels == self.tokenizer.pad_token_id] = PAD_TOKEN_LABEL_ID

        # Inference
        model_inputs["labels"] = labels
        output = self.model(**model_inputs)
        
        # model run & loss conversion into likelihood
        logits = output['logits']
        if self.pad_token_initialized:
            logits = logits[:, :, :-1]
            
        valid_length = (model_inputs["labels"] != PAD_TOKEN_LABEL_ID).sum(dim=-1)
        loss = self.loss_fn(
            logits.view(-1, self.cfg.vocab_size),
            model_inputs["labels"].view(-1)
        )
        loss = torch.sum(loss.view(len(logits), -1), -1) / valid_length

        if FORCE_RESET:
            del model_inputs
            del loss
            del output
            gc.collect()
            torch.cuda.empty_cache()

        # conversion to perplexity
        ppl = [exp(i) for i in loss]
        
        return ppl
    
    
if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    
    cfg = T5GeneratorConfig(
        batch_size=1,
        model_name='google/flan-t5-base',
        max_total_tokens=6000,
        max_new_tokens=200,
        min_length=1,
        do_sample=False,
        generation_strategy='sentence'
    )
    model = T5Generator(cfg)
    
    sample_inputs = [
        'Transformer-based GPTs have become extremely popular in 2024. What is the most popular deep learning architecture in 2024?',
        'Diffusion-based DALL-E have become extremely popular in 2024. What is the most popular deep learning architecture in 2024?',
        'What is the most popular deep learning architecture in 2024?'
    ]
    
    sample_outputs = model.generate(
        sample_inputs
    )
    
    sample_forced_outputs = [
        'So the answer is: Transformer',
        'So the answer is: Transformer',
        'So the answer is: Transformer'
    ]
    
    scores = model.score(
        input_texts=sample_inputs,
        output_texts=sample_forced_outputs
    )
    
    print("\nINFERENCE RESULTS\n")
    """
    ('Transformer-based GPTs have become extremely popular in 2024. What is the most popular deep learning architecture in 2024?', 'transformer-based GPTs')
    ('Diffusion-based DALL-E have become extremely popular in 2024. What is the most popular deep learning architecture in 2024?', 'Diffusion-based DALL-E')
    ('What is the most popular deep learning architecture in 2024?', 'linus')
    """
    for example in zip(sample_inputs, sample_outputs):
        print(example)
    
    print("\nPEREPLEXITY RESULTS\n")
    """
    ('Transformer-based GPTs have become extremely popular in 2024. What is the most popular deep learning architecture in 2024?', 'So the answer is: Transformer', 133.51657098713227)
    ('Diffusion-based DALL-E have become extremely popular in 2024. What is the most popular deep learning architecture in 2024?', 'So the answer is: Transformer', 791.8531919197658)
    ('What is the most popular deep learning architecture in 2024?', 'So the answer is: Transformer', 671.3740666226772)
    """
    for example in zip(sample_inputs, sample_forced_outputs, scores):
        print(example)
        
    