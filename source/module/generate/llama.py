import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import gc
from typing import List, Literal, Optional, Any
from dataclasses import dataclass, field
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
import torch
from math import exp
from source.module.generate.utils import EOSReachedCriteria
from source.module.generate.base import BaseGenerator, BaseGeneratorConfig

PAD_TOKEN_LABEL_ID = torch.nn.CrossEntropyLoss().ignore_index
FORCE_RESET = bool(int(os.getenv("FORCE_RESET", "0")))

@dataclass
class LlamaGeneratorConfig(BaseGeneratorConfig):
    # Base Setting
    model_name: Optional[
        Literal[
            'meta-llama/Meta-Llama-3.1-8B',
            'meta-llama/Meta-Llama-3.1-8B-Instruct',
        ]
    ] = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
    max_total_tokens: Optional[int] = 4096
    max_new_tokens: Optional[int] = 1024
    min_new_tokens: Optional[int] = 1

    # If use Sampling
    temperature: Optional[float] = 0.
    # top_k: Optional[float] = 50
    # top_p: Optional[float] = 1.0
    num_return_sequences: Optional[int] = 1
    # If use Greedy decoding
    repetition_penalty: Optional[float] = 1.0
    length_penalty: Optional[float] = 1.0
    # Tokenizer
    truncation: Optional[bool] = True
    padding: Optional[bool] = True
    # Etc
    stop: Optional[str] = field(default_factory=list)
    include_stop_str_in_output: Optional['bool'] = True
    gpu_memory_utilization: Optional[float] = 0.8
    # vocab_size: Optional[int] = 128256 # TODO: llama vocab_size? 128256 by default
    use_vllm: Optional[bool] = True
    eos_text: Optional[str] = None
    gpu : Optional[int] = None
    

class LlamaGenerator(BaseGenerator):
    
    def __init__(
        self,
        cfg: LlamaGeneratorConfig = LlamaGeneratorConfig()
    ):
        super().__init__(cfg)

        if self.cfg.gpu:
            self.device = torch.device(f'cuda:{self.cfg.gpu}' if torch.cuda.is_available() else 'cpu')

        if self.cfg.use_vllm: 
            self.model = LLM( 
                model=self.cfg.model_name, 
                gpu_memory_utilization=self.cfg.gpu_memory_utilization, 
                max_model_len=self.cfg.max_total_tokens, 
                tensor_parallel_size=torch.cuda.device_count(),
                device=self.device,
                # enable_prefix_caching=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_name, 
            )

        else:
            # self.hf_device_map = "auto"
            self.hf_device_map = { "": self.device }
            self.model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model_name,
                device_map=self.hf_device_map
            )
            self.model.eval()  # Set the model to evaluation mode
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_name, 
            )
            self.pad_token_initialized = False
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': "<PAD>"})
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.pad_token_initialized = True
            
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.tokenizer.padding_side = 'left'  # Set padding side to left for decoder model
            
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
        inputs = List[str],
    ):
        if self.cfg.use_vllm:
            sampling_params = SamplingParams(
                n=self.cfg.num_return_sequences,
                repetition_penalty=self.cfg.repetition_penalty,
                temperature=self.cfg.temperature,
                # top_p=self.cfg.top_p,
                # top_k=self.cfg.top_k,
                length_penalty=self.cfg.length_penalty,
                stop=[self.tokenizer.eos_token, "<|eot_id|>"] + self.cfg.stop,
                include_stop_str_in_output=self.cfg.include_stop_str_in_output,
                max_tokens=self.cfg.max_new_tokens, # Maximum number of tokens to generate per output sequence.
                min_tokens=self.cfg.min_new_tokens, # min_tokens
            )
            model_outputs = self.model.generate(
                prompts=inputs, 
                sampling_params=sampling_params
            )
            generated_texts = [model_output.outputs[0].text for model_output in model_outputs]
            
        else: 
            model_inputs = self.tokenizer(
                inputs,
                return_tensors="pt",
                max_length=self.cfg.max_total_tokens,
                truncation=self.cfg.truncation,
                padding=self.cfg.padding
            )
            model_inputs = {
                k:v.cuda() 
                for k, v in model_inputs.items()
            }
            input_ids_length = model_inputs['input_ids'].shape[1]
            # Define the generation configuration
            generation_args = { 
                "max_new_tokens":self.cfg.max_new_tokens, 
                "min_length": self.cfg.min_new_tokens,
                "do_sample": True if self.cfg.temperature != 0. else False,
                "temperature": self.cfg.temperature,
                "num_return_sequences": self.cfg.num_return_sequences,                
                "repetition_penalty": self.cfg.repetition_penalty,
                "length_penalty": self.cfg.length_penalty,
                "stopping_criteria": self.stopping_criteria_list,
                "eos_token_id": self.tokenizer.eos_token_id,  # EOS token
                "pad_token_id": self.tokenizer.pad_token_id,  # Padding token (important for batching)
            }
            # Generate outputs
            generated_outputs = self.model.generate(
                input_ids=model_inputs['input_ids'], 
                **generation_args
            )
            # Decode only the new tokens (exclude the input prompt)
            generated_texts = self.tokenizer.batch_decode(
                generated_outputs[:, input_ids_length:], 
                skip_special_tokens=True
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
        perplexities = []
    
        for input_text, output_text in zip(input_texts, output_texts):
            
            input_ids = self.tokenizer.encode(
                input_text, return_tensors='pt'
            ).to(self.device)
            
            answer_ids = self.tokenizer.encode(
                output_text, return_tensors='pt', add_special_tokens=False
            ).to(self.device)
            
            log_prob_sum = 0.0
            
            total_ids = torch.cat([input_ids, answer_ids], dim=1)
            logits = self.model(total_ids).logits.squeeze(0)
            
            answer_logits = logits[input_ids.shape[1] - 1:-1, :]
            answer_labels = answer_ids.squeeze(0)[:]
            
            perplexity = torch.exp(
                F.cross_entropy(answer_logits, answer_labels) # , reduction='none'
            )
            perplexities.append(perplexity.item())  # Directly store the float
            
        return perplexities  # This will return a list of floats
    

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    # tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3.1-8B-Instruct')
    model = LlamaGenerator(
        LlamaGeneratorConfig(
            model_name='meta-llama/Meta-Llama-3.1-8B-Instruct',
            max_total_tokens=128,
            max_new_tokens=16,
            min_new_tokens=1,
            use_vllm=False
        )
    )
    
    inputs = [
        'Transformer-based GPTs have become extremely popular in 2024. \
            What is the most popular deep learning architecture in 2024?',
        'Diffusion-based DALL-E have become extremely popular in 2024. \
            What is the most popular deep learning architecture in 2024?',
        'Does Roh Tae-woo died earlier then Jun Duhwan?',
        'President often live long\nDoes Roh Tae-woo died earlier then Jun Duhwan?',
        'Roh Tae-woo died in 21/10/26\nDoes Roh Tae-woo died earlier then Jun Duhwan?',
        'Jun Duhwan died in 21/11/23 \nDoes Roh Tae-woo died earlier then Jun Duhwan?',
        'Roh Tae-woo died in 21/10/26\nJun Duhwan died in 21/11/23\nDoes Roh Tae-woo died earlier then Jun Duhwan?'
    ]
    
    sample_texts = [
        model.tokenizer.apply_chat_template([{'role': 'user', 'content': i}], tokenize=False, add_generation_prompt=True).replace('<|begin_of_text|>', '')
        for i in inputs
    ]
    
    sample_forced_outputs = [
        'So the answer is: Transformer',
        'So the answer is: Transformer',
        'Yes',
        'Yes',
        'Yes',
        'Yes',
        'Yes',
    ]
    
    scores = model.score(
        input_texts=sample_texts,
        output_texts=sample_forced_outputs
    )
    
    print("\nPEREPLEXITY RESULTS\n")

    for example in zip(sample_texts, sample_forced_outputs, scores):
        print(example)
