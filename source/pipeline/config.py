import argparse
from pathlib import Path
import torch
import numpy as np
import random
from tqdm import tqdm

from source.utility.system_utils import (
    seed_everything
)

from source.module.generate.t5 import (
    T5Generator,
    T5GeneratorConfig
)
from source.module.retrieve.dense import (
    DenseRetriever,
    DenseRetrieverConfig
)
from source.module.index.index import (
    Indexer,
    IndexerConfig
)

from dataclasses import dataclass
import os
import json
import pickle
import wandb

from typing import Optional, Literal, Union, List, Dict, Any

def clean_arr(arr):
    return [part for part in arr if part]


@dataclass
class PipelineConfig:
    running_name: Optional[str] = None
    batch_size: Optional[int] = 2
    seed: Optional[int] = 100
    dataset: Optional[Literal['hotpotqa', '2wikimultihopqa', 'musique']] = 'musique'
    dataset_split: Optional[Literal['train', 'dev', 'test']] = 'dev'
    pipeline_type: Optional[Literal['single_retrieval', 'multi_retrieval', 'no_retrieval']] = 'multi_retrieval'

    # Prompt
    prompt_set: Optional[int] = 1
    prompt_document_from: Optional[
        Literal[
            'last_only',
            'full'
        ]
    ] = 'last_only'
    prompt_max_para_count: Optional[int] = 15
    prompt_max_para_words: Optional[int] = 350
    
    # Generator
    generation_model_name: Optional[str] = 'google/flan-t5-xl'
    generation_max_batch_size: Optional[int] = 1
    generation_max_total_tokens: Optional[int] = 4096
    generation_max_new_tokens: Optional[int] = 64
    generation_min_new_tokens: Optional[int] = 1
    
    # Retrieval
    retrieval_query_type: Optional[
        Literal[
            'last_only',
            'full'
        ]
    ] = 'full'
    retrieval_count: Optional[Literal[2, 4, 6, 8]] = 8
    retrieval_buffer_size: Optional[int] = 100
    retrieval_no_duplicates: Optional[bool] = True
    retrieval_no_reasoning_sentences: Optional[bool] = True # Store true
    retrieval_no_wh_words: Optional[bool] = True # Store true
    
    # Retriever
    retrieval_query_model_name_or_path: Optional[str] = 'facebook/contriever-msmarco'
    retrieval_passage_model_name_or_path: Optional[str] = None
    retrieval_batch_size: Optional[int] = 32
    retrieval_training_strategy: Optional[Literal['query_only', 'both']] = 'query_only'
    retrieval_use_fp16: Optional[int] = True
    
    # End
    max_num_thought: int = 6
    min_num_thought: int = 1
    answer_regex: str = ".* answer is:? (.*)\\.?" # answer_regex: str = ".* Answer: <.*>\\.?"
    match_all_on_failure: bool = True

    # Etc
    method: Optional[str] = "base"
    demo: Optional[bool] = False
    
    # Train
    train: bool = False
    training_score_method: Optional[Literal['qa_gen', 'ans_gen']] = "qa_gen"
    n_epochs: int = 1
    lr: float = 1e-6
    temperature_r: float = 0.1
    temperature_lm: float = 1.
    gradient_accumulation_steps: int = 1
    wandb_key: Optional[str] = None

    def __post_init__(self):
        seed_everything(self.seed)
        
        if self.method == "rescore":
            self.min_num_thought = 1
        else:
            self.min_num_thought = 0    
            
        if self.wandb_key:
            wandb.login(
                key=self.wandb_key
            )
            wandb.init(
                project='your_project',
                name=self.running_name,
                config=self.__dict__
            )    

    def save(self):
        with open(self.configuration_file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def database_path(self):
        
        if self.retrieval_passage_model_name_or_path:
            index_folder_name = self.retrieval_passage_model_name_or_path
        else:
            index_folder_name = self.retrieval_query_model_name_or_path
        index_folder_name = index_folder_name.split('/')[-1].replace('-', '_').strip()
        
        return os.path.join(
            './', "data", "database", index_folder_name, self.dataset,
        )

    @property
    def prediction_file_dir(self):
        
        prediction_file_directory_arr = [
            './', 
            'predictions', 
            f"{self.dataset}",
            '___'.join(clean_arr([
                self.running_name,
                self.generation_model_name.split('/')[-1].replace('-', '_').lower(),
                self.retrieval_query_model_name_or_path.split('/')[-1].replace('-', '_').lower(),
            ])),
            '___'.join(clean_arr([
                self.pipeline_type,
                'train' if self.train else 'inference',
            ])),
            f"prompt_set__{self.prompt_set}",
        ]
        if self.dataset_split == 'test':
            prediction_file_directory_arr.append(
                'best'
            )
        else:
            prediction_file_directory_arr.append(
                f"retr_count__{self.retrieval_count}"
            )
            
        prediction_file_dir = os.path.join(
            *prediction_file_directory_arr
        )
        return prediction_file_dir
    
    @property
    def configuration_file_path(self):
        return os.path.join(
            self.prediction_file_dir, 'configuration.json'
        )
    
    @property
    def data_file_path(self):
        if self.dataset_split == 'train':
            return os.path.join(
                './', 'data', 'processed_data', self.dataset, f'{self.dataset_split}.jsonl'
            )
        else:
            return os.path.join(
                './', 'data', 'processed_data', self.dataset, f'{self.dataset_split}_subsampled.jsonl'
            )

    @property
    def id_to_log_file_path(self):
        return os.path.join(
            self.prediction_file_dir, f'{self.dataset_split}_id_to_log.jsonl'
        )

    @property
    def logging_file_path(self):
        return os.path.join(
            self.prediction_file_dir, f'{self.dataset_split}_logging.jsonl'
        )
        
    @property
    def id_to_ground_truths_file_path(self):
        return os.path.join(
            self.prediction_file_dir, f'{self.dataset_split}_id_to_ground_truths.json'
        )

    @property
    def ground_truth_file_path(self):
        return os.path.join(
            self.prediction_file_dir, f'{self.dataset_split}_ground_truth.json'
        )

    @property
    def id_to_predictions_file_path(self):
        return os.path.join(
            self.prediction_file_dir, f'{self.dataset_split}_id_to_predictions.json'
        )

    @property
    def prediction_file_path(self):
        return os.path.join(
            self.prediction_file_dir, f'{self.dataset_split}_prediction.json'
        )

    @property
    def id_to_evaluation_file_path(self):
        return os.path.join(
            self.prediction_file_dir, f'{self.dataset_split}_id_to_evaluation.json'
        )

    @property
    def evaluation_file_path(self):
        return os.path.join(
            self.prediction_file_dir, f'{self.dataset_split}_evaluation.json'
        )
        
    @property
    def official_evaluation_file_path(self):
        return os.path.join(
            self.prediction_file_dir, f'{self.dataset_split}_official_evaluation.json'
        )

    @property
    def qa_gen_input_prompt_file_path(self):
        return os.path.join(
            './', 'prompts', f"prompt_set__{self.prompt_set}", "qa_gen_input.txt"
        )   

    @property
    def qa_gen_output_prompt_file_path(self):
        return os.path.join(
            './', 'prompts', f"prompt_set__{self.prompt_set}", "qa_gen_output.txt"
        ) 
    
    @property
    def multi_retr_answer_direct_gen_prompt_file_path(self):
        return os.path.join(
            './', 'prompts', f"prompt_set__{self.prompt_set}", "multi_retr_answer_direct_gen.txt"
        ) 
        
    @property
    def multi_retr_thought_direct_gen_prompt_file_path(self):
        return os.path.join(
            './', 'prompts', f"prompt_set__{self.prompt_set}", "multi_retr_thought_direct_gen.txt"
        )

    @property
    def answer_gen_prompt_file_path(self):
        return os.path.join(
            './', 'prompts', f"prompt_set__{self.prompt_set}", "answer_gen.txt"
        )
        
    @property
    def thought_gen_prompt_file_path(self):
        return os.path.join(
            './', 'prompts', f"prompt_set__{self.prompt_set}", "thought_gen.txt"
        )

    def save(self):
        with open(self.configuration_file_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
