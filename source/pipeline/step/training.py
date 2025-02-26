import torch
import numpy as np
import random
from tqdm import tqdm
import spacy

from source.utility.data_utils import (
    load_data_from_jsonl
)
from source.utility.system_utils import (
    seed_everything,
)
from source.module.generate.llama import (
    LlamaGenerator,
    LlamaGeneratorConfig
)
from source.module.retrieve.dense import (
    DenseRetriever,
    DenseRetrieverConfig
)
from source.module.index.index import (
    Indexer,
    IndexerConfig
)
from source.pipeline.controller import (
    PipelineController
)
from source.pipeline.state import (
    BaseState,
    QuestionState,
    AnswerState,
    DocumentState,
    ResumeState,
    ThoughtState,
)
from source.module.index.docstore import (
    Docstore,
    Document
)
from dataclasses import dataclass
import os
import json

from typing import Optional, Literal, Union, List, Dict, Any

from abc import ABC, abstractmethod

import wandb

from source.pipeline.utils import (
    parse_path,
    preprocess_documents,
    filter_document,
    preprocess_retrieval_query,
    clean_wrong_json_format
)
from source.pipeline.constants import (
    DOC_DOC_DELIM,
    THOUGHT_THOUGHT_DELIM
)

import copy

import torch.nn.functional as F

import sys

class TrainStep:
    
    def __init__(
        self,
        cfg,
        generator,
        retriever,
        indexer,
    ):
        self.cfg = cfg
        self.generator = generator
        self.retriever = retriever
        self.indexer = indexer
        
        with open(self.cfg.qa_gen_input_prompt_file_path, 'r', encoding='utf-8') as file:
            self.prompt_template = file.read()
        with open(self.cfg.qa_gen_output_prompt_file_path, 'r', encoding='utf-8') as file:
            self.answer_template = file.read()
            
    def formatting(
        self,
        path,
    ):
        question_id, question, thoughts, documents = parse_path(path)
        
        documents = documents[-1]
        query = preprocess_retrieval_query(
            question,
            thoughts,
            retrieval_query_type=self.cfg.retrieval_query_type
        )
        thoughts_str = THOUGHT_THOUGHT_DELIM.join(thoughts) if thoughts else 'None'
        prompts, answers = [], []
        for d in documents:
            d = str(d)
            prompt = self.prompt_template.format(
                question=question,
                thoughts=thoughts_str,
                documents=d
            )
            prompts.append(prompt)
            answers.append(self.answer_template.format(
                question=question,
                answer=path[0].answer,
                )
            )
        
        return query, prompts, answers, documents

    def __call__(
        self,
        paths: List[List[BaseState]]
    ) -> List[BaseState]:
        
        all_next_states = [copy.deepcopy(path[-1]) for path in paths]
        
        B, N = len(paths), self.cfg.retrieval_buffer_size
        
        queries, prompts, answers, documents = [], [], [], []
        for path in paths:
            query, _prompts, _answers, _documents = self.formatting(path)
            queries.append(query)
            prompts.extend(_prompts)
            answers.extend(_answers)
            documents.extend(_documents)

        query_embeddings = self.retriever.embed(
            input_texts=queries,
            input_type='query'
        ).view(B, 1, -1).expand(-1, N, -1).contiguous().view(B * N, -1)

        document_embeddings = torch.stack([
            self.indexer.get_embedding_from_docstore_id(doc.id, return_tensors='pt').to(query_embeddings.device)
            for doc in documents
        ], dim=0)
        
        r_score = torch.sum(
            query_embeddings * document_embeddings, 
            dim=-1
        ).view(B, N, 1)

        lm_score = torch.tensor(
            self.generator.score(prompts, answers),
            device=query_embeddings.device
        ).view(B, N, 1)

        retriever_likelihood = F.log_softmax(
            r_score / self.cfg.temperature_r, 
            dim=1
        )

        lm_likelihood = F.softmax(
            -lm_score / self.cfg.temperature_lm, 
            dim=1
        )

        loss = F.kl_div(
            retriever_likelihood, lm_likelihood, 
            reduction='batchmean'
        ).div_(self.cfg.gradient_accumulation_steps)     

        return all_next_states, loss
