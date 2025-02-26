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
    ResumeState
)
from source.module.index.docstore import (
    Docstore,
    Document
)
from dataclasses import dataclass
import os
import json

from typing import Optional, Literal, Union, List, Dict, Any


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
    
    
class RetrievalStep:
    
    def __init__(
        self,
        cfg,
        retriever,
        indexer,  
    ):
        self.cfg = cfg
        self.retriever = retriever
        self.indexer = indexer
        
    def __call__(
        self,
        paths: List[List[BaseState]]
    ) -> List[BaseState]:
        
        parent_state_ids = []
        queries, document_ids_so_far = [], []
        for path in paths:
            parent_state_ids.append(
                path[-1].state_id
            )

            question_id, question, thoughts, documents = parse_path(path)
            document_ids_so_far.append(
                doc.id for doc in sum(documents, [])
            )
            queries.append(
                preprocess_retrieval_query(
                    question,
                    thoughts,
                    retrieval_query_type=self.cfg.retrieval_query_type
                )
            )
            
        embeddings = self.retriever.embed(
            input_texts=queries,
            input_type='query'
        ).detach().cpu().numpy().astype('float32')
        
        indexer_outputs = self.indexer.search(
            query_embeddings=embeddings,
            k=self.cfg.retrieval_buffer_size
        )
        
        all_next_states = []
        for indexer_output, _document_ids_so_far, parent_state_id in zip(indexer_outputs, document_ids_so_far, parent_state_ids):
            documents = filter_document(
                documents=indexer_output.documents,
                document_ids_so_far=_document_ids_so_far,
                retrieval_no_duplicates=self.cfg.retrieval_no_duplicates
            )

            all_next_states.append(
                DocumentState(
                    parent_state_id=parent_state_id,
                    documents=documents
                )
            )

        return all_next_states
