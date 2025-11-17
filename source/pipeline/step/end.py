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
    EndState
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

class EndStep:
    
    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg

    def __call__(
        self,
        paths: List[List[BaseState]]
    ) -> List[BaseState]:
        
        parent_state_ids = []
        all_next_state = []

        for path in paths:
            parent_state_ids.append(
                path[-1].state_id
            )

            question_id, question, thoughts, documents = parse_path(path)
            
            # temp
            # question_id, questions_so_far, documents_so_far = parse_path_rewrite(path)
            
            # need to fix, this is slow, due to for loops
            is_end = False
            for state in path:
                if not type(state).__name__ == 'AnswerState':
                    continue
                if state.answer.lower() != 'unknown':
                    all_next_state.append(
                        EndState(
                            parent_state_id=path[-1].state_id,
                            answer=state.answer
                        )
                    )
                    is_end = True
                    break
            if is_end:
                continue        
            
            if len(thoughts) >= self.cfg.max_num_thought:
                all_next_state.append(
                    EndState(
                        parent_state_id=path[-1].state_id,
                        answer=' '.join(thoughts)
                    )
                )

            # temp
            # elif len(questions_so_far) >= self.cfg.max_num_thought:
            #     all_next_state.append(
            #         EndState(
            #             parent_state_id=path[-1].state_id,
            #             prediction=' '.join(questions_so_far) # ''
            #         )
            #     )

            else:
                all_next_state.append(
                    ResumeState(
                        parent_state_id=path[-1].state_id,
                    )
                )
            
        return all_next_state

class CoTAnswerExtractor:
    
    def __init__(
        self,
        cfg,
    ):
        self.cfg = cfg
        self.regex = re.compile(self.cfg.answer_regex)

    def __call__(
        self,
        paths: List[List[BaseState]]
    ) -> List[BaseState]:
        
        all_next_states = []
        
        for path in paths:
            last_state = path[-1]
            last_state_id = last_state.state_id
            m = self.regex.search(last_state.generated_text)
            if m:
                answer = m.group(1)
                answer = answer.strip()
                if answer.startswith('Answer:'):
                    answer = answer.replace('Answer:', '')
                if answer.endswith('.'):
                    answer = answer.replace('.', '')
                if answer.endswith(';'):
                    answer = answer.replace(';', '')
                
            elif self.cfg.match_all_on_failure:
                answer = last_state.generated_text
            
            all_next_states.append(
                EndState(
                    parent_state_id=last_state_id,
                    answer=str(answer)
                )
            )
                
        return all_next_states
