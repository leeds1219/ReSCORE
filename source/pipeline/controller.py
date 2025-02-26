import torch
import uuid
import json
from collections import deque, defaultdict
from typing import List, Optional, Dict, Callable, Any, Literal
from dataclasses import dataclass, field
from .state import BaseState
import jsonlines
import os 
from tqdm import tqdm
from .state import QuestionState
import time
from source.evaluation.evaluate import (
    evaluate_by_dicts,
    official_evaluate_by_dicts,
)
from source.pipeline.config import PipelineConfig

def logging_to_jsonl(logging_file_path, id, data_to_append):
    logs = {}
    if os.path.exists(logging_file_path):
        with open(logging_file_path, mode='r') as file:
            try:
                logs = json.load(file)
            except json.JSONDecodeError:
                logs = {}

    # Update or add the log entry
    if id in logs:
        logs[id].append(data_to_append)
    else:
        logs[id] = [data_to_append]

    with open(logging_file_path, mode='w') as file:
        json.dump(logs, file, indent=4)


class PipelineController(object):
    
    def __init__(
        self, 
        pipeline: Optional[List] = [], 
        logging_file_path: Optional[str] = '', 
        prediction_file_path: Optional[str] = '', 
    ):
        self.pipeline = pipeline
        self.state_tree = {}
        self.running_state_ids = []
        self.end_state_ids = []
        
        self.logging_file_path = logging_file_path
        self.prediction_file_path = prediction_file_path

    def run(
        self, 
        start_states: List[QuestionState],
        batch_size: int = 1
    ):
        for i in tqdm(range(0, len(start_states), batch_size)):
            batch = start_states[i:i + batch_size]
            self.update(batch)
            while not self.is_running_completed:
                self.run_once()
        
        return self.save_completed_states()
        
    def run_once(
        self,
    ):
        for fn in self.pipeline:
            self.update(
                fn(self.next())
            )

    def train(
        self, 
        start_states: List[QuestionState],
    ):
        self.update(start_states)

        total_loss = torch.tensor(0.0, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        step_count = 0 
        while not self.is_running_completed:
            step_count += 1
            loss = self.train_once()
            total_loss += loss 
        if step_count > 0:
            total_loss = total_loss / float(step_count)
        else:
            print("Warning: No steps were executed, returning zero loss.")
            total_loss = torch.tensor(0.0, device=device)

        return total_loss
        

    def train_once(
        self,
    ):
        for fn in self.pipeline:
            if fn == self.pipeline[1]: 
                cur_paths = self.next()
                next_states, loss = fn(cur_paths)
                self.update_trainstep(next_states) 
            else:    
                cur_paths = self.next()
                next_states = fn(cur_paths)
                self.update(next_states)

        return loss

    @property
    def is_running_completed(
        self,
    ) -> bool:
        return len(self.running_state_ids) == 0
    
    def get_state(
        self,
        state_id: str
    ):
        return self.state_tree[state_id]
    
    def get_completed_states(
        self
    ) -> List[BaseState]:
        return [
            self.get_state(state_id) for state_id in self.end_state_ids
        ]
    
    def save_completed_states(
        self,
    ):
        completed_states = self.get_completed_states()

        question_answer_dict = {}
        for state in completed_states:
            question_id = state.question_id
            question_answer_dict[question_id] = str(state.answer)
        
        if self.prediction_file_path:
            with open(self.prediction_file_path, 'w', encoding='utf-8') as json_file:
                json.dump(
                    question_answer_dict, 
                    json_file, ensure_ascii=False, indent=4
                )
            
        return question_answer_dict

    def get_path(
        self,
        leaf_state_or_id: str,
    ):
        path = list()
        cur_state_id = leaf_state_or_id if isinstance(leaf_state_or_id, str) else leaf_state_or_id.state_id
        
        while cur_state_id != None:
            state = self.state_tree.get(cur_state_id, None)
            assert state != None, \
                f'No State with id {cur_state_id}'
                
            path.append(state)
            cur_state_id = state.parent_state_id
            
        path.reverse()
        
        return path

    def update(
        self,
        states: List[BaseState],
    ):
        assert len(self.running_state_ids) == 0, (
            "Running states should be reset before update"
        )
        
        for i, state in enumerate(states):
            self.state_tree[state.state_id] = state # error
                
            if self.logging_file_path:
                logging_to_jsonl(
                    logging_file_path=self.logging_file_path,
                    id=self.get_path(state.state_id)[0].state_id, 
                    data_to_append=state.to_logging_format()
                )
                
            if (type(state).__name__ == 'EndState'):
                self.end_state_ids.append(state.state_id)
                path = self.get_path(state.state_id)
                for _state in path:
                    if (type(_state).__name__ == 'QuestionState'):
                        state.question_id = _state.question_id
                    
                    if _state.state_id == state.state_id:
                        continue
                    
                    del self.state_tree[_state.state_id]
                
            else:
                self.running_state_ids.append(state.state_id)

    def next(
        self,
    ) -> List[List[BaseState]]:
        
        running_paths = []
        while self.running_state_ids:
            running_paths.append(
                self.get_path(self.running_state_ids.pop())
            )
            
        return running_paths
