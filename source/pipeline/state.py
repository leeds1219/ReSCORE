import uuid
import json
from collections import deque
from typing import List, Optional, Dict
from dataclasses import dataclass, field
from source.module.index.docstore import Document

from abc import ABC, abstractmethod

@dataclass
class BaseState:
    state_id: str = field(init=False)
    parent_state_id: Optional[str] = None

    def __post_init__(self):
        self.state_id = str(uuid.uuid4())
    
    @abstractmethod
    def to_logging_format(self):
        raise NotImplementedError(
            "Not implemented yet."
        )
    
        
@dataclass
class QuestionState(BaseState):
    question_id: Optional[str] = None
    question: Optional[str] = None
    answer: Optional[str] = None
    
    def __call__(self):
        return self.question
    
    def to_logging_format(self):
        return (
            f"({self.question_id}) Q: {self.question}"
        )
        
@dataclass
class ThoughtState(BaseState):
    thought: Optional[str] = None
    
    def __call__(self):
        return self.thought
    
    def to_logging_format(self):
        return (
            f"Thought: {self.thought}"
        )
        
@dataclass
class AnswerState(BaseState):
    question_id: Optional[str] = None
    answer: Optional[str] = None
    
    def __call__(self):
        return self.answer
    
    def to_logging_format(self):
        return (
            f"Ans: {self.answer}"
        )
        
@dataclass
class EndState(BaseState):
    answer: Optional[str] = None
    # temp
    prediction: Optional[str] = None
    
    def __call__(self):
        return self.answer
    
    def to_logging_format(self):
        return (
            f"End: {self.answer}"
        )


@dataclass
class DocumentState(BaseState):
    documents: Optional[List[Document]] = None
    question_id: Optional[str] = None 
    question: Optional[str] = None
    
    def __call__(self):
        return self.documents
        
    def to_logging_format(self):
        document_title_arr = []
        for document in self.documents:
            document_title_arr.append(document.metadata['title'])
            
        return (
            f"Doc: {document_title_arr}"
        )
        
@dataclass
class ResumeState(BaseState):
    
    def __call__(self):
        return None
    
    def to_logging_format(self):
        return (
            f" ->: KEEP"
        )

@dataclass
class GenerateState(BaseState):
    generated_text: Optional[str] = None
    
    def __call__(self):
        return self.generated_text
    
    def to_logging_format(self):
        return (
            f"Gen: {self.generated_text}"
        )
