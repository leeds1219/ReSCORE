from typing import List, Any, Optional, Literal
from source.pipeline.state import (
    BaseState,
    QuestionState,
    AnswerState,
    DocumentState,
    ResumeState,
    EndState
)
from source.module.index.docstore import Document
from source.pipeline.constants import (
    DOC_DOC_DELIM,
    THOUGHT_THOUGHT_DELIM,
    QUESTION_THOUGHT_DELIM
)

def parse_path(
    path: List[BaseState]
) -> Any:
    question_id = None
    question = None
    thoughts = []
    documents = []
    
    for state in path:
        if type(state).__name__ == 'QuestionState':
            question_id = state.question_id
            question = state.question
            
        elif type(state).__name__ == 'ThoughtState':
            thoughts.append(state.thought)
            
        elif (type(state).__name__ == 'DocumentState'):
            documents.append(state.documents)

    return question_id, question, thoughts, documents


def preprocess_documents(
    documents: List[List[Document]],
    prompt_document_from,
    retrieval_count,
    prompt_max_para_count,
):          
    if prompt_document_from == 'last_only':
        documents = documents[-1][:retrieval_count] if documents else []

    else:
        documents = []
        document_ids = set()
        full_flag = False
        
        for _documents in documents:
            if full_flag:
                break
            for _document in _documents[:retrieval_count]:
                if full_flag:
                    break
                if _document.id not in document_ids:
                    document_ids.add(_document.id)
                    documents.append(_document)
                if len(documents) > prompt_max_para_count:
                    full_flag=True
                    break
                
    documents = documents[:prompt_max_para_count]
    
    return documents


def filter_document(
    documents: List[Document],
    document_ids_so_far: set,
    retrieval_no_duplicates: Optional[bool] = False
) -> List[Document]:
    
    filtered_documents = []
    for document in documents:
        if retrieval_no_duplicates and document.id in document_ids_so_far:
            print(
                f"{document.metadata['title']} is in retrieved_so_far_. Skip."
            )
            continue
        
        filtered_documents.append(document)
    
    return filtered_documents


def preprocess_retrieval_query(
    question: str,
    thoughts: List[str],
    retrieval_query_type: Optional[
        Literal[
            'last_only',
            'full'
        ]
    ] = 'full'
):
    query_arr = []
    
    if retrieval_query_type == 'last_only':
        if thoughts:
            query_arr.append(thoughts[-1])
        else:
            query_arr.append(question)
        
    elif retrieval_query_type == 'full':
        query_arr.append(question)
        query_arr.append(THOUGHT_THOUGHT_DELIM.join(thoughts))
        
    query = QUESTION_THOUGHT_DELIM.join(query_arr)
    
    return query


def clean_wrong_json_format(
    text,
    json_key
) -> str:
    text = text.replace('{', '')
    text = text.replace('}', '')
    text = text.replace(json_key, '')
    text = text.replace(':', '')
    text = text.strip()
    
    return str(text)
