import os
import glob
import torch
import random
import json
import csv
import numpy as np
import numpy.random
import logging
from collections import defaultdict, Counter
import torch.distributed as dist
from typing import Optional


def format_drop_answer(answer_json):
    if answer_json["number"]:
        return answer_json["number"]
    if len(answer_json["spans"]):
        return answer_json["spans"]
    # only date possible
    date_json = answer_json["date"]
    if not (date_json["day"] or date_json["month"] or date_json["year"]):
        print("Number, Span or Date not set in {}".format(answer_json))
        return None
    return date_json["day"] + "-" + date_json["month"] + "-" + date_json["year"]

def load_data_from_jsonl(
    file_path: str, 
    ground_truth_file_path: str = None,
    return_contexts: bool = False,
    is_demo: bool = False
):
    # Open the .jsonl file and read line by line
    inputs = {}
    ground_truth = {}
    contexts = {}
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            if is_demo and i >= 1024:
                break
                
            input_instance = json.loads(line)
            qid = input_instance["question_id"]
            query = question = input_instance["question_text"]
            answers_objects = input_instance["answers_objects"]
            

            formatted_answers = [  # List of potentially validated answers. Usually it's a list of one item.
                tuple(format_drop_answer(answers_object)) for answers_object in answers_objects
            ]
            answer = Counter(formatted_answers).most_common()[0][0]

            output_instance = {
                "qid": qid,
                "query": query,
                "answer": answer,
                "question": question,
            }
            inputs[qid] = question
            ground_truth[qid] = answer
            if return_contexts:
                contexts_ = input_instance["contexts"] # List of Dicts "idx", "title", "paragraph_text"
                contexts[qid] = contexts_
            
    if ground_truth_file_path:
        with open(ground_truth_file_path, 'w') as f:
            json.dump(ground_truth, f, indent=4)
    
    if return_contexts:
        return inputs, ground_truth, contexts
    else:
        return inputs, ground_truth
        
# def load_data_from_jsonl(
#     file_path: str, 
#     ground_truth_file_path: str = None,
#     is_demo: bool = False
# ):
#     # Open the .jsonl file and read line by line
#     inputs = {}
#     ground_truth = {}
#     with open(file_path, 'r') as file:
#         for i, line in enumerate(file):
#             if is_demo and i >= 2048:
#                 break
                
#             input_instance = json.loads(line)
#             qid = input_instance["question_id"]
#             query = question = input_instance["question_text"]
#             answers_objects = input_instance["answers_objects"]

#             formatted_answers = [  # List of potentially validated answers. Usually it's a list of one item.
#                 tuple(format_drop_answer(answers_object)) for answers_object in answers_objects
#             ]
#             answer = Counter(formatted_answers).most_common()[0][0]

#             output_instance = {
#                 "qid": qid,
#                 "query": query,
#                 "answer": answer,
#                 "question": question,
#             }
#             inputs[qid] = question
#             ground_truth[qid] = answer
            
#     if ground_truth_file_path:
#         with open(ground_truth_file_path, 'w') as f:
#             json.dump(ground_truth, f, indent=4)
            
#     return inputs, ground_truth

# Use for passage retrieval
def load_passages(
    path: str,
    shard_id: Optional[int] = None,
    num_shards: Optional[int] = None,
):
    assert shard_id < num_shards, (
        "Ensure that shard_id is less than num_shards. Note that shard_id always starts from 0."
    )
    assert os.path.exists(path), (
        "Path does't exist."
    )
    
    passages = []
    with open(path) as fin:
        if path.endswith(".jsonl"):
            for k, line in enumerate(fin):
                ex = json.loads(line)
                passages.append(ex)
        else:
            reader = csv.reader(fin, delimiter="\t")
            for k, row in enumerate(reader):
                if not row[0] == "id":
                    ex = {"id": row[0], "title": row[2], "text": row[1]}
                    passages.append(ex)
                    
    shard_size = len(passages) // num_shards
    start_idx = shard_id * shard_size
    end_idx = start_idx + shard_size
    if shard_id == num_shards - 1:
        end_idx = len(passages)
    passages = passages[start_idx:end_idx]
    
    return passages


def clean_and_create_dir(dir_path):
    # 디렉토리가 이미 존재하는 경우 안의 파일들 삭제
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    os.rmdir(file_path)  # 만약 하위 디렉토리가 있을 경우 비워야 함
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    # 디렉토리 생성 (존재하면 아무 일도 하지 않음)
    os.makedirs(dir_path, exist_ok=True)