import os
import json
import argparse
from tqdm import tqdm
from typing import List, Dict, Any

from collections import defaultdict
from fuzzywuzzy import fuzz
import numpy as np
from experiments.score_with_retrieval.evaluation import calculate_ndcg, calculate_matches


def calculate_recall(data, workers_num, cal_avg: bool=False):
    match_stats = calculate_matches(data, workers_num, cal_avg)
    top_k_hits = match_stats.top_k_hits

    #logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    #logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)

    results = {}
    r20, r100 = [], []
    for k in [2, 4, 6, 8, 16, 32, 64]:
        if k <= len(top_k_hits):
            recall = 100 * top_k_hits[k-1]
            if k == 20:
                r20.append(f"{recall:.1f}")
            if k == 100:
                r100.append(f"{recall:.1f}")
            results[f'Recall@{k}'] = f'{recall:.3f}'
            
    return results


parser = argparse.ArgumentParser(description='Process some datasets.')
parser.add_argument(
    '--dataset', 
    type=str, choices=['2wikimultihopqa', 'musique', 'hotpotqa', 'nq', 'trivia', 'squad'],
    help='Select the dataset to use'
)
parser.add_argument(
    '--method', 
    type=str, # choices=['qa_gen', 'qlm', 'ans_gen', 'retrieval_only'],
    help='Select the dataset to use'
)
parser.add_argument(
    '--folder', 
    type=str, 
    help='Select the folder name to save'
)
args = parser.parse_args()

query_model_name_or_path = 'facebook/contriever-msmarco'
method = args.method
database_folder_name = query_model_name_or_path.split('/')[-1].replace('-', '_')
if os.path.exists(args.method):
    query_model_name_or_path = args.method
    method = query_model_name_or_path.split('/')[-1].split('.')[0]
    print(method)

res_dir = f"./experiments/score_with_retrieval/{args.folder}/{args.dataset}"
results_documents_file_path = os.path.join(res_dir, "result_documents.json")
gold_documents_file_path = os.path.join(res_dir, "gold_documents.json")
result_file_path = os.path.join(res_dir, "result.json")

# Load results_documents
with open(results_documents_file_path, 'r', encoding='utf-8') as f:
    results_documents = json.load(f)

# Load gold_documents
with open(gold_documents_file_path, 'r', encoding='utf-8') as f:
    gold_documents = json.load(f)

# Compute and Save
data = [
    {
        'question': None,
        'ctxs': [
            {
                'text': x['paragraph_text']
            } for x in results_documents[qid]
        ],
        'answers': [x['paragraph_text'] for x in gold_documents[qid]]
    } for qid in results_documents
]

# print(data[0]['ctxs'][0])
# print(data[0]['answers'])
# exit()
if args.dataset in ['hotpotqa', '2wikimultihopqa', 'musique']:
    cal_avg = True
else:
    cal_avg = False
    
    
results = {
    'nDCG': calculate_ndcg(data, 4),
    'Recall': calculate_recall(data, 4),
    'Recall(Avg)': calculate_recall(data, 4, True)
}

with open(result_file_path, 'w') as result_file:
    json.dump(results, result_file, indent=4)