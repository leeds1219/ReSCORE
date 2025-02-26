# This script is based on code from the IRCOT project by Stony Brook NLP.
# Source: https://github.com/StonyBrookNLP/ircot

"""
raw file to .tsv
"""


import json
import argparse
import hashlib
import io
import dill
from tqdm import tqdm
import glob
import bz2
import base58
from bs4 import BeautifulSoup
import os
import random
from typing import Dict, List, Any, Tuple, Literal, Optional, Generator

def hash_object(o: Any) -> str:
    """Returns a character hash code of arbitrary Python objects."""
    m = hashlib.blake2b()
    with io.BytesIO() as buffer:
        dill.dump(o, buffer)
        m.update(buffer.getbuffer())
        return base58.b58encode(m.digest()).decode()


def make_hotpotqa_documents_tsv(
    output_filepath: str,
    metadata: Dict = None
) -> None:
    raw_glob_filepath = os.path.join("data", "raw_data", "hotpotqa", "wikipedia-paragraphs", "*", "wiki_*.bz2")
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata

    with open(output_filepath, "w", encoding="utf-8") as tsv_file:
        # Write headers to the TSV file
        tsv_file.write("id\ttext\ttitle\n")

        for filepath in tqdm(glob.glob(raw_glob_filepath)):
            for datum in bz2.BZ2File(filepath).readlines():
                instance = json.loads(datum.strip())

                id_ = hash_object(json.dumps(instance))[:32]
                title = instance["title"]
                sentences_text = [e.strip() for e in instance["text"]]
                paragraph_text = ' '.join(sentences_text)
                url = instance.get("url", "")
                is_abstract = True
                paragraph_index = 0

                # Write the passage data to the TSV file
                paragraph_text = paragraph_text.replace('\n', ' ').strip()
                title = title.replace('\n', ' ').strip()
                tsv_file.write(f"{id_}\t{paragraph_text}\t{title}\n")
                metadata["idx"] += 1


def make_iirc_documents_tsv(
    output_filepath: str,
    metadata: Dict = None
) -> None:
    raw_filepath = os.path.join("data", "raw_data", "iirc", "context_articles.json")

    metadata = metadata or {"idx": 1}
    assert "idx" in metadata
    random.seed(13370)  # Don't change.
    
    with open(output_filepath, "w", encoding="utf-8") as tsv_file:
        # Write headers to the TSV file
        tsv_file.write("id\ttext\ttitle\n")

        with open(raw_filepath, "r", encoding="utf-8") as file:
            full_data = json.load(file)

            for title, page_html in tqdm(full_data.items()):
                title = title if title else 'None'
                page_soup = BeautifulSoup(page_html, "html.parser")
                paragraph_texts = [
                    text for text in page_soup.text.split("\n") if text.strip() and len(text.strip().split()) > 10
                ]

                # IIRC has a positional bias. 70% of the times, the first
                # is the supporting one, and almost all are in 1st 20.
                # So we scramble them to make it a more challenging retrieval
                # problem.
                paragraph_indices_and_texts = [
                    (paragraph_index, paragraph_text) for paragraph_index, paragraph_text in enumerate(paragraph_texts)
                ]
                random.shuffle(paragraph_indices_and_texts)
                
                for paragraph_index, paragraph_text in paragraph_indices_and_texts:
                    url = ""
                    id_ = hash_object(title + paragraph_text)
                    is_abstract = paragraph_index == 0

                    # Write the passage data to the TSV file
                    paragraph_text = paragraph_text.replace('\n', ' ').strip()
                    title = title.replace('\n', ' ').strip()
                    tsv_file.write(f"{id_}\t{paragraph_text}\t{title}\n")
                    metadata["idx"] += 1
                    

def make_2wikimultihopqa_documents_tsv(
    output_filepath: str,
    metadata: Dict = None
) -> None:
    raw_filepaths = [
        os.path.join("data", "raw_data", "2wikimultihopqa", "train.json"),
        os.path.join("data", "raw_data", "2wikimultihopqa", "dev.json"),
        os.path.join("data", "raw_data", "2wikimultihopqa", "test.json"),
    ]
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata
    used_full_ids = set()
    
    with open(output_filepath, "w", encoding="utf-8") as tsv_file:
        # Write headers to the TSV file
        tsv_file.write("id\ttext\ttitle\n")
        
        for raw_filepath in raw_filepaths:
            with open(raw_filepath, "r", encoding="utf-8") as file:
                full_data = json.load(file)
                for instance in tqdm(full_data):

                    for paragraph in instance["context"]:
                        title = paragraph[0]
                        paragraph_text = ' '.join(paragraph[1])
                        paragraph_index = 0
                        url = ""
                        is_abstract = paragraph_index == 0

                        full_id = hash_object(' '.join([title, paragraph_text]))
                        if full_id in used_full_ids:
                            continue

                        used_full_ids.add(full_id)
                        id_ = full_id[:32]

                        # Write the passage data to the TSV file
                        paragraph_text = paragraph_text.replace('\n', ' ').strip()
                        title = title.replace('\n', ' ').strip()
                        tsv_file.write(f"{id_}\t{paragraph_text}\t{title}\n")
                        metadata["idx"] += 1


def make_musique_documents_tsv(
    output_filepath: str,
    metadata: Dict = None
) -> None:
    raw_filepaths = [
        os.path.join("data", "raw_data", "musique", "musique_ans_v1.0_dev.jsonl"),
        os.path.join("data", "raw_data", "musique", "musique_ans_v1.0_test.jsonl"),
        os.path.join("data", "raw_data", "musique", "musique_ans_v1.0_train.jsonl"),
        os.path.join("data", "raw_data", "musique", "musique_full_v1.0_dev.jsonl"),
        os.path.join("data", "raw_data", "musique", "musique_full_v1.0_test.jsonl"),
        os.path.join("data", "raw_data", "musique", "musique_full_v1.0_train.jsonl"),
    ]
    metadata = metadata or {"idx": 1}
    assert "idx" in metadata
    used_full_ids = set()
    
    with open(output_filepath, "w", encoding="utf-8") as tsv_file:
        # Write headers to the TSV file
        tsv_file.write("id\ttext\ttitle\n")
        
        for raw_filepath in raw_filepaths:
            with open(raw_filepath, "r", encoding="utf-8") as file:
                for line in tqdm(file.readlines()):
                    if not line.strip():
                        continue
                    instance = json.loads(line)

                    for paragraph in instance["paragraphs"]:
                        title = paragraph["title"]
                        paragraph_text = paragraph["paragraph_text"]
                        paragraph_index = 0
                        url = ""
                        is_abstract = paragraph_index == 0

                        full_id = hash_object(' '.join([title, paragraph_text]))
                        if full_id in used_full_ids:
                            continue

                        used_full_ids.add(full_id)
                        id_ = full_id[:32]

                        # Write the passage data to the TSV file
                        paragraph_text = paragraph_text.replace('\n', ' ').strip()
                        title = title.replace('\n', ' ').strip()
                        tsv_file.write(f"{id_}\t{paragraph_text.strip() if paragraph_text else 'None'}\t{title.strip() if title.strip() else 'None'}\n")
                        metadata["idx"] += 1


def main(args):
    output_filepath = os.path.join("data", "embed_ready_data", f"{args.dataset_name}.tsv")
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    if args.dataset_name == "hotpotqa":
        func = make_hotpotqa_documents_tsv
    elif args.dataset_name == "iirc":
        func = make_iirc_documents_tsv
    elif args.dataset_name == "2wikimultihopqa":
        func = make_2wikimultihopqa_documents_tsv
    elif args.dataset_name == "musique":
        func = make_musique_documents_tsv
    else:
        raise Exception(f"Unknown dataset_name {args.dataset_name}")
    
    func(output_filepath)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset_name", type=str, default=None, 
        help="dataset name to preprocess"
    )
    args = parser.parse_args()
    
    main(args)
