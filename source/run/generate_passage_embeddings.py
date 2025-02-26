import os

import argparse
import pickle
import pandas as pd
import torch
from tqdm import tqdm

from source.utility import (
    text_utils,
    data_utils,
    slurm
)
from source.module.index.docstore import Document
from source.module.retrieve.dense import DenseRetriever, DenseRetrieverConfig


def preprocess_passage_to_doc(passage):
    if args.no_title or not "title" in passage:
        text = passage["text"]
    else:
        text = passage["title"] + " " + passage["text"]
    if args.lowercase:
        text = text.lower()
    if args.normalize_text:
        text = text_utils.normalize_text(text)
        
    doc = Document(
        id=passage['id'],
        content=text,
        metadata=passage
    )
    
    return doc


def main(args):
    
    print(
        f"Load Model, Tokenizer from {args.model_name_or_path}.", flush=True
    )
    
    cfg = DenseRetrieverConfig(
        batch_size=args.per_gpu_batch_size,
        training_strategy='query_only',
        use_fp16=True,
        query_model_name_or_path='facebook/contriever-msmarco',
    )
    
    retriever = DenseRetriever(
        cfg=cfg
    )
        
        
    print(
        f"Load passages from {args.passages}.", flush=True
    )
    passages = data_utils.load_passages(
        path=args.passages,
        shard_id=args.shard_id,
        num_shards=args.num_shards
    )


    print(
        f"Embedding generation for {len(passages)} passages."
    )
    documents = [preprocess_passage_to_doc(p) for p in passages]
    passages = [d.content for d in documents]
    
    embeddings = retriever.embed(
        input_texts=passages,
        input_type='passsage'
    )
    embeddings = embeddings.detach().cpu().numpy()
    
    
    print(
        f"Saving {len(documents)} passage embeddings to {args.output_dir}."
    )
    save_file = os.path.join(
        args.output_dir, 'embeddings' + f"_{args.shard_id:02d}"
    )
    os.makedirs(
        args.output_dir, exist_ok=True
    )
    with open(save_file, mode="wb") as f:
        pickle.dump((embeddings, documents), f)


    print(
        f"Complete! Saved at {save_file}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--model_name_or_path", type=str,
        help="path to directory containing model weights and config file",
    )
    parser.add_argument(
        "--passages", type=str, default=None, 
        help="Path to passages (.tsv file)"
    )
    parser.add_argument(
        "--output_dir", type=str,
        help="dir path to save embeddings"
    )
    parser.add_argument(
        "--shard_id", type=int, default=0,
        help="Id of the current shard",
    )
    parser.add_argument(
        "--num_shards", type=int, default=1,
        help="Total number of shards",
    )
    parser.add_argument(
        "--per_gpu_batch_size", type=int, default=512,
        help="Batch size for the passage encoder forward pass",
    )
    parser.add_argument(
        "--passage_maxlength", type=int, default=512,
        help="Maximum number of tokens in a passage",
    )
    parser.add_argument(
        "--no_fp16", action="store_true",
        help="inference in fp32",
    )
    parser.add_argument(
        "--no_title", action="store_true",
        help="title not added to the passage body",
    )
    parser.add_argument(
        "--lowercase", action="store_true", 
        help="lowercase text before encoding"
    )
    parser.add_argument(
        "--normalize_text", action="store_true", 
        help="lowercase text before encoding"
    )
    
    args = parser.parse_args()

    slurm.init_distributed_mode(args)

    main(args)