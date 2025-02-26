import os
import argparse
import json
import pickle
import time
import glob
from tqdm import tqdm
import os

import argparse
import pickle

from source.module.index.index import Indexer, IndexerConfig
from source.utility import slurm

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data


def main(args):
    print(
        f"Init Indexer at {args.output_dir}.", flush=True
    )
    cfg = IndexerConfig(
        embedding_sz=args.projection_size,
        n_subquantizers=args.n_subquantizers,
        n_bits=args.n_bits,
        database_path=args.output_dir
    )
    indexer = Indexer(
        cfg=cfg
    )
    
    print(
        f"Indexing passages from {args.output_dir}..."
    )
    pattern = 'embeddings_[0-9][0-9]*'
    matching_files = glob.glob(
        os.path.join(
            args.output_dir, pattern
        )
    )
    pbar = tqdm(
        matching_files, 
        desc="Shard", 
        postfix={"file": None}
    )
    for file_path in pbar:
        pbar.set_postfix(
            file=os.path.basename(file_path)
        )
        try:
            with open(file_path, 'rb') as file:
                embeddings, documents = pickle.load(file)
                indexer.index(
                    documents=documents,
                    embeddings=embeddings
                )
        except Exception as e:
            pbar.write(f"Fail at {file_path}. {e}")
    
    print(
        f"Saving index to {args.output_dir}."
    )
    indexer.save_local(
        override=True
    )
    
    print(
        f"Complete!"
    )
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--projection_size", type=int, default=768,
        help=""
    )
    parser.add_argument(
        "--n_subquantizers", type=int, default=0,
        help="Number of subquantizer used for embedding quantization, if 0 flat index is used"
    )
    parser.add_argument(
        "--n_bits", type=int, default=8, 
        help="Number of bits per subquantizer"
    )
    parser.add_argument(
        "--output_dir", type=str,
        help="dir path to save index"
    )
    args = parser.parse_args()
    slurm.init_distributed_mode(args)
    main(args)
