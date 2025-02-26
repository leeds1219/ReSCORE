import os
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from numpy.typing import ArrayLike
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Union

from .docstore import Docstore, Document
import torch

@dataclass
class IndexerConfig:
    embedding_sz: int = 768
    n_subquantizers: int = 0
    n_bits: int = 8
    database_path: str = "./database"
    index_file_name: str = "index.faiss"


@dataclass
class IndexerOutput:
    documents: List[Document]
    scores: List[int]


class Indexer(object):

    def __init__(
        self, 
        cfg: IndexerConfig = IndexerConfig(),
        faiss_index: Optional[faiss.Index] = None,
        docstore: Optional[Docstore] = None,
        faiss_id_to_docstore_id: Optional[List[str]] = None,
    ):
        assert (cfg.database_path), (
            f"[{'Indexer':^16}] You must specify the `database_path` directly."
        )
        self.database_path = cfg.database_path
        self.cfg = cfg
        # Init Index
        if not faiss_index:
            if cfg.n_subquantizers > 0:
                faiss_index = faiss.IndexPQ(
                    cfg.embedding_sz, 
                    cfg.n_subquantizers, 
                    cfg.n_bits, 
                    faiss.METRIC_INNER_PRODUCT
                )
            else:
                faiss_index = faiss.IndexFlatIP(
                    cfg.embedding_sz
                )
        self.faiss_index = faiss_index
        # Initialized Docstore
        if not docstore:
            docstore = Docstore(
                database_path=cfg.database_path
            )
        self.docstore = docstore
        # Dict to match Index with Docstore
        if not faiss_id_to_docstore_id:
            faiss_id_to_docstore_id = []
        self.faiss_id_to_docstore_id = faiss_id_to_docstore_id
        self.docstore_id_to_faiss_id = {
            docstore_id: faiss_id
            for faiss_id, docstore_id in enumerate(faiss_id_to_docstore_id)
        }
    
    @classmethod
    def load_local(
        cls,
        cfg: IndexerConfig
    ):
        # Paths to load from
        index_path = os.path.join(
            cfg.database_path, cfg.index_file_name
        )
        faiss_id_to_docstore_id_path = os.path.join(
            cfg.database_path, "faiss_id_to_docstore_id.pkl"
        )
        print(
            f"[{'Indexer':^16}] Deserializing Index from {cfg.database_path}...", 
            end=" "
        )
        # Load Index
        faiss_index = faiss.read_index(index_path)
        # Load Docstore
        docstore = Docstore.load(cfg.database_path)
        # Load index_to_docstore_id
        with open(faiss_id_to_docstore_id_path, 'rb') as f:
            faiss_id_to_docstore_id = pickle.load(f)
        # Perform integrity check
        index_length = len(faiss_id_to_docstore_id)
        cursor = docstore.conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM documents"
        )
        docstore_length = cursor.fetchone()[0]
        assert index_length == faiss_index.ntotal, (
            f" Deserialized index_to_docstore_id size {index_length} should match FAISS index size {faiss_index.ntotal}"
        )
        assert docstore_length == faiss_index.ntotal, (
            f" Docstore size {docstore_length} should match FAISS index size {faiss_index.ntotal}"
        )
        # Initialize Object
        obj = cls(
            faiss_index=faiss_index,
            docstore=docstore,
            faiss_id_to_docstore_id=faiss_id_to_docstore_id,
            cfg=cfg
        )
        print(
            f" Deserializing Complete!"
        )
        
        return obj
            
            
    def save_local(
        self,
        override: bool = False
    ) -> None:
        # Paths to save to
        index_path = os.path.join(
            self.database_path, self.cfg.index_file_name
        )
        faiss_id_to_docstore_id_path = os.path.join(
            self.database_path, "faiss_id_to_docstore_id.pkl"
        )
        if os.path.exists(index_path) and override is False:
            raise FileExistsError(
                f"[{'Indexer':^16}] Index file already exists at {index_path}. Set 'override=True' to overwrite."
            )
        
        print(
            f"[{'Indexer':^16}] Serializing Index to {self.database_path}...", 
            end=" "
        )
        # save Index
        faiss.write_index(self.faiss_index, index_path)
        # save index_to_docstore_id
        with open(faiss_id_to_docstore_id_path, 'wb') as f:
            pickle.dump(self.faiss_id_to_docstore_id, f)
            
        print(
            f"Serializing Complete!"
        )
        
    def add(
        self,
        documents: Union[Document, List[Document]],
        embeddings: ArrayLike
    ):
        if type(documents) != list:
            documents = [documents]
        
        if len(embeddings.shape) == 1:
            embeddings = embeddings[np.newaxis, :]
            
        assert embeddings.shape[1] == self.cfg.embedding_sz, (
            f"[{Indexer:^16}] Embedding size {embeddings.shape[1]} doesn't match indexer embedding size {self.cfg.embedding_sz}."
        )
        
        embeddings = embeddings.astype('float32')
        
        pbar = tqdm(
            zip(embeddings, documents),
            desc=f"Indexing",
            total=len(documents)
        )
        
        for embedding, document in pbar:
            self.faiss_index.add(embedding[np.newaxis, :])
            self.docstore._add(document)
            self.faiss_id_to_docstore_id.append(document.id)
            
    def get_embedding_from_docstore_id(
        self,
        docstore_id: str,
        return_tensors='pt'
    ):
        faiss_id = self.docstore_id_to_faiss_id[docstore_id]
        
        return self.get_embedding_from_faiss_id(
            faiss_id=int(faiss_id),
            return_tensors=return_tensors
        )
        
    def get_embedding_from_faiss_id(
        self,
        faiss_id: int,
        return_tensors='pt'
    ):
        assert type(faiss_id) == int, (
            f"faiss_id should be type of `int`"
        )
        embedding = self.faiss_index.reconstruct(faiss_id)
        
        if return_tensors == 'np':
            embedding = embedding
        elif return_tensors == 'pt':
            embedding = torch.from_numpy(embedding)

        return embedding

    def index(
        self, 
        documents: List[Document], 
        embeddings: ArrayLike
    ):
        if not self.faiss_index.is_trained:
            self.faiss_index.train(embeddings)
            
        self.add(documents, embeddings)
            
        print(
            f"[{'Indexer':^16}] Build Complete! Total {len(self.faiss_id_to_docstore_id)}"
        )

    def search(
        self, 
        query_embeddings: np.array, 
        k: int = 8, 
        index_batch_size: int = 2048,
        return_embeddings: bool = False
    ) -> List[IndexerOutput]:
        
        assert query_embeddings.dtype == np.float32, (
            f"[{'Indexer':^16}] Vectors must be of type float32"
        )

        outputs = []
        i, n_total = 0, len(query_embeddings)
        while i < n_total:
            batched_query_embeddings = query_embeddings[i:i+index_batch_size]
            i += index_batch_size
            
            batched_scores, batched_faiss_ids = self.faiss_index.search(
                batched_query_embeddings, k
            )
            for scores, faiss_ids in zip(batched_scores, batched_faiss_ids):
                docstore_ids = [
                    self.faiss_id_to_docstore_id[index_id] 
                    for index_id in faiss_ids
                ]
                documents = [
                    self.docstore.search(docstore_id)
                    for docstore_id in docstore_ids
                ]
                _output = IndexerOutput(
                    documents=documents,
                    scores=scores
                )
                if return_embeddings:
                    embeddings = [
                        self.faiss_index.reconstruct(int(faiss_id))
                        for faiss_id in faiss_ids
                    ]
                    _output.embeddings = embeddings

                outputs.append(_output)
            
        return outputs