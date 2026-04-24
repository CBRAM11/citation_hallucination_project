import pickle
 
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
 
INDEX_PATH = "retrieval/faiss.index"
META_PATH  = "retrieval/corpus_meta.pkl"
MODEL_NAME = "all-MiniLM-L6-v2"
 
 
class Retriever:
    def __init__(self):
        print(f"[Retriever] Loading index from {INDEX_PATH} ...")
        self.index = faiss.read_index(INDEX_PATH)
 
        with open(META_PATH, "rb") as f:
            self.papers = pickle.load(f)
 
        self.model = SentenceTransformer(MODEL_NAME)
        print(f"[Retriever] Ready — {self.index.ntotal:,} papers indexed.")
 
    def search(self, query, top_k=5):
        """
        Return the top_k most relevant papers for the query.
 
        Each returned dict is the original paper metadata from corpus.jsonl
        plus a 'retrieval_score' field (cosine similarity, 0-1).
        """
        # Embed and normalise the query
        vec = self.model.encode([query], convert_to_numpy=True).astype("float32")
        faiss.normalize_L2(vec)
 
        scores, indices = self.index.search(vec, top_k)
 
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:          
                continue
            paper = dict(self.papers[idx])   
            paper["retrieval_score"] = float(score)
            results.append(paper)
 
        return results