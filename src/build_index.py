import json
import os
import pickle
 
import faiss
from sentence_transformers import SentenceTransformer
 
CORPUS_PATH = "data/corpus.jsonl"
INDEX_PATH  = "retrieval/faiss.index"
META_PATH   = "retrieval/corpus_meta.pkl"
MODEL_NAME  = "all-MiniLM-L6-v2"   
 
 
def load_corpus(path):
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers
 
 
def build_doc_text(paper):
    """
    Build the text that will be embedded for a paper.
 
    Key change from the original: abstract is the PRIMARY field.
    Title alone is too short and produces poor retrieval. We truncate
    the abstract at 400 chars so the embedding model is not swamped
    (all-MiniLM-L6-v2 has a 256-token limit; ~400 chars ≈ 90 tokens).
    """
    title    = paper.get("title", "").strip()
    abstract = (paper.get("abstract", "") or "").strip()
    mesh     = paper.get("mesh_terms", [])
 
    # Truncate abstract to keep within token budget
    if len(abstract) > 400:
        abstract = abstract[:400].rsplit(" ", 1)[0] + "..."
 
    mesh_str = ", ".join(mesh) if isinstance(mesh, list) else ""
 
    
    parts = [f"Title: {title}"]
    if abstract:
        parts.append(f"Abstract: {abstract}")
    if mesh_str:
        parts.append(f"MeSH: {mesh_str}")
 
    return " | ".join(parts)
 
 
def main():
    os.makedirs("retrieval", exist_ok=True)
 
    papers = load_corpus(CORPUS_PATH)
    print(f"Loaded {len(papers):,} papers from {CORPUS_PATH}")
 
    if not papers:
        raise ValueError(
            "Corpus is empty. Run build_corpus.py first to produce data/corpus.jsonl"
        )
 
    
    empty = sum(1 for p in papers if not (p.get("abstract") or "").strip())
    if empty:
        print(f"[WARN] {empty} papers still have empty abstracts — "
              "consider re-running build_corpus.py with stricter filtering.")
 
    print(f"Building embeddings with '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)
    texts = [build_doc_text(p) for p in papers]
 
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True,
        batch_size=64,
    ).astype("float32")
 
    
    faiss.normalize_L2(embeddings)
 
    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
 
    faiss.write_index(index, INDEX_PATH)
    print(f"FAISS index saved  → {INDEX_PATH}  ({index.ntotal} vectors, dim={dim})")
 
    with open(META_PATH, "wb") as f:
        pickle.dump(papers, f)
    print(f"Metadata saved     → {META_PATH}")
 
    print("\nBuild complete. Run generate.py next.")
 
 
if __name__ == "__main__":
    main()
 