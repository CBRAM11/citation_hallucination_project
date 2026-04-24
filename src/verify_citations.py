import json
import os
import re
import string
from difflib import SequenceMatcher
 
INPUT_PATH  = "evaluation/extracted_citations.json"
CORPUS_PATH = "data/corpus.jsonl"
OUTPUT_PATH = "evaluation/verified_citations.json"
 
# Thresholds
EXACT_THRESHOLD  = 1.00   # normalised after punctuation removal
PARTIAL_THRESHOLD = 0.60  # Jaccard on title tokens
 
 

 
def normalise(text: str) -> str:
    """Lowercase, remove punctuation, collapse whitespace."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
 
 
def token_jaccard(a: str, b: str) -> float:
    """Jaccard similarity on word sets of two normalised strings."""
    sa = set(a.split())
    sb = set(b.split())
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)
 
 
def sequence_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()
 
 

 
def load_corpus(path: str) -> list[dict]:
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers
 
 
def build_lookup(papers: list[dict]) -> dict[str, dict]:
    """Build a dict: normalised_title → paper for O(1) exact lookup."""
    lookup = {}
    for p in papers:
        key = normalise(p.get("title", ""))
        if key:
            lookup[key] = p
    return lookup
 
 

 
def verify_citation(
    cited_title: str,
    corpus_lookup: dict[str, dict],
    corpus_papers: list[dict],
) -> dict:
    """
    Returns a dict with:
      status        : "valid" | "partial" | "hallucinated"
      matched_title : str | None
      match_score   : float
    """
    norm_cited = normalise(cited_title)
 
    # 1. Exact match
    if norm_cited in corpus_lookup:
        paper = corpus_lookup[norm_cited]
        return {
            "status":        "valid",
            "matched_title": paper["title"],
            "match_score":   1.0,
        }
 
    
    best_score  = 0.0
    best_paper  = None
    for p in corpus_papers:
        norm_corpus = normalise(p.get("title", ""))
        score = token_jaccard(norm_cited, norm_corpus)
        if score > best_score:
            best_score = score
            best_paper = p
 
    if best_score >= PARTIAL_THRESHOLD:
        
        seq = sequence_ratio(norm_cited, normalise(best_paper["title"]))
        if seq >= 0.55:
            return {
                "status":        "partial",
                "matched_title": best_paper["title"],
                "match_score":   round(best_score, 3),
            }
 
    return {
        "status":        "hallucinated",
        "matched_title": None,
        "match_score":   round(best_score, 3),
    }
 
 

 
def main():
    os.makedirs("evaluation", exist_ok=True)
 
    print(f"Loading corpus from {CORPUS_PATH} ...")
    corpus_papers = load_corpus(CORPUS_PATH)
    corpus_lookup = build_lookup(corpus_papers)
    print(f"  {len(corpus_papers):,} papers loaded, {len(corpus_lookup):,} unique titles")
 
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        extracted = json.load(f)
 
    verified = {}
 
    for system, rows in extracted.items():
        print(f"\nVerifying: {system}")
        system_rows = []
 
        for row in rows:
            checks = []
            retrieved_norm = {normalise(t) for t in row.get("retrieved_titles", [])}
 
            for cited_title in row.get("citation_candidates", []):
                result = verify_citation(cited_title, corpus_lookup, corpus_papers)
 
                
                grounded = (normalise(cited_title) in retrieved_norm)
 
                checks.append({
                    "cited_title":    cited_title,
                    "status":         result["status"],
                    "matched_title":  result["matched_title"],
                    "match_score":    result["match_score"],
                    "grounded":       grounded,   
                })
 
            system_rows.append({
                "id":          row["id"],
                "question":    row["question"],
                "domain":      row.get("domain", ""),
                "answer":      row.get("answer", ""),
                "parse_error": row.get("parse_error", False),
                "checks":      checks,
            })
 
        # Quick summary
        total = sum(len(r["checks"]) for r in system_rows)
        hall  = sum(
            1 for r in system_rows
            for c in r["checks"] if c["status"] == "hallucinated"
        )
        rate  = hall / total if total else 0
        print(f"  {total} citations checked — hallucination rate: {rate:.1%}")
 
        verified[system] = system_rows
 
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(verified, f, indent=2, ensure_ascii=False)
 
    print(f"\nVerified citations saved → {OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    main()