import json
import os
import re
 
RAW_PATHS = [
    "data/corpus_raw.jsonl",  # preferred (after rename)
    "data/corpus.jsonl",      # fallback (if not renamed yet)
]
OUT_PATH = "data/corpus.jsonl"
MIN_ABSTRACT_LEN = 50
 
 
def load_raw(path):
    papers = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers
 
 
def is_valid(paper):
    abstract = paper.get("abstract", "") or ""
    abstract = re.sub(r"<[^>]+>", "", abstract).strip()
    return len(abstract) >= MIN_ABSTRACT_LEN
 
 
def clean_paper(paper):
    abstract = paper.get("abstract", "") or ""
    abstract = re.sub(r"<[^>]+>", "", abstract).strip()
    paper["abstract"] = abstract
 
    if not isinstance(paper.get("authors", []), list):
        paper["authors"] = []
    if not isinstance(paper.get("mesh_terms", []), list):
        paper["mesh_terms"] = []
 
    for key in ["title", "journal", "specialty", "publication_date",
                "doi", "authors_str", "paper_id", "pmid"]:
        if not isinstance(paper.get(key), str):
            paper[key] = str(paper.get(key, ""))
 
    return paper
 
 
def find_raw_path():
    for path in RAW_PATHS:
        if os.path.exists(path):
            print("[INFO] Reading raw corpus from: {}".format(path))
            return path
    raise FileNotFoundError(
        "Could not find corpus file. Make sure data/corpus.jsonl exists."
    )
 
 
def main():
    os.makedirs("data", exist_ok=True)
 
    raw_path = find_raw_path()
 
    # If reading and writing the same file, use a temp file to avoid data loss
    writing_in_place = os.path.abspath(raw_path) == os.path.abspath(OUT_PATH)
    write_path = "data/corpus_clean_temp.jsonl" if writing_in_place else OUT_PATH
 
    papers = load_raw(raw_path)
    print("Loaded  : {:,} papers".format(len(papers)))
 
    valid = [clean_paper(p) for p in papers if is_valid(p)]
    removed = len(papers) - len(valid)
    print("Removed : {:,} papers with missing/short abstracts".format(removed))
    print("Kept    : {:,} papers with usable abstracts".format(len(valid)))
 
    specialty_counts = {}
    for p in valid:
        s = p.get("specialty", "unknown")
        specialty_counts[s] = specialty_counts.get(s, 0) + 1
 
    print("\nPapers per specialty after filtering:")
    for s, c in sorted(specialty_counts.items(), key=lambda x: -x[1]):
        print("  {:<25} {}".format(s, c))
 
    with open(write_path, "w", encoding="utf-8") as f:
        for p in valid:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
 
    if writing_in_place:
        os.replace(write_path, OUT_PATH)
 
    print("\nSaved clean corpus to {}".format(OUT_PATH))
 
 
if __name__ == "__main__":
    main()
 