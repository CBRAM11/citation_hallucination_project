import json
import os
import re
 
INPUT_FILES = {
    "vanilla":        "outputs/vanilla_outputs.json",
    "rag":            "outputs/rag_outputs.json",
    "constrained_rag": "outputs/constrained_rag_outputs.json",
}
 
OUTPUT_PATH = "evaluation/extracted_citations.json"
 

 
_BAD_PATTERNS = [
    "paper title", "exact title", "1-3 sentences", "1-2 sentences",
    "answer:", "citations:", "<", ">", "{", "}", "write 1",
    "exact retrieved", "retrieved paper",
]
 
 
def is_bad_citation(text: str) -> bool:
    if not text:
        return True
    low = text.lower().strip()
    if low in {"none", "n/a", "null", "none."}:
        return True
    if any(p in low for p in _BAD_PATTERNS):
        return True
    if len(text) < 8 or len(text) > 300:
        return True
    
    if text.count(".") > 3:
        return True
    return False
 
 
def clean_citation(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\(\d{4}\)", "", text).strip()          
    text = re.sub(r"^\d+[\.\)]\s*", "", text).strip()      
    text = re.sub(r"^[-*]\s*", "", text).strip()            
    return text
 
 
def clean_citation_list(citations: list) -> list[str]:
    if not isinstance(citations, list):
        return []
    cleaned, seen = [], set()
    for c in citations:
        if not isinstance(c, str):
            continue
        c = clean_citation(c)
        if is_bad_citation(c):
            continue
        if c not in seen:
            cleaned.append(c)
            seen.add(c)
    return cleaned
 
 

 
def load_json(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
 
 
def main():
    os.makedirs("evaluation", exist_ok=True)
    extracted = {}
 
    for system, path in INPUT_FILES.items():
        if not os.path.exists(path):
            print(f"[WARN] {path} not found — skipping {system}")
            continue
 
        rows = load_json(path)
        system_records = []
 
        for row in rows:
            citations = clean_citation_list(row.get("citations", []))
 
            system_records.append({
                "id":               row["id"],
                "question":         row["question"],
                "domain":           row.get("domain", ""),
                "answer":           row.get("answer", ""),
                "raw_output":       row.get("raw_output", ""),
                
                "citation_candidates": citations,
                
                "retrieved_titles": row.get("retrieved_titles", []),
                "parse_error":      row.get("parse_error", False),
            })
 
        extracted[system] = system_records
        total_cits = sum(len(r["citation_candidates"]) for r in system_records)
        parse_errs = sum(1 for r in system_records if r["parse_error"])
        print(f"  {system:<20} {len(system_records)} rows, "
              f"{total_cits} citations, {parse_errs} parse errors")
 
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(extracted, f, indent=2, ensure_ascii=False)
 
    print(f"\nExtracted citations saved → {OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    main()