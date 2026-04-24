import json
import os
 
INPUT_PATH  = "evaluation/verified_citations.json"
OUTPUT_PATH = "evaluation/metrics.json"
 
 
def compute_metrics(system: str, rows: list[dict]) -> dict:
    total_questions  = len(rows)
    total_citations  = 0
    valid            = 0
    partial          = 0
    hallucinated     = 0
    grounded         = 0      
    grounded_eligible = 0     
    parse_errors     = 0
    no_citation_qs   = 0
 
    is_rag = system in {"rag", "constrained_rag"}
 
    for row in rows:
        checks = row.get("checks", [])
 
        if row.get("parse_error", False):
            parse_errors += 1
        if not checks:
            no_citation_qs += 1
 
        for check in checks:
            total_citations += 1
            status = check.get("status", "hallucinated")
            if status == "valid":
                valid += 1
            elif status == "partial":
                partial += 1
            else:
                hallucinated += 1
 
            if is_rag:
                grounded_eligible += 1
                if check.get("grounded", False):
                    grounded += 1
 
    def rate(num, den):
        return round(num / den, 4) if den else 0.0
 
    return {
        
        "total_questions":         total_questions,
        "total_citations":         total_citations,
        "valid":                   valid,
        "partial":                 partial,
        "hallucinated":            hallucinated,
 
        "hallucination_rate":      rate(hallucinated, total_citations),
        "valid_rate":              rate(valid,        total_citations),
        "partial_rate":            rate(partial,      total_citations),
 
       
        "retrieval_grounding_rate": rate(grounded, grounded_eligible),
 
        
        "parse_errors":            parse_errors,
        "parse_error_rate":        rate(parse_errors,   total_questions),
        "no_citation_rate":        rate(no_citation_qs, total_questions),
        "avg_citations_per_question": rate(total_citations, total_questions),
    }
 
 
def print_table(metrics: dict) -> None:
    """Pretty-print a comparison table to stdout."""
    systems = list(metrics.keys())
    col_w   = 32
 
    header = f"{'Metric':<{col_w}}" + "".join(f"{s:>20}" for s in systems)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
 
    row_keys = [
        ("total_questions",          "Total questions"),
        ("total_citations",          "Total citations"),
        ("valid",                    "  Valid"),
        ("partial",                  "  Partial"),
        ("hallucinated",             "  Hallucinated"),
        (None, ""),
        ("hallucination_rate",       "Hallucination rate"),
        ("valid_rate",               "Valid rate"),
        ("partial_rate",             "Partial rate"),
        ("retrieval_grounding_rate", "Retrieval grounding rate"),
        (None, ""),
        ("avg_citations_per_question", "Avg citations / question"),
        ("parse_error_rate",         "Parse error rate"),
        ("no_citation_rate",         "No-citation rate"),
    ]
 
    for key, label in row_keys:
        if key is None:
            print()
            continue
        vals = ""
        for s in systems:
            v = metrics[s].get(key, "-")
            if isinstance(v, float):
                vals += f"{v:>20.4f}"
            else:
                vals += f"{v:>20}"
        print(f"{label:<{col_w}}{vals}")
 
    print("=" * len(header) + "\n")
 
 
def main():
    os.makedirs("evaluation", exist_ok=True)
 
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        verified = json.load(f)
 
    metrics = {}
    for system, rows in verified.items():
        metrics[system] = compute_metrics(system, rows)
 
    print_table(metrics)
 
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
 
    print(f"Metrics saved → {OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    main()