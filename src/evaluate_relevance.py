import json
import os
from collections import defaultdict
 
INPUT_PATH  = "evaluation/relevance_annotations.json"
OUTPUT_PATH = "evaluation/relevance_metrics.json"
 
 
def compute_rates(counts: dict) -> dict:
    total = counts.get("total", 0)
    if total == 0:
        return {**counts, "relevant_rate": 0.0, "partial_rate": 0.0, "irrelevant_rate": 0.0}
    return {
        **counts,
        "relevant_rate":   round(counts.get("relevant",   0) / total, 4),
        "partial_rate":    round(counts.get("partial",    0) / total, 4),
        "irrelevant_rate": round(counts.get("irrelevant", 0) / total, 4),
    }
 
 
def main():
    if not os.path.exists(INPUT_PATH):
        print(
            f"[INFO] {INPUT_PATH} not found.\n"
            "This file requires manual annotation.\n"
            "Create it by reviewing ~10-20% of citation checks from "
            "evaluation/verified_citations.json and labelling each as "
            "'relevant', 'partial', or 'irrelevant'."
        )
        return
 
    os.makedirs("evaluation", exist_ok=True)
 
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)
 
    
    by_system = defaultdict(lambda: {"relevant": 0, "partial": 0, "irrelevant": 0, "total": 0})
 
    
    by_domain = defaultdict(lambda: defaultdict(
        lambda: {"relevant": 0, "partial": 0, "irrelevant": 0, "total": 0}
    ))
 
    valid_labels = {"relevant", "partial", "irrelevant"}
 
    for row in rows:
        system = row.get("system", "unknown")
        domain = row.get("domain", "unknown")
        label  = row.get("relevance", "").lower().strip()
 
        if label not in valid_labels:
            print(f"[WARN] Unknown label '{label}' for {row.get('id')} — skipping")
            continue
 
        by_system[system][label]  += 1
        by_system[system]["total"] += 1
        by_domain[system][domain][label]  += 1
        by_domain[system][domain]["total"] += 1
 
    
    output = {}
    for system, counts in by_system.items():
        output[system] = {
            "overall": compute_rates(dict(counts)),
            "by_domain": {
                domain: compute_rates(dict(dcounts))
                for domain, dcounts in by_domain[system].items()
            },
        }
 
    
    print("\nSupport-Consistency (Relevance) Results")
    print("=" * 55)
    for system, data in output.items():
        o = data["overall"]
        print(f"\n{system.upper()}")
        print(f"  Annotated citations : {o['total']}")
        print(f"  Relevant rate       : {o['relevant_rate']:.1%}")
        print(f"  Partial rate        : {o['partial_rate']:.1%}")
        print(f"  Irrelevant rate     : {o['irrelevant_rate']:.1%}")
 
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
 
    print(f"\nRelevance metrics saved → {OUTPUT_PATH}")
 
 
if __name__ == "__main__":
    main()