import json
import os
import re
import time

import ollama
from retrieve import Retriever

PROMPTS_PATH = "data/prompts.json"
OUTPUT_DIR   = "outputs"
MODEL        = "llama3.2" 



def load_prompts():
    with open(PROMPTS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_existing(filename):
    """Load already-completed results so the run can be resumed."""
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        print(f"  [RESUME] Loaded {len(results)} existing results from {path}")
        return results
    return []



def shorten(text: str, max_chars: int = 300) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "..."


def build_context(papers):
    chunks = []
    for i, p in enumerate(papers, start=1):
        title    = p.get("title", "").strip()
        authors  = p.get("authors_str", "").strip()
        journal  = p.get("journal", "").strip()
        pub_date = p.get("publication_date", "").strip()
        abstract = shorten(p.get("abstract", ""), 300)
        chunks.append(
            f"[{i}] Title: {title}\n"
            f"    Authors: {authors}\n"
            f"    Journal: {journal} ({pub_date})\n"
            f"    Abstract: {abstract}\n"
        )
    return "\n".join(chunks)




SYSTEM_PROMPT = (
    "You are a research assistant that answers scientific questions by citing "
    "published papers.\n\n"
    "Always respond in EXACTLY this format and do not add extra text:\n\n"
    "ANSWER: <1-3 sentence answer>\n"
    "CITATIONS:\n"
    "- <Exact paper title>\n"
    "- <Exact paper title>\n\n"
    "If you have no citations, write exactly:\n"
    "CITATIONS:\n"
    "- NONE"
)


def make_vanilla_prompt(question: str) -> str:
    return (
        "Answer the following research question using your knowledge of published "
        "scientific literature. Cite real paper titles.\n\n"
        f"Question: {question}"
    )


def make_rag_prompt(question: str, context: str) -> str:
    return (
        "Use the retrieved papers below to answer the question. "
        "Prefer titles from the retrieved papers when they are relevant, "
        "but you may also cite other real papers you know.\n\n"
        f"Retrieved papers:\n{context}\n"
        f"Question: {question}"
    )


def make_constrained_prompt(question: str, context: str) -> str:
    return (
        "Answer using ONLY the retrieved papers listed below. "
        "Do NOT invent any paper title. "
        "Copy titles EXACTLY as they appear. "
        "If none are relevant, write 'NONE' under CITATIONS.\n\n"
        f"Retrieved papers:\n{context}\n"
        f"Question: {question}"
    )




def call_llm(user_prompt: str) -> str:
    full_prompt = SYSTEM_PROMPT + "\n\n" + user_prompt

    for attempt in range(3):
        try:
            response = ollama.chat(
                model=MODEL,
                messages=[{"role": "user", "content": full_prompt}],
                options={"temperature": 0, "num_predict": 400},
            )
            return response["message"]["content"].strip()

        except Exception as e:
            msg = str(e)
            print(f"[WARN] Ollama call failed (attempt {attempt + 1}/3): {msg}")

            # Ollama not running — tell the user clearly
            if "connection" in msg.lower() or "refused" in msg.lower():
                print("[ERROR] Cannot reach Ollama. Make sure it is running:")
                print("        ollama serve")
                raise SystemExit(1)

            # Model not pulled yet
            if "model" in msg.lower() and "not found" in msg.lower():
                print(f"[ERROR] Model '{MODEL}' not found. Pull it first:")
                print(f"        ollama pull {MODEL}")
                raise SystemExit(1)

            wait_s = 5 * (attempt + 1)
            print(f"[INFO] Retrying in {wait_s}s ...")
            time.sleep(wait_s)

    return "ANSWER: Error\nCITATIONS:\n- NONE"



_BAD_PATTERNS = [
    "paper title", "exact title", "1-3 sentences", "1-2 sentences",
    "answer:", "citations:", "<", ">", "{", "}", "[retrieved",
]


def _is_bad_citation(text: str) -> bool:
    if not text or len(text) < 8 or len(text) > 300:
        return True
    low = text.lower().strip()
    if low in {"none", "n/a", "null", "none."}:
        return True
    if any(p in low for p in _BAD_PATTERNS):
        return True
    if text.count(".") > 3:
        return True
    return False


def clean_citations(citations):
    cleaned, seen = [], set()
    for c in citations:
        if not isinstance(c, str):
            continue
        c = re.sub(r"\s+", " ", c.strip())
        c = re.sub(r"^\d+[\.\)]\s*", "", c)
        c = re.sub(r"^[-*]\s*", "", c)
        c = c.strip()
        if _is_bad_citation(c):
            continue
        if c not in seen:
            cleaned.append(c)
            seen.add(c)
    return cleaned[:3]


def safe_parse_output(text: str) -> dict:
    answer_match = re.search(
        r"ANSWER\s*:\s*(.+?)(?=\nCITATIONS\s*:|$)",
        text, re.S | re.I
    )
    answer = answer_match.group(1).strip() if answer_match else text.strip()

    cit_match = re.search(
        r"CITATIONS\s*:\s*\n((?:[ \t]*[-*\d].*\n?)+)",
        text, re.I
    )

    citations = []
    if cit_match:
        for line in cit_match.group(1).strip().splitlines():
            line = line.strip()
            if line.startswith(("-", "*")):
                citations.append(line[1:].strip())
            elif re.match(r"^\d+[\.\)]", line):
                citations.append(re.sub(r"^\d+[\.\)]\s*", "", line).strip())

    return {
        "answer":      answer,
        "citations":   clean_citations(citations),
        "parse_error": not bool(cit_match),
    }




def save_outputs(filename, results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved -> {path}")



def main():
    prompts   = load_prompts()
    retriever = Retriever()

    # Resume from previous run if outputs already exist
    vanilla_results     = load_existing("vanilla_outputs.json")
    rag_results         = load_existing("rag_outputs.json")
    constrained_results = load_existing("constrained_rag_outputs.json")

    done_ids  = {r["id"] for r in vanilla_results}
    remaining = [p for p in prompts if p["id"] not in done_ids]
    print(f"[INFO] {len(done_ids)} done, {len(remaining)} remaining.\n")

    total = len(remaining)
    for i, item in enumerate(remaining, start=1):
        qid      = item["id"]
        question = item["question"]
        domain   = item.get("domain", "")
        print(f"[{i:03d}/{total}] {qid} ({domain}): {question[:65]}...")

        retrieved        = retriever.search(question, top_k=5)
        context          = build_context(retrieved)
        retrieved_titles = [p.get("title", "").strip() for p in retrieved]

        v_raw = call_llm(make_vanilla_prompt(question))
        r_raw = call_llm(make_rag_prompt(question, context))
        c_raw = call_llm(make_constrained_prompt(question, context))

        v_parsed = safe_parse_output(v_raw)
        r_parsed = safe_parse_output(r_raw)
        c_parsed = safe_parse_output(c_raw)

        vanilla_results.append({
            "id": qid, "question": question, "domain": domain,
            "raw_output":  v_raw,
            "answer":      v_parsed["answer"],
            "citations":   v_parsed["citations"],
            "parse_error": v_parsed["parse_error"],
        })
        rag_results.append({
            "id": qid, "question": question, "domain": domain,
            "retrieved_titles":  retrieved_titles,
            "retrieved_papers":  retrieved,
            "raw_output":        r_raw,
            "answer":            r_parsed["answer"],
            "citations":         r_parsed["citations"],
            "parse_error":       r_parsed["parse_error"],
        })
        constrained_results.append({
            "id": qid, "question": question, "domain": domain,
            "retrieved_titles":  retrieved_titles,
            "retrieved_papers":  retrieved,
            "raw_output":        c_raw,
            "answer":            c_parsed["answer"],
            "citations":         c_parsed["citations"],
            "parse_error":       c_parsed["parse_error"],
        })

        print(
            f"         vanilla={len(v_parsed['citations'])} | "
            f"rag={len(r_parsed['citations'])} | "
            f"constrained={len(c_parsed['citations'])} citations"
        )

        # Save after every question — no progress lost if you stop early
        save_outputs("vanilla_outputs.json",          vanilla_results)
        save_outputs("rag_outputs.json",              rag_results)
        save_outputs("constrained_rag_outputs.json",  constrained_results)

    print("\nAll done.")


if __name__ == "__main__":
    main()