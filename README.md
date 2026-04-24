# Citation Hallucination in LLMs: Vanilla vs RAG vs Constrained RAG

This repository contains the code, data pipeline, and evaluation framework for our final paper:

**Investigating Citation Hallucinations in Large Language Models with Retrieval-Augmented Generation**

## Overview

Large language models can generate fluent and convincing academic answers, but they may also produce **hallucinated citations** that look real while failing to match actual papers. This project studies that problem in a controlled experimental setting.

We compare three generation settings:

1. **Vanilla**  
   The language model answers a research-style question without retrieval.

2. **RAG**  
   The model receives retrieved papers as context before answering.

3. **Constrained RAG**  
   The model receives retrieved papers and is instructed to cite only retrieved sources.

The final benchmark uses:

- **510 medical research prompts**
- **2,989 PubMed-derived papers**
- automatic citation extraction and verification
- manual relevance auditing for support consistency

## Research Goal

The main question is:

> How frequently do large language models generate hallucinated scientific citations, and to what extent can retrieval grounding reduce this problem?

We evaluate not only whether citations are real, but also whether they are:

- matched to the corpus
- grounded in retrieval
- actually relevant to the question

## Repository Structure

```text
.
├── data/
│   ├── corpus.jsonl
│   └── prompts.json
│
├── retrieval/
│   ├── faiss.index
│   └── corpus_meta.pkl
│
├── outputs/
│   ├── vanilla_outputs.json
│   ├── rag_outputs.json
│   └── constrained_rag_outputs.json
│
├── evaluation/
│   ├── extracted_citations.json
│   ├── verified_citations.json
│   ├── metrics.json
│   ├── relevance_annotations.json
│   └── relevance_metrics.json
│
├── src/
│   ├── build_index.py
│   ├── retrieve.py
│   ├── generate.py
│   ├── extract_citations.py
│   ├── verify_citations.py
│   ├── evaluate.py
│   └── evaluate_relevance.py
│
├── paper/
│   └── final_paper.tex
│
└── README.md


## Method Summary

### 1. Corpus Construction
A local corpus of PubMed-derived papers is stored in `data/corpus.jsonl`. Each paper record includes metadata such as:

- title
- abstract
- journal
- publication date
- specialty
- authors
- MeSH terms

### 2. Retrieval Indexing
We build a FAISS index over sentence-transformer embeddings of the corpus. The indexed document text prioritizes the abstract, with title and MeSH terms used as additional context.

### 3. Generation Conditions
For each prompt, the system generates one response in each of the following settings:

- Vanilla
- RAG
- Constrained RAG

The model is instructed to return structured output containing:

- an answer
- a list of cited paper titles

### 4. Citation Extraction
Generated outputs are parsed and cleaned to recover citation candidates.

### 5. Citation Verification
Each citation candidate is compared against the local evaluation corpus using:

- normalized exact matching
- approximate / fuzzy title matching

Each citation is labeled as:

- **Valid**
- **Partial**
- **Hallucinated**

### 6. Manual Relevance Audit
A smaller manually annotated subset is used to evaluate whether valid citations actually support the question being asked.

Environment Setup
Python Version

Recommended: Python 3.10+

Create a Virtual Environment
Windows CMD
python -m venv venv
venv\Scripts\activate

Windows PowerShell
python -m venv venv
venv\Scripts\Activate.ps1

macOS / Linux
python3 -m venv venv
source venv/bin/activate

Install Dependencies

If you already have a requirements.txt, run:

pip install -r requirements.txt

Ollama Setup

The final large-scale run uses Ollama for local inference instead of a cloud API.

Install Ollama

Install Ollama for your operating system.

Pull the model

Example:

ollama pull llama3.2

Running the Full Pipeline

Run the following steps from the project root.

Step 1: Build the Retrieval Index
python src/build_index.py

This creates:

retrieval/faiss.index
retrieval/corpus_meta.pkl
Step 2: Run Generation
python src/generate.py

This generates outputs for:

Vanilla
RAG
Constrained RAG

Saved files:

outputs/vanilla_outputs.json
outputs/rag_outputs.json
outputs/constrained_rag_outputs.json

Step 3: Extract Citations
python src/extract_citations.py

Saved file:

evaluation/extracted_citations.json
Step 4: Verify Citations
python src/verify_citations.py

Saved file:

evaluation/verified_citations.json

Step 5: Compute Automatic Metrics
python src/evaluate.py

Saved file:

evaluation/metrics.json
Step 6: Run Manual Relevance Evaluation

First create or update:

evaluation/relevance_annotations.json

Then run:

python src/evaluate_relevance.py

Step 5: Compute Automatic Metrics
python src/evaluate.py

Saved file:

evaluation/metrics.json
Step 6: Run Manual Relevance Evaluation

First create or update:

evaluation/relevance_annotations.json

Then run:

python src/evaluate_relevance.py

Main Output Files
outputs/vanilla_outputs.json

Contains the raw and parsed outputs from the non-retrieval setting.

outputs/rag_outputs.json

Contains outputs generated with retrieved context.

outputs/constrained_rag_outputs.json

Contains outputs generated with retrieved context and citation constraints.

evaluation/extracted_citations.json

Stores extracted citation candidates from generated outputs.

evaluation/verified_citations.json

Stores verification labels for each citation candidate.

evaluation/metrics.json

