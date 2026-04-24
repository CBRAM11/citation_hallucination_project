
from Bio import Entrez
import pandas as pd
import time
import json
import os
from datetime import datetime


Entrez.email = "cbram2003@gmail.com"




def safe_get_pub_date(article):
    pub_date = ""

    try:
        journal = article.get("Journal", {})
        journal_issue = journal.get("JournalIssue", {})
        pub_info = journal_issue.get("PubDate", {})

        year = str(pub_info.get("Year", "")).strip()
        month = str(pub_info.get("Month", "")).strip()
        day = str(pub_info.get("Day", "")).strip()

        if year and month and day:
            pub_date = f"{year}-{month}-{day}"
        elif year and month:
            pub_date = f"{year}-{month}"
        elif year:
            pub_date = year
    except Exception:
        pub_date = ""

    return pub_date


def safe_get_authors(article):
    authors = []

    try:
        if "AuthorList" in article:
            for author in article["AuthorList"]:
                if "LastName" in author:
                    name = f"{author.get('LastName', '')} {author.get('Initials', '')}".strip()
                    if name:
                        authors.append(name)
    except Exception:
        pass

    return authors


def safe_get_abstract(article):
    abstract = ""

    try:
        if "Abstract" in article:
            abstract_texts = article["Abstract"].get("AbstractText", [])
            abstract = " ".join([str(text) for text in abstract_texts]).strip()
    except Exception:
        abstract = ""

    return abstract


def safe_get_doi(pubmed_data):
    doi = ""

    try:
        article_ids = pubmed_data.get("ArticleIdList", [])
        for article_id in article_ids:
            attrs = getattr(article_id, "attributes", {})
            if attrs.get("IdType") == "doi":
                doi = str(article_id)
                break
    except Exception:
        doi = ""

    return doi


def safe_get_mesh_terms(medline_citation):
    mesh_terms = []

    try:
        if "MeshHeadingList" in medline_citation:
            for heading in medline_citation["MeshHeadingList"]:
                descriptor = heading.get("DescriptorName")
                if descriptor:
                    mesh_terms.append(str(descriptor))
    except Exception:
        pass

    return mesh_terms


def collect_pubmed_fixed(search_term, max_results=300, batch_size=100):
    print(f"Collecting articles for: {search_term}")

    try:
        handle = Entrez.esearch(
            db="pubmed",
            term=search_term,
            retmax=max_results,
            sort="relevance"
        )
        results = Entrez.read(handle)
        handle.close()

        pmids = results["IdList"]
        total_count = int(results["Count"])

        print(f"Found {total_count} total articles")
        print(f"Retrieving {len(pmids)} articles\n")

        if not pmids:
            print("No PMIDs found!")
            return []

    except Exception as e:
        print(f"Search failed: {e}")
        return []

    articles = []

    for i in range(0, len(pmids), batch_size):
        batch_pmids = pmids[i:i + batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (len(pmids) + batch_size - 1) // batch_size

        print(f"Fetching batch {batch_num}/{total_batches} ({len(batch_pmids)} articles)...")

        try:
            handle = Entrez.efetch(
                db="pubmed",
                id=batch_pmids,
                rettype="abstract",
                retmode="xml"
            )
            records = Entrez.read(handle)
            handle.close()

            for record in records["PubmedArticle"]:
                try:
                    medline_citation = record["MedlineCitation"]
                    article = medline_citation["Article"]
                    pubmed_data = record.get("PubmedData", {})

                    pmid = str(medline_citation["PMID"])
                    title = str(article.get("ArticleTitle", "")).strip()
                    abstract = safe_get_abstract(article)
                    journal = str(article.get("Journal", {}).get("Title", "")).strip()
                    pub_date = safe_get_pub_date(article)
                    authors = safe_get_authors(article)
                    doi = safe_get_doi(pubmed_data)
                    mesh_terms = safe_get_mesh_terms(medline_citation)

                    
                    article_data = {
                        "paper_id": pmid,
                        "pmid": pmid,
                        "title": title,
                        "abstract": abstract,
                        "journal": journal,
                        "publication_date": pub_date,
                        "authors": authors,
                        "authors_str": ", ".join(authors),
                        "num_authors": len(authors),
                        "doi": doi,
                        "mesh_terms": mesh_terms,
                        "has_abstract": bool(abstract),
                        "abstract_length": len(abstract)
                    }

                    if title:
                        articles.append(article_data)

                except Exception as e:
                    print(f"Warning: skipped one article: {str(e)[:80]}")
                    continue

            print(f"Collected {len(articles)} articles so far")
            time.sleep(0.4)

        except Exception as e:
            print(f"Batch {batch_num} failed: {e}")
            continue

    print(f"\nCollection complete: {len(articles)} records\n")
    return articles


def collect_medical_specialities(max_per_speciality=300):
    os.makedirs("data_specialities_checkpoints", exist_ok=True)

    # 10 specializations
    specialities = {
        "cardiology": 'cardiology AND 2020:2024[dp] AND English[la]',
        "oncology": 'oncology AND 2020:2024[dp] AND English[la]',
        "neurology": 'neurology AND 2020:2024[dp] AND English[la]',
        "radiology": 'radiology AND 2020:2024[dp] AND English[la]',
        "psychiatry": 'psychiatry AND 2020:2024[dp] AND English[la]',
        "endocrinology": 'endocrinology AND 2020:2024[dp] AND English[la]',
        "infectious_disease": '"infectious disease" AND 2020:2024[dp] AND English[la]',
        "pediatrics": 'pediatrics AND 2020:2024[dp] AND English[la]',
        "pulmonology": 'pulmonology AND 2020:2024[dp] AND English[la]',
        "gastroenterology": 'gastroenterology AND 2020:2024[dp] AND English[la]'
    }

    all_articles = []
    seen_pmids = set()

    for speciality_name, query in specialities.items():
        print("=" * 80)
        print(f"COLLECTING: {speciality_name.upper()}")
        print("=" * 80)

        articles = collect_pubmed_fixed(query, max_results=max_per_speciality)

        speciality_articles = []
        for article in articles:
            pmid = article["pmid"]
            if pmid in seen_pmids:
                continue

            seen_pmids.add(pmid)
            article["specialty"] = speciality_name
            article["collection_date"] = datetime.now().strftime("%Y-%m-%d")
            speciality_articles.append(article)

        all_articles.extend(speciality_articles)

        print(f"{speciality_name}: {len(speciality_articles)} unique articles collected")
        print(f"Total unique articles so far: {len(all_articles)}\n")

        checkpoint_df = pd.DataFrame(speciality_articles)
        checkpoint_df.to_csv(
            f"data_specialities_checkpoints/{speciality_name}_checkpoint.csv",
            index=False
        )
        print(f"Checkpoint saved: data_specialities_checkpoints/{speciality_name}_checkpoint.csv\n")

        time.sleep(2)

    return all_articles


def save_final_outputs(all_articles):
    df_final = pd.DataFrame(all_articles)

    df_final.to_csv("medical_literature_dataset.csv", index=False)
    df_final.to_json("medical_literature_dataset.json", orient="records", indent=2)

    
    with open("medical_literature_dataset.jsonl", "w", encoding="utf-8") as f:
        for row in all_articles:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print("COLLECTION COMPLETE")
    print(f"\nTotal unique articles: {len(df_final)}")

    if "specialty" in df_final.columns:
        print("\nSpecialty distribution:")
        print(df_final["specialty"].value_counts())

    print("\nFiles saved:")
    print("medical_literature_dataset.csv")
    print("medical_literature_dataset.json")
    print("medical_literature_dataset.jsonl")


if __name__ == "__main__":
    medical_data = collect_medical_specialities(max_per_speciality=300)
    save_final_outputs(medical_data)