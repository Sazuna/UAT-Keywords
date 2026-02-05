#!/bin/env python3
"""
Find bests UAT matches for each paper.
"""

import json
import numpy as np
from collections import defaultdict
from typing import Iterable

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def read_pre9forADS() -> Iterable[tuple[str, str]]:

    with open("../corpus/pre9forADS_all.dat", "r") as file:
        lines = file.readlines()
        all_docs = dict()
        doc = None
        state = None
        prefix = ""
        doi = None
        for line in lines:
            if line.startswith("%R"): # reference
                # all_docs.append(doc)
                if doi:
                    all_docs[doi] = doc
                doc = defaultdict(str)
                state = "reference"
                prefix = "%R"
            elif line.startswith("%T"):
                state = "title"
                prefix = "%T"
            elif line.startswith("%B"):
                state = "abstract"
                prefix = "%B"
            elif line.startswith("%A"):
                state = "authors"
                prefix = "%A"
            elif line.startswith("%F"):
                state = "affiliation"
                prefix = "%F"
            elif line.startswith("%I"):
                state = "DOI"
                prefix = "%I"
            elif line.startswith("%K"):
                state = "Keywords"
                prefix = "%K"
            elif line.startswith("%C"):
                state = "copyright"
                prefix = "%C"
            elif line.startswith("%D"):
                state = "date"
                prefix = "%D"
            elif line.startswith("%J"):
                state = "journal"
                prefix = "%J"
            elif line.startswith("%R"):
                state = "r..."
                prefix = "%R"
            elif line.startswith("%Z"):
                state = "z..."
                prefix = "%Z"

            if not state:
                continue
            elif state == "DOI":
                doi = line.removeprefix(prefix).strip()
            else:
                doc[state] += line.removeprefix(prefix).strip() + ' '


        for doi, doc in all_docs.items():
            doc_str = doc.get("title", "") + doc.get("abstract", "")
            yield doi, doc_str


def main():
    with open("embeddings.npy", "rb") as file:
        uat_embeddings = np.load(file)
        # Normalize for better performance
        uat_embeddings = uat_embeddings / np.linalg.norm(uat_embeddings, axis=1, keepdims=True)
    with open("uat_labels.json", "r") as file:
        uat_labels = json.load(file)

    for doi, doc_str in read_pre9forADS():
        print("----------------------------------------------")
        print(doi, doc_str)    
        query_emb = model.encode(doc_str)
        query_emb = query_emb / np.linalg.norm(query_emb)
        scores = uat_embeddings @ query_emb

        best_id = np.argmax(scores)
        best_score = scores[best_id]
        print(best_id, best_score)
        print("UAT LABEL:", uat_labels[str(int(best_id))])




if __name__ == "__main__":
    main()