#!/bin/env python3
"""
Generate embeddings of keywords using an LLM embedder
"""
import json
import numpy as np
from rdflib import SKOS, DCTERMS
from time import time

from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def main():
    with open("output.json", "r") as file:
        uats = json.load(file)


    all_p = [SKOS.prefLabel,
             SKOS.altLabel,
             SKOS.definition,
             SKOS.example,
             SKOS.scopeNote,
             # SKOS.changeNote,
             # SKOS.editorialNote,
             # SKOS.topConceptOf,
             # SKOS.hasTopConcept,
             # SKOS.related,
             # SKOS.broader,
             # SKOS.narrower,
             DCTERMS.description,
             # DCTERMS.title, # Blank nodes
             # DCTERMS.created,
             # DCTERMS.modified,
             # DCTERMS.contributor,
             # DCTERMS.creator,
             # DCTERMS.publisher,
             # DCTERMS.subject,
    ]
    uats_str = []
    keys = sorted(uats)
    p_str = [str(p) for p in all_p]
    uat_labels = dict()
    for i, key in enumerate(keys):
        uat_str = ""
        uat = uats[key]
        label = None
        for p in p_str:
            value = uat.get(p, "")
            if value:
                if p == str(SKOS.scopeNote):
                    uat_str += "Scope: "
                elif p == str(SKOS.example):
                    uat_str += "Examples: "
                elif p == str(SKOS.prefLabel):
                    label = value
                uat_str += '; '.join(value) + '. '
        uat_labels[i] = [key, label[0]]
        uat_str = uat_str.strip()
        uat_str = uat_str.replace('..', '.')
        uats_str.append(uat_str)

    start = time()
    embeddings = model.encode(uats_str)
    print(len(embeddings), "/ 2312 UATS (UAT 1 ignored)") # cat output.json | grep "uat" | sed -e "s/ *//g" | sed -e "s/[,:{\"]//g" | grep -e "^h.*" | sort | uniq | wc -l
    print("Elapsed:", time() - start)
    np.save("embeddings.npy", embeddings)

    with open("uat_labels.json", "w") as file:
        json.dump(uat_labels, file, indent = 2)


if __name__ == "__main__":
    main()