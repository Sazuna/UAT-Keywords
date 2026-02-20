#!/bin/env python3
"""
Generate embeddings of keywords using an LLM embedder
"""
import json
import numpy as np
from time import time
from rdflib import SKOS, DCTERMS
import encoder
from llm_connection import encode as llm_encode
from config import UATS_JSON, LLM_UATS_EMBEDDINGS_FILE

### AllMiniLM (test)
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def main():
    with open(UATS_JSON, "r") as file:
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
             SKOS.related,
             SKOS.broader,
             SKOS.narrower,
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
    p_str = [str(p) for p in all_p]
    uat_labels = dict()
    i_by_uri = dict()
    keys = sorted(uats)
    for i, key in enumerate(keys):
        i_by_uri[key] = i
        uat_str = ""
        uat = uats[key]
        label = None
        narrower = []
        broader = []
        related = []
        for p in p_str:
            value = uat.get(p, "")
            if value:
                if p == str(SKOS.scopeNote):
                    uat_str += "Scope: "
                elif p == str(SKOS.example):
                    uat_str += "Examples: "
                elif p == str(SKOS.prefLabel):
                    label = value
                elif p == str(SKOS.narrower):
                    narrower = value
                    continue
                elif p == str(SKOS.broader):
                    broader = value
                    continue
                elif p == str(SKOS.related):
                    related = value
                    continue
                uat_str += '; '.join(value) + '. '
        uat_labels[i] = [key, label[0], broader, narrower, related]
        uat_str = uat_str.strip()
        uat_str = uat_str.replace('..', '.')
        uats_str.append(uat_str)

    for key, label, narrowers, broaders, relateds in uat_labels.values():
        for i, related in enumerate(relateds):
            relateds[i] = i_by_uri[related]
        for i, narrower in enumerate(narrowers):
            narrowers[i] = i_by_uri[narrower]
        for i, broader in enumerate(broaders):
            broaders[i] = i_by_uri[broader]

    """
    start = time()
    embeddings = model.encode(uats_str)
    print(len(embeddings), "/ 2312 UATS (UAT 1 ignored)") # cat output.json | grep "uat" | sed -e "s/ *//g" | sed -e "s/[,:{\"]//g" | grep -e "^h.*" | sort | uniq | wc -l
    print("Elapsed:", time() - start)
    np.save(LLM_EMBEDDINGS_FILE, embeddings)
    """

    # HuggingFace model (AstroLLaMa, AstroBERT)
    """
    start = time()
    embeddings_astrollama = encoder.encode_batch(uats_str)
    print("Elapsed:", time() - start)
    print(type(embeddings_astrollama))
    np.save("embeddings_astrobert.npy", embeddings_astrollama)
    """
    # embeddings_llm = [llm_encode(uat_str) for uat_str in uats_str]
    embeddings_llm = llm_encode(uats_str)
    print(type(embeddings_llm))
    embeddings_llm = np.asarray(embeddings_llm, dtype = np.float32)
    print(embeddings_llm.shape)
    np.save(LLM_UATS_EMBEDDINGS_FILE, embeddings_llm)


if __name__ == "__main__":
    main()