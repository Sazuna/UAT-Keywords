#!/bin/env python3
"""
Generate embeddings of keywords using an LLM embedder
"""
import json
import numpy as np
import pickle
from time import time
from rdflib import SKOS, DCTERMS
from src.AstroBERT import encoder
from src.corpus import uat_to_corpus
from src.utils.llm_connection import encode as llm_encode
from src.utils.config import UATS_JSON, CORPUS_DIR, UATS_LABELS_JSON, BERT_UATS_EMBEDDINGS_FILE, UATS_JSON_VERBALIZED, BERT_UATS_EMBEDDINGS_FILE_VERBALIZED

def main(only_label: bool = False,
         verbalization: bool = False):
    if verbalization:
        if not UATS_JSON_VERBALIZED.exists():
            uat_to_corpus.main(CORPUS_DIR / "UAT_v6.0.0.rdf", verbalization=True)
        with open(UATS_JSON_VERBALIZED, "r") as file:
            uats = json.load(file)
    else:
        if not UATS_JSON.exists():
            uat_to_corpus.main(CORPUS_DIR / "UAT_v6.0.0.rdf", verbalization=False)
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
    keys = sorted(uats, key = lambda x: int(x.split('/')[-1]))
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
                    if only_label:
                        continue
                    uat_str += "Scope: "
                elif p == str(SKOS.example):
                    if only_label:
                        continue
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
                uat_str += ' '.join(value)
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

    # HuggingFace model (AstroLLaMa, AstroBERT)
    start = time()
    embeddings = encoder.astrobert_encode(uats_str)
    print("Elapsed:", time() - start)
    #with open(BERT_UATS_EMBEDDINGS_FILE, "wb") as file:
    #    pickle.dump(embeddings, file)
    if not verbalization:
        np.save(BERT_UATS_EMBEDDINGS_FILE, arr=embeddings, allow_pickle=True)
    else:
        np.save(BERT_UATS_EMBEDDINGS_FILE_VERBALIZED, arr=embeddings, allow_pickle=True)
    with open(UATS_LABELS_JSON, "w") as file:
        json.dump(uat_labels, file, indent = 2)

    return embeddings

if __name__ == "__main__":
    main(only_label = True)
