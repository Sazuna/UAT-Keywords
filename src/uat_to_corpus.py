#!/bin/env python3
"""
Transforms UATs into text chunks
"""
import atexit
import pathlib
from rdflib import Graph, SKOS, OWL, DCTERMS, Literal, XSD
from collections import defaultdict
from llm_connection import generate, save_cache
from config import DATA_DIR, UATS_JSON
import json


def augment_definition(definition: str, key: str):
    """
    Verbalization of UATs for UATs with no definition.
    """
    prompt = f"Define this astrophysics concept: {definition}. Maximum 1 sentence. Maximum 30 words. No examples. No bullet points."
    return generate(prompt, cache_key = key)


def main(uat_file: pathlib.Path):
    g = Graph()
    g.parse(uat_file)
    description_by_uat = defaultdict(lambda: defaultdict(list))
    all_p = {# RDF.type,
             OWL.deprecated,
             # RDFS.comment, # Blank nodes
             SKOS.prefLabel,
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
             }
    for p in all_p:
        for s, _, o in g.triples((None, p, None)):
            if o not in description_by_uat[s][p]:
                description_by_uat[s][p].append(o)

    for s, _, o in g.triples((None, OWL.deprecated,  Literal(True, datatype=XSD.boolean))):
        del description_by_uat[s]

    # Remove UATs that do not have a prefLabel (uat/1)
    for uat, values in description_by_uat.copy().items():
        if not SKOS.prefLabel in values:
            del description_by_uat[uat]
            continue
        #if not SKOS.definition in values:
        #    definition = augment_definition(str(values[SKOS.prefLabel][0]), key = str(uat))
        #    values[SKOS.definition] = [definition]



    with open(UATS_JSON, "w") as file:
        json.dump(description_by_uat, file, indent = 2)


if __name__ == "__main__":
    main(DATA_DIR / "UAT_v6.0.0.rdf")
