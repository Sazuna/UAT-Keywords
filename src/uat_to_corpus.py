#!/bin/env python3
"""
Transforms UAT into text chunks
"""

from rdflib import Graph, SKOS, OWL, RDFS, DCTERMS, RDF, Literal, XSD
from collections import defaultdict
import json

def main(uat_file: str):
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


    with open("output.json", "w") as file:
        json.dump(description_by_uat, file, indent = 2)


if __name__ == "__main__":
    main("../corpus/UAT_v6.0.0.rdf")