from config import ADS_CORPUS_DIR, ADS_HELIO_CORPUS_DIR
import os
import glob

from corpus_loader import Reader
from ontology_graph import OntologyGraph

def main():
    reader = Reader()
    onto = OntologyGraph()
    text_list = []
    uats_list = []
    for text, uats in reader.read_corpus(ignore_kailas = True, ADS_HELIO_CORPUS_DIR):
        text_list.append(text)
        uats_list.append(uats)
        
    ds = onto.corpus_to_hf_dataset(texts, uats)
    ds.save_to_disk("ads_helio_ds.hf")
