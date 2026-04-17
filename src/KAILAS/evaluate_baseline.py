"""
Evaluate SciX's KAILAS model on the HP corpus.

https://zenodo.org/records/17460502
"""
import json
import os
from tqdm import tqdm
from src.utils.config import ADS_CORPUS_DIR, ADS_HELIO_CORPUS_DIR, TEST_CORPUS_FILE
from src.KAILAS import uat_utils
from src.utils.corpus_loader import Reader
from src.utils.util import print_results
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer

import heapq
from transformers import pipeline
from transformers import AutoTokenizer

MODEL    = "adsabs/KAILAS"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

classifier = pipeline("text-classification", model = MODEL, tokenizer = tokenizer)#, return_all_scores = True)
zero_shot_classifier = pipeline("zero-shot-classification", model = "adsabs/KAILAS", tokenizer = tokenizer)

from datasets import load_dataset
dataset = load_dataset("adsabs/SciX_UAT_Keywords")

UAT_NAMESPACE = "http://astrothesaurus.org/uat/"
TOP_K = 10


def top_k_scores(scores, k):
    return(heapq.nlargest(k, scores, key=lambda x: x['score']) )

def compute_on_ads_corpus(sentencize: bool = False):
    y_true = []
    y_pred = []
    total = 0
    reader = Reader()
    micro_precision, macro_precision, micro_recall, macro_recall, micro_f1, macro_f1 = 0, 0, 0, 0, 0, 0
    for document in reader.read_corpus(ignore_kailas = False, # It is not trained on the same corpus anyways !
                                       corpus_folder = ADS_HELIO_CORPUS_DIR):
        candidate_uris = []
        # for candidate_uris in document.get_best_keywords(top_k = 1):
        # candidate_uris_by_sentence = list(document.get_best_keywords(top_k = 3))
        # candidate_uris = sum(candidate_uris_by_sentence, [])
        if sentencize:
            sent_uri = []
            for sentence in document.sentencize():
                res = classifier(sentence, truncation = True, max_length = 512)
                res = top_k_scores(res, 1)
                sent_uri += res[0]["label"]
            candidate_uris = sent_uri
        else:
            res = classifier(document.text, truncation = True, max_length = 512, top_k = TOP_K)
            candidate_uris = [r["label"] for r in res[0:TOP_K]]
        uat_labels = [uat.split('/')[-1] for uat in document.uats]
        y_pred.append(list(set(candidate_uris)))
        y_true.append(uat_labels)
        total += 1
    print_results(y_true, y_pred, "ADS", total)

compute_on_ads_corpus(False)

def compute_on_test_corpus(sentencize: bool = False):
    corpus_reader = Reader()
    y_pred = []
    y_true = []

    total = 0
    for document in corpus_reader.read_pre9forADS():
        doi = document.bibcode
        text = document.text
        title = document.title
        uats = document.uats
        papers_uats = set(uats)
        candidate_uris = []
        candidate_uris_scores = []

        if sentencize:
            for sentence in document.sentencize():
                res = classifier(sentence, truncation = True, max_length = 512)
                candidate_uris.append(res[0]["label"])
        else:
            res = classifier(document.text, truncation = True, max_length = 512, top_k = TOP_K)
            candidate_uris.extend([r["label"] for r in res[0:TOP_K]]) # Does not work (pipeline only returns one element)
            candidate_uris_scores.extend([r["score"] for r in res[0:TOP_K]])
        uat_labels = [uat.split('/')[-1] for uat in document.uats]
        y_pred.append(list(set(candidate_uris)))
        y_true.append(uat_labels)

        total += 1
        print(doi, title)
        print("Papers UATs:", ', '.join([f"{uat_utils.get_uat_label(u)} ({u})" for u in sorted(papers_uats)]))
        # output_uats = set(candidate_uris)
        print("Output UATs:", '\n\t'.join([f"{score} {uat_utils.get_uat_label(u)} ({u})" for score, u in zip(candidate_uris, candidate_uris_scores)]))
        print("\n")
    print_results(y_true, y_pred, "preprint", total)

compute_on_test_corpus(False)
