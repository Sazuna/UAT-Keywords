"""
Evaluate SciX's KAILAS model on the HP corpus.

https://zenodo.org/records/17460502
"""
import json
import os
from src.utils.config import ADS_HELIO_CORPUS_DIR, TEST_CORPUS_FILE
from tqdm import tqdm
from uat_utils import *
from document import Document
from src.utils.corpus_loader import Reader

from transformers import pipeline
classifier = pipeline("text-classification", model = "adsabs/KAILAS")
# zero_shot_classifier = pipeline("zero-shot-classification", model = "adsabs/KAILAS")

from datasets import load_dataset
dataset = load_dataset("adsabs/SciX_UAT_Keywords")

UAT_NAMESPACE = "http://astrothesaurus.org/uat/"

def sentencize(text):
    return [t for t in text.split('.') if t]

def get_kailas_training_corpus():
    # https://huggingface.co/datasets/adsabs/SciX_UAT_keywords
    print(dataset["train"])
    print("bibcodes:")
    print(dataset["train"]["bibcode"])
    return dataset["train"]["bibcode"]

kailas_bibcodes = get_kailas_training_corpus()

def compute_on_ads_corpus():
    FP, FN, TP = 0, 0, 0
    ignored = 0
    total = 0
    print(ADS_HELIO_CORPUS_DIR)
    for filename in tqdm(os.listdir(ADS_HELIO_CORPUS_DIR)):
        print(filename)
        with open(ADS_HELIO_CORPUS_DIR / filename, "r") as file:
            doc = json.load(file)
        bibcode = doc["bibcode"]
        if bibcode in kailas_bibcodes:
            ignored += 1
            continue
        total += 1
        document = Document(doc.get("bibcode", ""),
                            doc.get("title", "")[0],
                            doc.get("journal", ""),
                            doc.get("abstract", ""),
                            doc.get("keywords", ""),
                            None)
        # candidate_uris = []
        # for candidate_uris in document.get_best_keywords(top_k = 1):
        candidate_uris_by_sentence = list(document.get_best_keywords(top_k = 3))
        candidate_uris = sum(candidate_uris_by_sentence, [])

        """
        for sentence in sentencize(title + '.' + abstract + '.' + keywords):
            res = classifier(sentence)
            candidate_uris.append(res[0]["label"])
        """
        # Get the paper's UATs indexes
        # papers_uats = {keyword for keyword in keywords if keyword.isnumeric()}
        papers_uats = set(document.uats)
        output_uats = set(candidate_uris)

        # Evaluation
        # print(bibcode, title)
        # print("Papers UATs:", papers_uats)
        # print("Output UATs:", output_uats)
        TP += len(papers_uats & output_uats)
        FP += len(output_uats - papers_uats)
        FN += len(papers_uats - output_uats)

        # Propose related
        """
        for uri in candidate_uris.copy():
            # get their broaders to test better match
            #broaders = get_uat_broader(uri)
            #candidate_uris.extend(broaders)
            narrowers = get_uat_narrower(uri)
            candidate_uris.extend(narrowers)
        # Get labels from UATs ontology
        candidate_labels = [get_uat_label(uri) for uri in set(candidate_uris)]
        res = zero_shot_classifier(title + " " + abstract,
                                candidate_labels = candidate_labels,#sorted(set(candidate_labels)),
                                multi_label = True)
        print(res)
        """

    print("ADS corpus results")
    print(TP, FP, FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print("precision =", precision)
    print("recall =", recall)
    print("f1 =", f1)

    print("Total papers:", total)
    print("Ignored (in KAILAS training dataset):", ignored)

compute_on_ads_corpus()

def compute_on_test_corpus():
    corpus_reader = Reader()

    TP, FP, FN = 0, 0, 0
    for document in corpus_reader.read_pre9forADS():
        doi = document.bibcode
        text = document.text
        title = document.title
        uats = document.uats
        papers_uats = set(uats)
        candidate_uris = []
        for sentence in sentencize(text):
            res = classifier(sentence)
            candidate_uris.append(res[0]["label"])

        # Evaluation
        print(doi, title)
        print("Papers UATs:", ', '.join([f"{get_uat_label(u)} ({u})" for u in sorted(papers_uats)]))
        output_uats = set(candidate_uris)
        print("Output UATs:", ', '.join([f"{get_uat_label(u)} ({u})" for u in sorted(output_uats)]))
        print("\n")
        TP += len(papers_uats & output_uats)
        FP += len(output_uats - papers_uats)
        FN += len(papers_uats - output_uats)
    print("Test corpus results")
    print("TP:", TP, "FP:", FP, "FN:", FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    print("precision =", precision)
    print("recall =", recall)
    print("f1 =", f1)

compute_on_test_corpus()
