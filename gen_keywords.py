"""
Generate keywords for a bibtex document.
"""

import argparse
from src.utils.corpus_loader import Reader
from src.KAILAS.classifier import classifier as kailas
from src.label_match.label_match import label_match
from src.utils.uat_utils import get_uat_label
from src.tfidf.main import classify as tfidf

TOP_K = 10
def main(input_file):
    reader = Reader()
    for document in reader.read_bibtex():
        text = document.text
        kailas_keywords = kailas(text, truncation = True, max_length = 512, top_k = TOP_K)
        print(document.title)
        print(kailas_keywords)
        print("KAILAS top10")
        for keyword in sorted(kailas_keywords, key = lambda x: x["score"], reverse = True):
            keyword = get_uat_label(keyword["label"])
            print(keyword)
        label_match_keywords = label_match(text)
        print("Label match")
        for keyword in label_match_keywords:
            print(keyword)
        print("Tf-idf")
        tfidf_keywords = tfidf(text)
        for keyword in tfidf_keywords:
            print(keyword)
        exit()
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input")
    args = parser.parse_args()

    main(args.input)
