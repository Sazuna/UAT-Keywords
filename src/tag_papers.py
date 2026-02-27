#!/bin/env python3
"""
Find bests UAT matches for each paper.
"""

import json
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import Iterable
import regex
import matplotlib.pyplot as plt
from encoder import astrobert_encode

from config import BERT_UATS_EMBEDDINGS_FILE, UATS_LABELS_JSON


from sentence_transformers import SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


TEST = False
PLOT = False
p = print
def print(*string, force = False):
    if TEST or force:
        p(string)


# Embeddings
with open(BERT_UATS_EMBEDDINGS_FILE, "rb") as file:
    uat_embeddings = np.load(file, allow_pickle = True)
    # Normalize for better performance
    uat_embeddings = uat_embeddings / np.linalg.norm(uat_embeddings, axis=1, keepdims=True)

# UATs informations
with open(UATS_LABELS_JSON, "r") as file:
    uat_labels = json.load(file)

class UatUtils():
    def get_uat(idx: int):
        idx = str(int(idx))
        return uat_labels[idx]


    def get_uat_label(idx: int):
        return UatUtils.get_uat(idx)[1]


    def get_uat_broaders(idx: int):
        return UatUtils.get_uat(idx)[2]


    def get_and_print_top_k_keywords(top_idx: list[int]) -> list[str]:
        """
        Return and print top_k keywords.

        Args:
            top_idx: indexes of UATs in uat_labels
        """
        labels = []
        for rank, top_id in enumerate(top_idx):
            print(f"Top {rank + 1}: ", top_id, force = False)
            uat_info = UatUtils.get_uat(top_id)
            print("UAT info:", uat_info)
            uat_uri = uat_info[0]
            uat_label = uat_info[1]
            print(uat_uri, uat_label, force = False)
            labels.append(uat_label)
        return labels

    def get_top_infos(top_idx: list[int]) -> tuple[list[str]]:
        """
        Return lists of URIs, labels, broaders, narrowers, and relateds
        for each index in top_idx.
        """
        uris = []
        labels = []
        broaders = []
        narrowers = []
        relateds = []
        for top_id in top_idx:
            uat_info = uat_labels[str(int(top_id))]
            uris.append(uat_info[0])
            labels.append(uat_info[1])
            broaders.append(uat_info[2])
            narrowers.append(uat_info[3])
            relateds.append(uat_info[4])
        return uris, labels, broaders, narrowers, relateds


    def get_uat_categories(uat_idx: np.int64):
        """
        Find the keyword's category (top hierarchy under root)
        """
        broaders = UatUtils.get_uat_broaders(uat_idx)
        print("broaders:", broaders, force = True)
        if not broaders:
            print(uat_idx, force = True)
            yield uat_idx
        else:
            for broader in broaders:
                yield from UatUtils.get_uat_categories(broader)



class Reader():
    def read_pre9forADS(self) -> Iterable[tuple[str, str, str]]:
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
                keywords = doc.get("keywords", "").strip()
                title = doc.get("title", "").strip()
                abstract = doc.get("abstract", "").strip()
                doc_str = title.strip() + '. ' + doc.get("abstract", "")  + '. ' + keywords
                doc_str = regex.sub('\.+', '\.', doc_str)
                yield doi, title, abstract, keywords, doc_str


    def __iter__(self):
        return self.read_pre9forADS()


class DocumentProcesser():
    """
    Saves a document
    """
    def __init__(self,
                 doi: str,
                 title: str,
                 abstract: str,
                 papers_keywords: str,
                 doc_str: str):
        self._doi = doi
        self._title = title
        self._abstract = abstract
        self._papers_keywords = papers_keywords
        self._doc_str = doc_str


    ### Iterate over sentences ###
    def sentencize(self) -> list[str]:
        """
        Cut by sentences to extract sub-categories.
        """
        return [s.strip() for s in regex.split(r"\.|\?|\!", self._doc_str) if s.strip()]


    def sentencize_with_overlap(self) -> list[tuple[str]]:
        """
        Cut by sentences two by two to extract sub-categories.
        FIXME the first and last sentences are under represented
        """
        sentences = self.sentencize()
        return zip(sentences[:-1], sentences[1:])

    def __iter__(self):
        #for x in self.sentencize():
        #    yield x
        yield self._doc_str


    def process(self):
        top_k = 3
        all_labels_by_sentences = []
        query_embs = astrobert_encode([txt for txt in self]) # model.encode(txt)
        query_embs = query_embs / np.linalg.norm(query_embs, axis=1, keepdims=True)
        keywords_score_by_batch = defaultdict(float)
        for query_emb in query_embs:
            scores = uat_embeddings @ query_emb
            if PLOT:
                self.plot_top_k_curve(top_k = top_k, scores = scores)
            top_idx = self.get_top_idx(top_k = top_k, scores = scores)
            for idx, score in zip(top_idx, scores):
                keywords_score_by_batch[idx] += score
            # Find clusters on different levels
            self.graph_clusters(top_idx)
            keywords = UatUtils.get_and_print_top_k_keywords(top_idx)
            affixes_counts = self.get_affixes_counts(keywords = keywords)
            # print(affixes_counts, force = True)
            uris, labels, broaders, narrowers, relateds = UatUtils.get_top_infos(top_idx)
            # 1-level relations between entities
            counts, counts_sum = self.count_relations(top_idx,
                                                      broaders,
                                                      narrowers,
                                                      [], # relateds
                                                      )
        print(self._doi, self._title, self._doc_str, self._papers_keywords, force = True)
        bests_idx = sorted(keywords_score_by_batch.items(), key = lambda x: x[1], reverse = True)
        for idx, score in bests_idx:
            for uat_category in UatUtils.get_uat_categories(idx):
                category_label = UatUtils.get_uat_label(uat_category)
                print(uat_category, category_label, force = True)
        bests_keywords = [UatUtils.get_uat_label(idx) for idx, _ in bests_idx]
        print(bests_keywords, force = True)


    ### Functions to clusterize ###
    def get_affixes_counts(self,
                           keywords: list[str]) -> dict[int]:
        """
        Remove keywords that are suffixes or prefixes of another keyword from the list.
        Count the amount of prefixes and suffixes of each word in the initial list.
        The output is a count of clusters by affixes, keeping the longest labels.
        """
        counts = dict()
        for keyword in keywords:
            counts[keyword] = 1

        for keyword in keywords.copy():
            for keyword2 in keywords.copy():
                if keyword == keyword2:
                    continue
                if keyword.startswith(keyword2):
                    keywords.remove(keyword2)
                    counts[keyword] += counts.pop(keyword2)
                    break
                elif keyword.endswith(keyword2):
                    keywords.remove(keyword2)
                    counts[keyword] += counts.pop(keyword2)
                    break
        return counts


    def count_relations(self,
                        top_k: list,
                        broaders: list[list],
                        narrowers: list[list],
                        relateds: list[list]) -> pd.DataFrame:
        """
        Create sub graphs (represented as tables)
        """
        not_in_top = []
        df = pd.DataFrame(
            data=0,
            index=top_k,
            columns=top_k)

        def add_links(relateds):
            for top, others in zip(top_k, relateds):
                if not others:
                    continue

                for other in others:
                    if other in df.index:
                        df.loc[other, top] += 1
                    else:
                        not_in_top.append(other)

        add_links(broaders)
        add_links(narrowers)
        add_links(relateds)
        print(df)
        return df, df.sum(axis = 0)


    def graph_clusters(self,
                       top_idx: list[np.int64]):
        """
        Return subtrees of related UATs
        """
        paths = []
        for idx in top_idx:
            # Find path to root
            uat_info = UatUtils.get_uat(idx)
            print(uat_info)
            label = uat_info[1] # uat_labels[str(idx)][1]
            path = [(int(idx), label)]
            broaders = uat_info[2] # broader
            while broaders:
                broader = broaders[0]
                label = uat_labels[str(broader)][1]
                path.append((broader, label))
                broaders = uat_labels[str(broader)][2]
            print(path)
            paths.append(path)
        counts = Counter([p for path in paths for p in path ])
        print(counts)


    ### view functions ###
    def plot_top_k_curve(self,
                         top_k: int,
                         scores: np.array):
        """
        Plot curve (x = k, y = score(k))
        """
        # TODO
        top_idx = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_idx]

        x = range(1, top_k + 1)
        y = top_scores
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, marker="o")
        plt.xlabel("k")
        plt.ylabel("score(k)")
        plt.title("Top-k curve")
        plt.grid(True)

        plt.show()


    def get_top_idx(self,
                    top_k: int,
                    scores: np.array):
        """
        Returns top_k keywords idx
        """
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return top_idx


    def get_top_k_keywords(self,
                           top_k: int,
                           scores: np.array):
        """
        Return and print top_k keywords
        """
        top_idx = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_idx]
        broaders = []
        narrowers = []
        relateds = []
        labels = []
        for rank, (top_id, top_score) in enumerate(zip(top_idx, top_scores)):
            print(f"Top {rank + 1}: ", top_id, top_score)
            uat_info = UatUtils.get_uat(top_id)# uat_labels[str(int(top_id))]
            print("UAT info:", uat_info)
            uat_uri = uat_info[0]
            uat_label = uat_info[1]
            uat_broader = uat_info[2]
            uat_narrower = uat_info[3]
            uat_related = uat_info[4]
            print(uat_uri, uat_label)#, uat_broader, uat_narrower, uat_related)
            broaders.append(uat_broader)
            narrowers.append(uat_narrower)
            relateds.append(uat_related)
            labels.append(uat_label)


        # Count inner-relations between top-k entities
        counts, counts_sum = self.count_relations(top_idx,
                                                  broaders,
                                                  narrowers,
                                                  [], # relateds
                                                  )

        counts = counts.sum(axis = 0)
        print(counts)
        return labels


def main():

    for doi, title, abstract, papers_keywords, doc_str in Reader():
        doc = DocumentProcesser(doi, title, abstract, papers_keywords, doc_str)
        doc.process()
        continue
        print("----------------------------------------------")
        print(doi, doc_str)
        query_emb = model.encode(doc_str)
        query_emb = query_emb / np.linalg.norm(query_emb)
        scores = uat_embeddings @ query_emb
        get_top_k(30, scores, uat_labels)
        plot_top_k_curve(30, scores)

        for sentence in sentencize(doc_str):
            query_emb = model.encode(sentence)
            query_emb = query_emb / np.linalg.norm(query_emb)
            scores_sentence = uat_embeddings @ query_emb

        """
        best_id = np.argmax(scores)
        best_score = scores[best_id]
        print(best_id, best_score)
        print("UAT LABEL:", uat_labels[str(int(best_id))])
        """

if __name__ == "__main__":
    main()
