
from typing import Iterable
from transformers import pipeline
classifier = pipeline("text-classification", model = "adsabs/KAILAS")

class Document():
    def __init__(self, bibcode, title, journal, abstract, keywords: list[str], uats: list[str]):
        self.bibcode = bibcode
        self.title = title
        self.journal = journal
        self.abstract = abstract
        self.keywords = keywords
        if not uats:
            # Extract UATs from keywords
            uats = {keyword for keyword in keywords if keyword.isnumeric()}
        self.uats = uats

    @property
    def text(self):
        return self.title + '. ' + self.abstract + '. ' + ', '.join(self.keywords)

    def sentencize(self):
        return [t.strip() for t in self.text.split('.') if t.strip()]


    def get_best_keywords(self,
                          top_k: int = None,
                          filter_on_keywords: list[str] = None) -> Iterable[list[str]]:
        """
        For each sentence, get top_k keywords.

        Args:
            top_k: get top_k best keywords
            filter_on_keywords: only allow keywords that are listed.
        """
        for sentence in self.sentencize():
            res = classifier(sentence)
            res_uats = []
            i = 0
            for uat in res:
                uat_label = uat["label"]
                if not filter_on_keywords or uat_label in filter_on_keywords:
                    res_uats.append(uat_label)
                    i += 1
                    if i == top_k:
                        break
            yield res_uats


    def select_keyword(self,
                       uats: list[str],
                       top_k: int = None):
        """
        """
