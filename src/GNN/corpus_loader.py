import json
import os
from config import TEST_CORPUS_FILE, ADS_HELIO_CORPUS_DIR, ADS_CORPUS_DIR
from typing import Iterable
from pathlib import Path
from collections import defaultdict

from datasets import load_dataset
dataset = load_dataset("adsabs/SciX_UAT_Keywords")


def get_kailas_training_bibcodes():
    # https://huggingface.co/datasets/adsabs/SciX_UAT_keywords
    return dataset["train"]["bibcode"]

kailas_bibcodes = get_kailas_training_bibcodes()
uat_namespace = "http://astrothesaurus.org/uat/"
class Reader():

    class Document():

        def __init__(self, bibcode, title, journal, abstract, keywords: list[str], uats: list[str]):
            self.bibcode = bibcode
            self.title = title
            self.journal = journal
            self.abstract = abstract
            self.keywords = keywords
            if not uats:
                # Extract UATs from keywords
                uats = {f"{uat_namespace}{keyword}" for keyword in keywords if keyword.isnumeric()}
            else:
                for i, uat in enumerate(uats):
                    if not type(uat) == str or not uat.startswith(uat_namespace):
                        uats[i] = f"{uat_namespace}{uat}"
            self.uats = uats


        @property
        def text(self):
            return self.title + '. ' + self.abstract + '. ' + ', '.join(self.keywords)

        def sentencize(self):
            return [t.strip() for t in self.text.split('.') if t.strip()]


    def read_pre9forADS(self) -> Iterable[tuple]:
        """
        Load our unpublished preprint corpus. Yield tuples (doc_str, list_of_uats).

        Args:
            ignore_kailas: ignore documents that are in the KAILAS training set.
        """
        with open(TEST_CORPUS_FILE, "r") as file:
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
                elif line.startswith("%U"):
                    state = "uats"
                    prefix = "%U"
                elif line.startswith("%Z"):
                    state = "z..."
                    prefix = "%Z"

                if not state:
                    continue
                elif state == "DOI":
                    doi = line.removeprefix(prefix).strip()
                else:
                    doc[state] += line.removeprefix(prefix).strip() + ' '

            for bibcode, doc in all_docs.items():
                keywords = doc.get("keywords", "").strip()
                title = doc.get("title", "").strip()
                journal = doc.get("journal", "")
                abstract = doc.get("abstract", "").strip()
                uats = [u.strip() for u in doc.get("uats", "").split(',') if u.strip()]
                document = Reader.Document(doi, title, journal, abstract, keywords, uats)
                yield document.text, document.uats
                # yield bibcode, title, abstract, keywords, journal, uats


    def read_corpus(self,
                    ignore_kailas: bool = False,
                    corpus_folder: Path = ADS_HELIO_CORPUS_DIR) -> Iterable[tuple]:
        """
        Load corpus collected on ADS. Yield tuples (doc_str, list_of_uats).

        Args:
            ignore_kailas: ignore documents that are in the KAILAS training set.
        """
        total = 0
        for filename in sorted(os.listdir(corpus_folder)):
            with open(corpus_folder / filename, "r") as file:
                doc = json.load(file)
            abstract = doc["abstract"]
            keywords = doc["keywords"]
            title = doc["title"][0]
            bibcode = doc["bibcode"]
            if ignore_kailas and bibcode in kailas_bibcodes:
                ignored += 1
                continue
            total += 1
            document = Reader.Document(bibcode, title, None, abstract, keywords, None)
            yield document.text, document.uats
        print(f"Total of documents in the corpus: {total}")
        if ignore_kailas:
            print(f"Ignored documents that are in KAILAS training set: {ignored}")
