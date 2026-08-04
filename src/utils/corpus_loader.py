import json
import os
import bibtexparser
from src.utils.config import (TEST_CORPUS_FILE,
    ADS_HELIO_CORPUS_DIR,
    ADS_CORPUS_DIR,
    UATS_JSON,
    BIBTEX_PATH)
from tqdm import tqdm
from typing import Iterable
from pathlib import Path
from collections import defaultdict
from datasets import Dataset
from rdflib import SKOS

from datasets import load_dataset
dataset = load_dataset("adsabs/SciX_UAT_Keywords")

with open(UATS_JSON, "r") as file:
    uat_data = json.load(file)


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
            self.has_uat_in_keywords = False
            if not uats:
                # Extract UATs from keywords
                uats = {f"{uat_namespace}{keyword}" for keyword in keywords if keyword.isnumeric()}
                if uats:
                    self.has_uat_in_keywords = True
            else:
                for i, uat in enumerate(uats):
                    if not type(uat) == str or not uat.startswith(uat_namespace):
                        uats[i] = f"{uat_namespace}{uat}"
            self.uats = uats
            uats_labels = [uat_data.get(uat, {SKOS.prefLabel: ["[DEPRECATED]"]}).get(str(SKOS.prefLabel), ["[NO_LABEL]"])[0] for uat in uats]
            if "[NO_LABEL]" in uats_labels:
                uats_labels.remove("[NO_LABEL]")
            self.uats_labels = uats_labels


        @property
        def text(self):
            res = self.title + '. ' + self.abstract
            if not self.has_uat_in_keywords:
                res += ', '.join(self.keywords)
            return res

        def sentencize(self):
            return [t.strip() for t in self.text.split('.') if t.strip()]


        def sentencize_with_overlap(self) -> list[tuple[str]]:
            """
            Cut by sentences two by two to extract sub-categories.
            FIXME the first and last sentences are under represented
            """
            sentences = self.sentencize()
            return zip(sentences[:-1], sentences[1:])


    def read_pre9forADS(self) -> Iterable[Document]:
        """
        Load our unpublished preprint corpus. Yield tuples (doc_str, list_of_uats).

        Args:
            ignore_kailas: ignore documents that are in the KAILAS training set.
        """
        with open(TEST_CORPUS_FILE, "r") as file:
            lines = file.readlines()
            all_docs = dict()
            doc = defaultdict(str)
            state = None
            prefix = ""
            doi = None
            for line in lines:
                if not line.strip():
                    # New line
                    if doi and doc:
                        all_docs[doi] = doc
                    doc = defaultdict(str)
                    state = None
                elif line.startswith("%R"): # reference
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
                elif state:
                    doc[state] += line.removeprefix(prefix).strip() + ' '
            if not doi in all_docs and doc:
                all_docs[doi] = doc

            for doi, doc in all_docs.items():
                keywords = doc.get("keywords", "").strip()
                title = doc.get("title", "").strip()
                journal = doc.get("journal", "")
                abstract = doc.get("abstract", "").strip()
                uats = [u.strip() for u in doc.get("uats", "").split(',') if u.strip()]
                document = Reader.Document(doi, title, journal, abstract, keywords, uats)
                yield document
                # yield bibcode, title, abstract, keywords, journal, uats


    def read_corpus(self,
                    ignore_kailas: bool = False,
                    corpus_folder: Path = ADS_HELIO_CORPUS_DIR) -> Iterable[Document]:
        """
        Load corpus collected on ADS. Yield tuples (doc_str, list_of_uats).

        Args:
            ignore_kailas: ignore documents that are in the KAILAS training set.
        """
        total = 0
        for filename in tqdm(sorted(os.listdir(corpus_folder))):
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
            yield document
        print(f"Total of documents in the corpus: {total}")
        if ignore_kailas:
            print(f"Ignored documents that are in KAILAS training set: {ignored}")
        if total == 0:
            raise FileNotFoundError(f"{corpus_folder} is empty.")


    def get_hf_corpus(self,
                      corpus_path: str = "Sazuna/UAT_keywords",
                      split: str = "train") -> Dataset:
        """
        Quicker than read_corpus.
        """
        dataset = load_dataset(corpus_path, split=split)
        return dataset


    def read_bibtex(self,
                    path: Path = BIBTEX_PATH) -> Iterable[Document]:
        """
        Read a bibtex (.bib) document and yields
        documents.
        """
        with open(path, "r") as file:
            # text = file.read()
            # print(text)
            library = bibtexparser.load(file)
        for paper in library.entries:
            bibcode = paper.get("bibcode", paper.get("doi", paper.get("ID", "")))
            title = paper.get("title", "")
            journal = paper.get("booktitle", "")
            abstract = paper.get("abstract", "")
            keywords = paper.get("keywords", [])
            uats = paper.get("uat", [])
            document = Reader.Document(bibcode,
                                       title,
                                       journal,
                                       abstract,
                                       keywords,
                                       uats)
            yield document
