"""
Upload Dataset to HF
"""
import torch
import os
import json
from datasets import load_dataset, Dataset, DatasetInfo
from config import ADS_HELIO_CORPUS_DIR, ADS_CORPUS_DIR
from pathlib import Path
from typing import Iterable, List, Dict
from torch_geometric.data import Data
from rdflib import Graph, SKOS, RDF
dataset = load_dataset("adsabs/SciX_UAT_Keywords")

def get_kailas_training_bibcodes():
    # https://huggingface.co/datasets/adsabs/SciX_UAT_keywords
    return dataset["train"]["bibcode"]

kailas_bibcodes = get_kailas_training_bibcodes()
uat_namespace = "http://astrothesaurus.org/uat/"
SKOS_BROADER = SKOS.broader
SKOS_NARROWER = SKOS.narrower
SKOS_RELATED = SKOS.related

class Reader():

    class Document():

        def __init__(self, bibcode, title, journal, abstract, keywords: list[str], uats: list[str]):
            self.bibcode = bibcode
            self.title = title
            self.journal = journal
            self.abstract = abstract
            self.keywords = keywords
            self.uat_labels = []
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

        @property
        def text(self):
            res = self.title + '. ' + self.abstract
            if not self.has_uat_in_keywords:
                res += ', '.join(self.keywords)
            return res


    def read_corpus(self,
                    corpus_folder: Path = ADS_HELIO_CORPUS_DIR,
                    ignore_kailas: bool = False) -> Iterable[tuple]:
        """
        Load corpus collected on ADS. Yield tuples (doc_str, list_of_uats).

        Args:
            ignore_kailas: ignore documents that are in the KAILAS training set.
        """
        total = 0
        ignored = 0
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

def build_multihot(
    annotations: List[List[str]],
    node2idx: Dict[str, int],
) -> torch.Tensor:
    """
    Convert a list of annotations (URIs) into multi-hot matrix.

    Args:
        annotations : list of lists of URIs (one per document)
        node2idx    : mapping URI → index

    Returns:
        tensor [D, N]
    """
    N = len(node2idx)
    D = len(annotations)
    multihot = torch.zeros(D, N)
    for i, ann_list in enumerate(annotations):
        if len(ann_list) == 0:
            print("No annotation for document.")
        for uri in ann_list:
            if uri in node2idx:
                multihot[i, node2idx[uri]] = 1.0
                # TODO label smoothing (https://www.mdpi.com/2079-9292/13/15/2944)
            else:
                print(f"[Warning] Unknown URI in node2idx : {uri}")
    return multihot

def load_nodes(uat_ontology_path) -> Data:
    rdf = Graph()
    rdf.parse(uat_ontology_path, format="xml")

    # 1. Collect all nodes (subjects or objects of a SKOS relation)
    relations = [
        (SKOS_BROADER, "broader"),
        (SKOS_NARROWER, "narrower"),
        (SKOS_RELATED, "related"),
    ] # "self" for self-attention (node to self edges)

    nodes_set = set()

    for predicate, etype in relations:
        for s, _, o in rdf.triples((None, predicate, None)):
            s_str, o_str = str(s), str(o)
            nodes_set.update([s_str, o_str])

    for s, _, _ in rdf.triples((None, RDF.type, SKOS.Concept)):
        s_str = str(s)
        nodes_set.add(s_str)

    # 2. Node indexation
    sorted_nodes = sorted(nodes_set)
    node2idx = {n: i for i, n in enumerate(sorted_nodes)}
    idx2node = {i: n for n, i in node2idx.items()}
    return node2idx, idx2node

def main():
    HF_USERNAME = os.environ["HF_USERNAME"]
    HF_TOKEN = os.environ["HF_TOKEN"]

    texts      = []
    uats       = []
    uats_label = []

    reader = Reader()
    for text, uat in reader.read_corpus(ADS_CORPUS_DIR):
        texts.append(text)
        uats.append(uat)

    node2idx, idx2node = load_nodes("../corpus/UAT_v6.0.0.rdf")
    multihot = build_multihot(annotations=uats, node2idx=node2idx)

    data = {
        "text": texts,
        "uat_uri": uats,
        "uat_label": uats_label,
        "multihot": [m.tolist() for m in multihot]
    }
    info = DatasetInfo(
        description=json.dumps({
            "idx2label": idx2node,
            "label2idx": node2idx
        })
    )
    dataset = Dataset.from_dict(data, info=info)

    dataset.push_to_hub(
        f"{HF_USERNAME}/UAT_keywords",
        token=f"{HF_TOKEN}"
    )


if __name__ == "__main__":
    main()
