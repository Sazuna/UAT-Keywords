"""
Upload Dataset to HF
"""
import torch
import os
import json
import uat_to_corpus
import label_match
from huggingface_hub import HfApi
from datasets import load_dataset, Dataset, DatasetInfo
from config import ADS_HELIO_CORPUS_DIR, ADS_CORPUS_DIR, UATS_JSON, CORPUS_DIR
from pathlib import Path
from typing import Iterable, List, Dict, Tuple
from rdflib import Graph, SKOS, RDF
dataset = load_dataset("adsabs/SciX_UAT_Keywords")

def get_kailas_training_bibcodes():
    # https://huggingface.co/datasets/adsabs/SciX_UAT_keywords
    return dataset["train"]["bibcode"]

UAT_NAMESPACE = "http://astrothesaurus.org/uat/"

if not UATS_JSON.exists():
    uat_to_corpus.main(CORPUS_DIR / "UAT_v6.0.0.rdf")
with open(UATS_JSON, "r") as file:
    uat_dict = json.load(file)

def get_uat_label(uat_uri: str):
    if uat_uri not in uat_dict:
        return None
    return uat_dict[uat_uri].get(str(SKOS.prefLabel), "")[0]


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

def load_nodes(uat_ontology_path) -> Tuple[Dict, Dict]:
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
    sorted_nodes = sorted(nodes_set, key = lambda x: int(x.split('/')[-1]))
    node2idx = {n: i for i, n in enumerate(sorted_nodes)}
    idx2node = {i: n for n, i in node2idx.items()}
    return node2idx, idx2node

def main():
    HF_USERNAME = os.environ["HF_USERNAME"]
    HF_TOKEN = os.environ["HF_TOKEN"]

    texts      = []
    uats_uri   = [] # verified
    uats_label = [] # verified
    uats_uri_extended   = [] # extended by label match
    uats_label_extended = [] # extended by label match
    all_uats_uri   = [] # all UAT uris (verified + extended)
    all_uats_label = [] # all UAT labels (verified + extended)

    reader = Reader()
    for text, uat in reader.read_corpus(ADS_CORPUS_DIR):
        texts.append(text)
        uat_uri_filtered = []
        uat_label_filtered = []
        # uats.append(uat)
        for uri in uat:
            label = get_uat_label(uri)
            if label is None:
                continue
            uat_uri_filtered.append(uri)
            uat_label_filtered.append(label)
        # uats_label.append([get_uat_label(uri) for uri in uat])
        uats_uri.append(uat_uri_filtered)
        uats_label.append(uat_label_filtered)

        # extend URIs with labels in the text
        uat_uri_extended = label_match.label_match(text)
        uat_uri_extended = sorted(set(uat_uri_extended) - set(uat_uri_filtered), key=lambda x: int(x.split('/')[-1])) # already in author's UAT
        uat_label_extended = [get_uat_label(uri) for uri in uat_uri_extended]

        uats_uri_extended.append(uat_uri_extended)
        uats_label_extended.append(uat_label_extended)

        all_uats_uri.append(uat_uri_filtered + uat_uri_extended)
        all_uats_label.append(uat_label_filtered + uat_label_extended)

    node2idx, idx2node = load_nodes("../corpus/UAT_v6.0.0.rdf")
    multihot = build_multihot(annotations=uats_uri,
                              node2idx=node2idx)
    multihot_extended = build_multihot(annotations=all_uats_uri,
                                       node2idx=node2idx)

    data = {
        "text": texts,
        "uat_uri": uats_uri,
        "uat_label": uats_label,
        "multihot": [m.tolist() for m in multihot],
        "uat_uri_extended": uats_uri_extended,
        "uat_label_extended": uats_label_extended,
        "multihot_extended": multihot_extended,
    }
    info = DatasetInfo(
        description="""Training dataset for multi-label classification of astrophysics literature.
v6.0.0 of UATs was used to generate the multihot labels (https://github.com/astrothesaurus/UAT/releases/tag/v6.0.0).
This dataset was generated exclusively from SciX, filtering on the keywords field by combining the UAT's label and numerical value.
Last update: 2026-03-13"""
    )
    with open("label_mapping.json", "w") as f:
        json.dump({"idx2label": idx2node, "label2idx": node2idx}, f)

    # Push to HF files
    api = HfApi()
    api.upload_file(
        path_or_fileobj="label_mapping.json",
        path_in_repo="label_mapping.json",
        repo_id=f"{HF_USERNAME}/UAT_keywords",
        repo_type="dataset",
        token=f"{HF_TOKEN}"
    )
    dataset = Dataset.from_dict(data, info=info)

    dataset.push_to_hub(
        f"{HF_USERNAME}/UAT_keywords",
        token=f"{HF_TOKEN}"
    )


if __name__ == "__main__":
    main()
