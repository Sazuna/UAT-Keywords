"""
Connect to ADS API to collect papers that have one keyword
under the Heliophysics category in the UATs' hierarchy.
"""
import json
import os
import regex
import requests
import torch
import atexit
from tqdm import tqdm
from typing import List, Tuple, Dict
from urllib.parse import urlencode
from rdflib import Graph, SKOS, RDF
from huggingface_hub import HfApi
from datasets import Dataset, DatasetInfo
from src.utils.config import CORPUS_DIR, UATS_JSON, ADS_HELIO_CORPUS_DIR, ADS_CORPUS_DIR, DATA_DIR, UATS_RDF_V6
from src.label_match.label_match import label_match
from src.corpus import uat_to_corpus


# Make a query with one keyword
ADS_API_TOKEN = os.environ.get("ADS_API_TOKEN")

# Categories of interest
HP_CATEGORIES = {"2373": "Heliophysics",
                 "1529": "Solar system astronomy"}

ALL_CATEGORIES = {"104":  "Astrophysical processes",
                  "343":  "Cosmology",
                  "486":  "Exoplanet astronomy",
                  "563":  "Galactic and extragalactic astronomy",
                  "2373": "Heliophysics",
                  "739":  "High energy astrophysics",
                  "804":  "Interdisciplinary astronomy",
                  "847":  "Interstellar medium",
                  "1145": "Observational astronomy",
                  "1529": "Solar system astronomy",
                  "1583": "Stellar astronomy"
                 }

UAT_NAMESPACE = "http://astrothesaurus.org/uat/"

# Load UATs
UAT_ONTOLOGY = "UAT_v6.0.0.rdf"
uat_to_corpus.main(CORPUS_DIR / UAT_ONTOLOGY)

with open (UATS_JSON, "r") as file:
    uat_labels = json.load(file)

# Build the labels2uat dicts
uats_by_pref_labels = dict()
uats_by_alt_labels = dict()
for uat_uri, uat_data in uat_labels.items():
    uat_idx = uat_uri.split('/')[-1] # str
    pref_label = uat_data[str(SKOS.prefLabel)][0]
    pref_label = pref_label.lower()
    uats_by_pref_labels[pref_label] = uat_idx
    if pref_label.endswith('s'):
        uats_by_alt_labels[pref_label[:-1]] = uat_idx
    alt_labels = uat_data.get(str(SKOS.altLabel), [])
    for alt_label in alt_labels:
        uats_by_alt_labels[alt_label] = uat_idx
        alt_label = alt_label.lower()
        if alt_label.endswith('s'):
            uats_by_alt_labels[alt_label[:-1]] = uat_idx


base_ads_query = lambda x, y: "keyword:\"{}\" bibstem:\"{}\"".format("\",\"".join(x), y)

def get_uats_under(uat_uri: str):
    """
    Find the list of UATs that are under Heliophysics concept

    Args:
        uat_uri: the URI of the UAT (namespace + index)
    """
    uat_info = uat_labels[uat_uri]
    narrowers = uat_info.get(str(SKOS.narrower), [])
    yield from narrowers
    for narrower in narrowers:
        yield from get_uats_under(narrower)


def get_uat_label(uat_uri: str):
    return uat_labels[uat_uri].get(str(SKOS.prefLabel), "")


def get_uat_alt_labels(uat_uri: str) -> List[str]:
    return uat_labels[uat_uri].get(str(SKOS.altLabel), [])


def get_uat_idx(uat_label: str) -> str:
    """
    Return the UAT index (str) if the label is in pref_label, else lookup in the alt_label dictionary.
    """
    uat_label = uat_label.lower()
    return uats_by_pref_labels.get(uat_label, uats_by_alt_labels.get(uat_label, None))


def make_query(uat_idx, uat_label, rows: int = 1000):
    """
    Use uat_label to prevent getting things like NGC 659
    """
    query = {"q": base_ads_query([str(uat_idx), uat_label]),
             "fl": "title, bibcode, abstract, keyword",
             "rows": rows}
    return urlencode(query)


def get_results(uat_idx, uat_label, corpus_dir):
    response = requests.get("https://api.adsabs.harvard.edu/v1/search/query?{}".format(make_query(uat_idx, uat_label)), \
                        headers={'Authorization': 'Bearer ' + ADS_API_TOKEN})
    try:
        response = response.json()["response"]
    except requests.exceptions.JSONDecodeError:
        print(response)
        print("https://api.adsabs.harvard.edu/v1/search/query?{}".format(make_query(uat_idx, uat_label)))
        exit()
    numFound = response["numFound"]
    numFoundExact = response["numFoundExact"]
    if not numFoundExact:
        raise ValueError("numFoundExact not found:", uat_idx, uat_label, numFound)
    docs = response["docs"]
    for doc in docs:
        bibcode = doc["bibcode"]
        filename = corpus_dir / f"{bibcode}.json"
        if filename.exists():
            continue
        title = doc.get("title", "")
        keywords = doc.get("keyword", [])
        abstract = doc.get("abstract", "")
        if not abstract or not title or not bibcode or not keywords:
            continue
        abstract = regex.sub(r"\<.*?\>", "", abstract)
        if uat_idx not in keywords:
            continue
        if uat_label not in keywords:
            continue
        with open(filename, "w") as file:
            json.dump({"title": title,
                       "bibcode": bibcode,
                       "abstract": abstract,
                       "keywords": keywords},
                       file,
                       indent = 2)


def make_query_keyword(uat_label, bibstem, rows: int = 1000):
    """
    Make a query that is only based on the UAT label.
    """
    query = {"q": base_ads_query([uat_label], bibstem),
             "fl": "title, bibcode, abstract, keyword",
             "rows": rows}
    return urlencode(query)

def get_results_keyword(uat_label,
                        corpus_dir,
                        bibstems,
                        save_on_disk: bool = False,
                        upload_to_hf: bool = False,
                        bibcodes_history: List[str] = [],
                        get_only_verified_uat: bool = False):

    docs = []
    for bibstem in bibstems:
        if get_only_verified_uat:
            query = make_query(uat_idx,
                               uat_label)
        else:
            query = make_query_keyword(uat_label, bibstem)
        response = requests.get("https://api.adsabs.harvard.edu/v1/search/query?{}".format(query), \
                            headers={'Authorization': 'Bearer ' + ADS_API_TOKEN})
        try:
            response = response.json()["response"]
        except requests.exceptions.JSONDecodeError:
            print(response)
            print("https://api.adsabs.harvard.edu/v1/search/query?{}".format(make_query_keyword(uat_label, bibstem)))
            exit()
        numFound = response["numFound"]
        numFoundExact = response["numFoundExact"]
        if not numFoundExact:
            raise ValueError("numFoundExact not found:", uat_label, numFound)
        docs.extend(response["docs"])

    # HF corpus
    bibcodes   = []
    texts      = []
    uats_uri   = [] # verified
    uats_label = [] # verified
    uats_uri_extended   = [] # extended by label match
    uats_label_extended = [] # extended by label match
    all_uats_uri   = [] # all UAT uris (verified + extended)
    all_uats_label = [] # all UAT labels (verified + extended)

    crash_message = ""
    for doc in docs:
        bibcode = doc["bibcode"]
        if bibcode in bibcodes_history:
            continue
        filename = corpus_dir / f"{bibcode}.json"
        if filename.exists():
            continue
        title = doc.get("title", [None])[0]
        keywords = doc.get("keyword", "")
        abstract = doc.get("abstract", "")
        if not abstract or not title or not bibcode or not keywords:
            continue

        # abstract = regex.sub(r"\<.*?\>", "", abstract)

        # GET UAT idx from keywords
        for keyword in keywords.copy():
            uat_idx = get_uat_idx(keyword)
            if uat_idx and uat_idx not in keywords:
                keywords.append(uat_idx)

        if save_on_disk and not crash_message:
            try:
                with open(filename, "w") as file:
                    json.dump({"title": title,
                            "bibcode": bibcode,
                            "abstract": abstract,
                            "keywords": keywords},
                            file,
                            indent = 2)
            except Exception as exc:
                crash_message = exc

        if upload_to_hf:
            uat_uri_filtered = list()
            uat_label_filtered = list()
            for keyword in keywords:
                uat_idx = get_uat_idx(keyword)
                if uat_idx:
                    uat_uri = UAT_NAMESPACE + uat_idx
                    if uat_uri not in uat_uri_filtered:
                        uat_uri_filtered.append(uat_uri)
                        uat_label_filtered.append(get_uat_label(uat_uri))
            text = title + " " + abstract
            texts.append(text)
            bibcodes.append(bibcode)
            uats_uri.append(uat_uri_filtered)
            uats_label.append(uat_label_filtered)

            # extend URIs with labels in the text
            uat_uri_extended = label_match(text)
            uat_uri_extended = sorted(set(uat_uri_extended) - set(uat_uri_filtered), key=lambda x: int(x.split('/')[-1])) # already in author's UAT
            uat_label_extended = [get_uat_label(uri) for uri in uat_uri_extended]

            uats_uri_extended.append(uat_uri_extended)
            uats_label_extended.append(uat_label_extended)

            all_uats_uri.append(uat_uri_filtered + uat_uri_extended)
            all_uats_label.append(uat_label_filtered + uat_label_extended)
    if crash_message:
        print("[Error] Error during save corpus to disk:", crash_message)

    return bibcodes, texts, uats_uri, uats_label, all_uats_uri, all_uats_label 


def upload_to_huggingface(
        bibcodes,
        texts,
        uats_uri,
        uats_label,
        uats_uri_extended,
        uats_label_extended,
        hf_dataset: str = "UAT_keywords"):
    def load_nodes(uat_ontology_path) -> Tuple[Dict, Dict]:
        rdf = Graph()
        rdf.parse(uat_ontology_path, format="xml")

        # 1. Collect all nodes (subjects or objects of a SKOS relation)
        nodes_set = set()
        for s, _, _ in rdf.triples((None, RDF.type, SKOS.Concept)):
            s_str = str(s)
            nodes_set.add(s_str)

        # 2. Node indexation
        sorted_nodes = sorted(nodes_set, key = lambda x: int(x.split('/')[-1]))
        node2idx = {n: i for i, n in enumerate(sorted_nodes)}
        idx2node = {i: n for n, i in node2idx.items()}
        return node2idx, idx2node 
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
                raise ValueError # TODO understand why this occurs
            for uri in ann_list:
                if uri in node2idx:
                    multihot[i, node2idx[uri]] = 1.0
                    # TODO label smoothing (https://www.mdpi.com/2079-9292/13/15/2944)
                else:
                    print(f"[Warning] Unknown URI in node2idx : {uri}")
        return multihot
    node2idx, idx2node = load_nodes(UATS_RDF_V6)
    multihot = build_multihot(annotations=uats_uri,
                              node2idx=node2idx)
    all_uats_uri = uats_uri + uats_uri_extended
    multihot_extended = build_multihot(annotations=all_uats_uri,
                                       node2idx=node2idx)

    data = {
        "bibcode": bibcodes,
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
    HF_USERNAME = os.environ.get("HF_USERNAME")
    HF_TOKEN = os.environ.get("HF_TOKEN")
    api = HfApi()
    api.upload_file(
        path_or_fileobj="label_mapping.json",
        path_in_repo="label_mapping.json",
        repo_id=f"{HF_USERNAME}/{hf_dataset}",
        repo_type="dataset",
        token=f"{HF_TOKEN}"
    )
    dataset = Dataset.from_dict(data, info=info)

    dataset.push_to_hub(
        f"{HF_USERNAME}/{hf_dataset}",
        token=f"{HF_TOKEN}"
    )

all_bibcodes, all_texts, all_uat_uris, all_uat_labels, all_uat_uris_extended, all_uat_labels_extended = [], [], [], [], [], []
uats_done = []

def main(categories: str = HP_CATEGORIES,
         corpus_dir: str = ADS_HELIO_CORPUS_DIR,
         bibstems: list[str] = ["AAS", "AJ", "ApJ", "PASP", "pds", "PhDT", "PSJ", "RNAAS", "SSRv"],
         upload_to_hf: bool = False,
         save_on_disk: bool = True,
         get_only_verified_uat: bool = False,
         hf_dataset: str = "UAT_keywords",
         from_cache: bool = False):
    """
    Args:
        get_only_verified_uat: if True, it will look for papers with both the label and the index of UAT in their keywords.
    """
    def save_cache():
        global uats_done, all_bibcodes, all_texts, all_uat_uris, all_uat_labels, all_uat_uris_extended, all_uat_labels_extended
        data = {"uats_done": uats_done,
                "all_bibcodes": all_bibcodes,
                "all_texts": all_texts,
                "all_uat_uris": all_uat_uris,
                "all_uat_labels": all_uat_labels,
                "all_uat_uri_extended": all_uat_uris_extended,
                "all_uat_labels_extended": all_uat_labels_extended}
        with open(DATA_DIR / "history_make_corpus.json", "w") as file:
            json.dump(data, file)

    def load_cache():
        history_file = DATA_DIR / "history_make_corpus.json"
        if os.path.exists(history_file):
            with open(history_file, "r") as file:
                data = json.load(file)
                return data["uats_done"], data["all_bibcodes"], data["all_texts"], data["all_uat_uris"], data["all_uat_labels"], data["all_uat_uri_extended"], data["all_uat_labels_extended"]
        return [], [], [], [], [], [], []

    def download(categories, corpus_dir, bibstems, upload_to_hf, save_on_disk, from_cache: bool = True):
        corpus_dir.mkdir(parents=True, exist_ok=True)
        global uats_done, all_bibcodes, all_texts, all_uat_uris, all_uat_labels, all_uat_uris_extended, all_uat_labels_extended

        if from_cache:
            uats_done, all_bibcodes, all_texts, all_uat_uris, all_uat_labels, all_uat_uris_extended, all_uat_labels_extended = load_cache()
            atexit.register(save_cache)
        for category in tqdm(categories.keys()):
            uats = get_uats_under(UAT_NAMESPACE + category)
            uats = sorted(set(uats))
            for uat in uats:
                if uat in uats_done:
                    continue
                uat_label = get_uat_label(uat)
                if uat_label:
                    uat_label = uat_label[0]
                    print("Getting papers for:", uat, uat_label)
                    # uat = uat.split("/")[-1]
                    # get_results(uat, uat_label, corpus_dir)
                    res = get_results_keyword(
                        uat_label,
                        corpus_dir,
                        bibstems=bibstems,
                        save_on_disk=save_on_disk,
                        upload_to_hf=upload_to_hf,
                        bibcodes_history=all_bibcodes,
                        get_only_verified_uat=get_only_verified_uat
                    )
                    bibcodes, texts, uat_uris, uat_labels, uat_uris_extended, uat_labels_extended = res
                    all_bibcodes.extend(bibcodes)
                    all_texts.extend(texts)
                    all_uat_uris.extend(uat_uris)
                    all_uat_labels.extend(uat_labels)
                    all_uat_uris_extended.extend(uat_uris_extended)
                    all_uat_labels_extended.extend(uat_labels_extended)  
                uat_alt_labels = get_uat_alt_labels(uat)
                for uat_alt_label in uat_alt_labels:
                    res = get_results_keyword(
                        uat_alt_label,
                        corpus_dir,
                        bibstems,
                        save_on_disk=save_on_disk,
                        upload_to_hf=upload_to_hf,
                        bibcodes_history = all_bibcodes,
                        get_only_verified_uat=get_only_verified_uat
                    )
                    bibcodes, texts, uat_uris, uat_labels, uat_uris_extended, uat_labels_extended = res
                    all_bibcodes.extend(bibcodes)
                    all_texts.extend(texts)
                    all_uat_uris.extend(uat_uris)
                    all_uat_labels.extend(uat_labels)
                    all_uat_uris_extended.extend(uat_uris_extended)
                    all_uat_labels_extended.extend(uat_labels_extended)
                uats_done.append(uat)
        if upload_to_hf:
            upload_to_huggingface(
                bibcodes=all_bibcodes,
                texts=all_texts,
                uats_uri=all_uat_uris,
                uats_label=all_uat_labels,
                uats_uri_extended=all_uat_uris_extended,
                uats_label_extended=all_uat_labels_extended,
                hf_dataset=hf_dataset)
            
    # download(HP_CATEGORIES, ADS_HELIO_CORPUS_DIR)
    # download(ALL_CATEGORIES, ADS_CORPUS_DIR)
    download(categories, #HP_CATEGORIES,
             corpus_dir, #ADS_HELIO_CORPUS_DIR,
             bibstems,
             upload_to_hf=upload_to_hf,
             save_on_disk=save_on_disk,
             from_cache=from_cache)


if __name__ == "__main__":
    main(categories=ALL_CATEGORIES,
         corpus_dir=ADS_CORPUS_DIR,
         bibstems = ["AAS", "AJ", "ApJ", "PASP", "pds", "PhDT", "PSJ", "RNAAS", "SSRv"],
         upload_to_hf=True,
         save_on_disk=False,
         hf_dataset="UAT_keywords_large",
         get_only_verified_uat=False,
         from_cache=True)
