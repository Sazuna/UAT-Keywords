"""
Connect to ADS API to collect papers that have one keyword
under the Heliophysics category in the UATs' hierarchy.
"""
import json
import os
import regex
import requests
from urllib.parse import urlencode
from rdflib import SKOS
from config import UATS_JSON, ADS_HELIO_CORPUS_DIR, ADS_CORPUS_DIR


# Make a query with one keyword
ADS_API_TOKEN = os.environ.get("ADS_API_TOKEN")

# Categories of interest
HP_CATEGORIES = {"2373": "Heliophysics"}

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
                  "1583": "Stellar astronomy"}

UAT_NAMESPACE = "http://astrothesaurus.org/uat/"

with open (UATS_JSON, "r") as file:
    uat_labels = json.load(file)
base_ads_query = lambda x: "keyword:\"{}\"".format("\",\"".join(x))

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


def make_query(uat_idx, uat_label, rows: int = 1000):
    """
    Use uat_label to prevent getting things like NGC 659
    """
    print([str(uat_idx), uat_label])
    query = {"q": base_ads_query([str(uat_idx), uat_label]),
             "fl": "title, bibcode, abstract, keyword",
             "rows": rows}
    return urlencode(query)

def get_results(uat_idx, uat_label, corpus_dir):
    response = requests.get("https://api.adsabs.harvard.edu/v1/search/query?{}".format(make_query(uat_idx, uat_label)), \
                        headers={'Authorization': 'Bearer ' + ADS_API_TOKEN})
    response = response.json()["response"]
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
        title = doc["title"]
        keywords = doc["keyword"]
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


def main():
    def download(categories, corpus_dir):
        for category in categories.keys():
            uats = get_uats_under(UAT_NAMESPACE + category)
            uats = sorted(set(uats))
            for uat in uats:
                uat_label = get_uat_label(uat)
                if uat_label:
                    uat_label = uat_label[0]
                    print("Getting papers for:", uat, uat_label)
                    uat = uat.split("/")[-1]
                    get_results(uat, uat_label, corpus_dir)
    download(HP_CATEGORIES, ADS_HELIO_CORPUS_DIR)
    download(ALL_CATEGORIES, ADS_CORPUS_DIR)

if __name__ == "__main__":
    main()
