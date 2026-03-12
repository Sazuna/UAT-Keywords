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
from config import UATS_JSON, ADS_HELIO_CORPUS_DIR


# Make a query with one keyword
ADS_API_TOKEN = os.environ.get("ADS_API_TOKEN")

# Categories of interest
CATEGORIES = {"2373": "Heliophysics"}

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

def get_results(uat_idx, uat_label):
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
        filename = ADS_HELIO_CORPUS_DIR / f"{bibcode}.json"
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
    for category in CATEGORIES.keys():
        uats_heliophysics = get_uats_under(UAT_NAMESPACE + category)
        uats_heliophysics = sorted(set(uats_heliophysics))
        for uat_heliophysics in uats_heliophysics:
            uat_label = get_uat_label(uat_heliophysics)
            if uat_label:
                uat_label = uat_label[0]
                print("Getting papers for:", uat_heliophysics, uat_label)
                uat_heliophysics = uat_heliophysics.split("/")[-1]
                get_results(uat_heliophysics, uat_label)

if __name__ == "__main__":
    main()
