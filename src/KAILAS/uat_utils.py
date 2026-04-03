from src.utils.config import ADS_HELIO_CORPUS_DIR, UATS_JSON, CORPUS_DIR
from src.corpus import uat_to_corpus
from rdflib import SKOS
import json

UAT_NAMESPACE = "http://astrothesaurus.org/uat/"


if not UATS_JSON.exists():
    uat_to_corpus.main(CORPUS_DIR / "UAT_v6.0.0.rdf")
with open(UATS_JSON, "r") as file:
    uat_labels = json.load(file)

def get_uat_label(uat_uri: int):
    uat_uri = UAT_NAMESPACE + str(uat_uri)
    return uat_labels[uat_uri].get(str(SKOS.prefLabel), "")[0]

def get_uat_broader(uat_uri: int) -> list[str]:
    uat_uri = UAT_NAMESPACE + str(uat_uri)
    return [l.split('/')[-1] for l in uat_labels[uat_uri].get(str(SKOS.broader), "")]

def get_uat_narrower(uat_uri: int) -> list[str]:
    uat_uri = UAT_NAMESPACE + str(uat_uri)
    return [l.split('/')[-1] for l in uat_labels[uat_uri].get(str(SKOS.narrower), "")]
