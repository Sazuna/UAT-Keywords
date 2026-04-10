"""
Find UATs by naive textual match
"""
from typing import Set
from rdflib import SKOS
from src.utils.config import UATS_JSON, CORPUS_DIR, ADS_HELIO_CORPUS_DIR
from src.corpus import uat_to_corpus
import json
import regex

if not UATS_JSON.exists():
    uat_to_corpus.main(CORPUS_DIR / "UAT_v6.0.0.rdf")
with open(UATS_JSON, "r") as file:
    uat_dict = json.load(file)

label2uat = dict()
for uat, data in uat_dict.items():
    label = data[str(SKOS.prefLabel)][0].strip()
    if label:
        label2uat[label.lower()] = uat
        if label.endswith("s"):
            label2uat[label[:-1].lower()] = uat
    alt_labels = data.get(str(SKOS.altLabel), [])
    for alt_label in alt_labels:
        alt_label = alt_label.strip()
        if alt_label:
            label2uat[alt_label.lower()] = uat
            if alt_label.endswith("s"):
                label2uat[alt_label[:-1].lower()] = uat

# Sort by length to make the selection greedy
sorted_labels = sorted(label2uat.keys(), key=len, reverse=True)

# This category matches with too many things (false positive) so we remove it
sorted_labels.remove("of star")
sorted_labels.remove("of stars")

# Escape special characters
escaped_labels = [regex.escape(label) for label in sorted_labels]

expression = r'(\b' + r'\b|\b'.join(escaped_labels) + r'\b)'
expression = regex.compile(expression, flags=regex.IGNORECASE)

def label_match(text: str) -> Set[str]:
    """
    Return the URIs of the UATs with the longest string matches.
    """
    matches = regex.findall(expression, text)
    if not matches:
        return []
    matches_by_longest = []
    for m in matches:
        added = False
        for i, l in enumerate(matches_by_longest):
            m_lower = m.lower()
            l_lower = l.lower()
            if m_lower.startswith(l_lower):
                added = True
                matches_by_longest[i] = m # longer
                break
            elif l_lower.startswith(m_lower):
                added = True
                break # shorter than a label already in
        if not added:
            matches_by_longest.append(m)
    uats = {label2uat[label.lower()] for label in matches_by_longest}
    return uats

def main():
    """
    Try to execute label_match on our test corpus
    """
    from src.utils import corpus_loader
    from src.utils.util import print_results
    reader = corpus_loader.Reader()
    y_pred = []
    y_true = []
    total = 0
    for document in reader.read_pre9forADS():
        predicted = label_match(document.text)
        print("text:", document.text)
        print("pred:", predicted)
        print("true:", document.uats)
        y_pred.append(predicted)
        y_true.append(document.uats)
        total += 1
    print_results(y_true, y_pred, "preprint", total)
    y_pred = []
    y_true = []
    total = 0
    for document in reader.read_corpus(False, ADS_HELIO_CORPUS_DIR):
        predicted = label_match(document.text)
        print("text:", document.text)
        print("pred:", predicted)
        print("true:", document.uats)
        y_pred.append(predicted)
        y_true.append(document.uats)
        total += 1
    print_results(y_true, y_pred, "ADS HELIO", total)

if __name__ == "__main__":
    main()
