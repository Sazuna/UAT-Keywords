import json
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from rdflib import URIRef, SKOS
from src.utils.config import UATS_LABELS_JSON, UATS_JSON

# UATs informations
#with open(UATS_LABELS_JSON, "r") as file:
#    uat_labels = json.load(file)

with open(UATS_JSON, "r") as file:
    uat_json = json.load(file)

def get_uat_broaders(uri: URIRef) -> list[URIRef]:
    """
    Return None if the URI is not in the uat_json,
    else return a list of broaders (empty if uri was the top concept)
    """
    data = uat_json.get(uri, None)
    if data is None:
        return None
    return data.get(str(SKOS.broader), [])
    
def get_depth(uri: str, depth: int = 0) -> int:
    """
    Get the minimal depth of the UAT in the hierarchy.

    Args:
        uri: an UAT's uri (str representation of the URIRef)
    """
    broaders = get_uat_broaders(uri)
    if broaders is None:
        return -1 # Deprecated
    if not broaders:
        return depth
    all_broaders_depths = []
    for broader in broaders:
        all_broaders_depths.append(get_depth(broader, depth + 1))
    return min(all_broaders_depths)


"""
def get_uat(idx: int):
    idx = str(int(idx))
    return uat_labels[idx]


def get_uat_label(idx: int):
    return get_uat(idx)[1]


def get_uat_broaders(idx: int):
    return get_uat(idx)[2]
"""


def print_results(y_true, y_pred, corpus_name, total: int):

    mlb = MultiLabelBinarizer()
    
    # Ensure all elements are lists/sets of hashable items
    # Flatten if needed, or ensure consistent format
    all_labels = []
    for labels in y_true + y_pred:
        if isinstance(labels, list):
            # Convert list items to hashable if needed
            all_labels.append([str(label) if isinstance(label, list) else label 
                              for label in labels])
        else:
            all_labels.append(labels)
    all_labels = y_true + y_pred
    mlb.fit(all_labels)
    """
    y_true_ml = []
    y_pred_ml = []
    for t in y_true:
        y_true_ml.append(mlb.transform(t))
    for p in y_pred:
        y_pred_ml.append(mlb.transform(p))
    y_true, y_pred = y_true_ml, y_pred_ml
    """
    y_true = mlb.transform(y_true)
    y_pred = mlb.transform(y_pred)
    micro_precision = precision_score(y_true, y_pred, average="micro")
    macro_precision = precision_score(y_true, y_pred, average="macro")
    micro_recall = recall_score(y_true, y_pred, average="micro")
    macro_recall = recall_score(y_true, y_pred, average="macro")
    micro_f1 = f1_score(y_true, y_pred, average="micro")
    macro_f1 = f1_score(y_true, y_pred, average="macro")


    print(f"Test on {corpus_name} corpus:")
    print("Total papers:", total)
    print("micro precision:", micro_precision)
    print("micro recall:", micro_recall)
    print("micro F1:", micro_f1)
    print()
    print("macro precision:", macro_precision)
    print("macro recall:", macro_recall)
    print("macro F1:", macro_f1)