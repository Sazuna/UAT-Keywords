
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import MultiLabelBinarizer


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