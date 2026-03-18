"""
Evaluate KAILAS fine-tuned on our corpus
"""
from typing import List
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
import corpus_loader
import config
import ontology_graph

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "./kailas-finetuned/checkpoint-21270"  # ← à adapter
TOKENIZER_NAME  = "adsabs/KAILAS"
THRESHOLD       = 0.62

# ── Ontologie ─────────────────────────────────────────────────────────────────
onto     = ontology_graph.OntologyGraph(config.CORPUS_DIR / "UAT_v6.0.0.rdf")
node2idx = onto.node2idx
idx2node = onto.idx2node
# Est-ce que l'index UAT correspond bien à la position dans node2idx ?
test_uri = "http://astrothesaurus.org/uat/1360"
print(f"node2idx[{test_uri}] = {node2idx[test_uri]}")
# Doit afficher : 1360
# Si ça affiche autre chose → vos labels sont décalés
exit()

# ── Tokenizer & modèle ────────────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model     = AutoModelForSequenceClassification.from_pretrained(CHECKPOINT_PATH)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Modèle chargé depuis : {CHECKPOINT_PATH}")
print(f"Device : {device}\n")

# ── Dataset ───────────────────────────────────────────────────────────────────
def build_multihot(annotations: List[List[str]]) -> torch.Tensor:
    N = len(node2idx)
    D = len(annotations)
    multihot = torch.zeros(D, N)
    for i, ann_list in enumerate(annotations):
        for uri in ann_list:
            if uri in node2idx:
                multihot[i, node2idx[uri]] = 1.0
            else:
                print(f"[Warning] Unknown URI in node2idx: {uri}")
    return multihot

def tokenize(batch):
    tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    tokens["labels"] = build_multihot(batch["labels"]).tolist()
    return tokens

validation_docs, validation_labels = [], []
reader = corpus_loader.Reader()
for document, annotation in reader.read_pre9forADS():
    validation_docs.append(document)
    validation_labels.append(annotation)

validation_dataset = Dataset.from_dict({"text": validation_docs, "labels": validation_labels})
validation_dataset = validation_dataset.map(tokenize, batched=True)
validation_dataset.set_format("torch")
print(f"{len(validation_dataset)} documents chargés\n")

# ── Inférence ─────────────────────────────────────────────────────────────────
print(f"{'='*60}")
print(f"Test sur {len(validation_dataset)} documents (threshold={THRESHOLD})")
print(f"{'='*60}\n")

all_probs = []

for i, sample in enumerate(validation_dataset):
    input_ids      = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    all_probs.append(probs)

    pred_idxs  = np.where(probs >= THRESHOLD)[0]
    pred_nodes = [idx2node[idx] for idx in pred_idxs]

    true_idxs  = np.where(sample["labels"].numpy() == 1.0)[0]
    true_nodes = [idx2node[idx] for idx in true_idxs]

    print(f"── Document {i} {'─'*40}")
    print(f"  Texte    : {validation_docs[i][:120]}...")
    print(f"  probs    : max={probs.max():.4f}  mean={probs.mean():.4f}")
    print(f"  Prédits  : {pred_nodes if pred_nodes else '(aucun)'}")
    print(f"  Attendus : {true_nodes}")
    print()

# ── Diagnostic global ─────────────────────────────────────────────────────────
all_probs = np.concatenate(all_probs)
print(f"{'='*60}")
print(f"Diagnostic global")
print(f"{'='*60}")
print(f"  probs mean  : {all_probs.mean():.4f}")
print(f"  probs max   : {all_probs.max():.4f}")
print(f"  probs > 0.1 : {(all_probs > 0.1).sum()}")
print(f"  probs > 0.3 : {(all_probs > 0.3).sum()}")
print(f"  probs > 0.5 : {(all_probs > 0.5).sum()}")