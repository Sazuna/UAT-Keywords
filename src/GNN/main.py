"""
main.py

Load the UATs ontology, vectorize documents, train GNN and infer on our pre-print corpus.
"""
import torch
import pathlib
import datetime
import os
from src.utils.config import BERT_MODEL, ADS_HELIO_CORPUS_DIR, ADS_CORPUS_DIR, UATS_RDF_V6
from src.GNN.ontology_graph import OntologyGraph
from src.utils import corpus_loader
from src.utils.util import print_results
from src.GNN.model import GNNOntologyClassifier
from src.GNN.train import (
    DocumentOntologyDataset,
    build_multihot,
    train,
    predict,
)

import random
import numpy as np
import torch

# Reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Reproducibility with concurrent streams with cuBLAS
torch.use_deterministic_algorithms(True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8" # increase library footprint in GPU by 24MiB but does not affect the performance


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

ONTO_PATH       = UATS_RDF_V6 # Input graph
CORPUS_PATH     = ADS_HELIO_CORPUS_DIR  # can use ADS_HELIOPHYSICS_CORPUS_DIR for HP-only
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
RUNS            = pathlib.Path("runs")
CHECKPOINT_PATH = "/scratch2/lfretel/GNN_best_model.pt"

# Hyperparameters
RGCN_HIDDEN   = 128
RGCN_OUT      = 56
NUM_LAYERS    = 2     # smoothing of the representations
DROPOUT       = 0.2 #0.3
EPOCHS        = 10
BATCH_SIZE    = 32
LR            = 1e-2
TOP_K         = 10

# Results:
# RGCN_HIDDEN => 128, RGCN_OUT => 56, LR => 1e-3: 0.5 after 3 ep


# ─────────────────────────────────────────────────────────────────────
# 1. LOAD ONTOLOGY
# ─────────────────────────────────────────────────────────────────────

print("=== Loading ontology ===")
onto = OntologyGraph(uat_ontology_path=ONTO_PATH, bert_model_name=BERT_MODEL)
graph = onto.load()

print(f"N nodes (vertex) : {graph.num_nodes}")
print(f"N edges : {graph.edge_index.shape[1]}")
print(f"Distribution of edge types : {torch.bincount(graph.edge_type)}")


# ─────────────────────────────────────────────────────────────────────
# 2. ANNOTATED TRAINING CORPUS
# ─────────────────────────────────────────────────────────────────────

# Format:
#   documents   : list of strings (plain text)
#   annotations : list of UATs URIs (expected nodes for each document)

documents = []
annotations = []

reader = corpus_loader.Reader()
for document in reader.read_corpus(ignore_kailas = False,
                                   corpus_folder = CORPUS_PATH):
    documents.append(document.text)
    annotations.append(document.uats)

# ─────────────────────────────────────────────────────────────────────
# 3. VECTORIZATION OF DOCUMENTS
# ─────────────────────────────────────────────────────────────────────

print("\n=== Vectorization of documents ===")
doc_embeddings = onto.embed_training_corpus(corpus_name = "ADS_HELIO_corpus",
                                            texts = documents)    # [D, 768]
print(f"Shape embeddings documents: {doc_embeddings.shape}")

# ─────────────────────────────────────────────────────────────────────
# 4. BUILDING DATASET
# ─────────────────────────────────────────────────────────────────────

labels_multihot = build_multihot(annotations, onto.node2idx)   # [D, N]
labels_smoothed = onto.labels_smoothing(labels_multihot,
                                        alpha = 0.95,
                                        steps = 3,
                                        edges = {0, 1, 2, 3}) # broader, narrower, related, self (for weight conservation)
print("labels multihot std:", labels_multihot.std(dim=0).mean())
print("labels smoothed std:", labels_smoothed.std(dim=0).mean())

dataset = DocumentOntologyDataset(doc_embeddings, labels_smoothed)#labels_multihot)
print(f"Labels density : {labels_multihot.mean():.4f} (ratio of 1s in the matrix)")
print(f"Labels smoothed density : {labels_smoothed.mean():.4f} (ratio of 1s in the matrix)")

# What are the most frequent nodes in the corpus ?
counts = labels_multihot.sum(dim=0)
top = counts.topk(20)
print("Most frequent nodes in the labels:")
for idx, count in zip(top.indices, top.values):
    print(f"{count:.0f}  {onto.node_texts[idx][:60]}")

# ─────────────────────────────────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────────────────────────────────
print("doc emb shape[1]:", doc_embeddings.shape[1])

model = GNNOntologyClassifier(
    bert_dim      = doc_embeddings.shape[1],
    rgcn_hidden   = RGCN_HIDDEN,
    rgcn_out      = RGCN_OUT,
    num_relations = len(OntologyGraph.EDGE_TYPES),
    num_layers    = NUM_LAYERS,
    dropout       = DROPOUT,
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal of trainable parameters: {total_params:,}")

# ─────────────────────────────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────────────────────────────

log_dir = f"{datetime.datetime.now().strftime(format='%Y%m%d-%H%M%S')}_{RGCN_HIDDEN}h-{RGCN_OUT}o-{NUM_LAYERS}l"
print("\n=== Training ===")
best_f1 = train(
    model           = model,
    graph           = graph,
    dataset         = dataset,
    epochs          = EPOCHS,
    batch_size      = BATCH_SIZE,
    lr              = LR,
    pos_weight_factor = 1.0,
    device          = DEVICE,
    checkpoint_path = CHECKPOINT_PATH,
    log_dir         = RUNS / log_dir
)


# ─────────────────────────────────────────────────────────────────────
# 7. INFERENCE
# ─────────────────────────────────────────────────────────────────────

print("\n=== Inference (examples) ===")

# Load the best model
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

new_texts = []
new_annotations = []
for doc in corpus_loader.Reader().read_pre9forADS():
    new_texts.append(doc.text)
    new_annotations.append(doc.uats)


new_emb = onto.embed_training_corpus(corpus_name = "pre9forADS",
                                     texts = new_texts)

results = predict(
    model          = model,
    doc_embeddings = new_emb,
    graph          = graph,
    idx2node       = onto.idx2node,
    node_texts     = onto.node_texts,
    threshold      = 0.05,    # variable (use tau = 0.01 or tau = 0.05 for 2411 labels)
    top_k          = TOP_K,
    device         = DEVICE,
)

y_pred = []
y_true = []
total = 0
for i, (doc, annotation, res) in enumerate(zip(new_texts, new_annotations, results)):
    print(f"\nDocument {i+1}: {doc}")
    if res:
        y_pred_doc = []
        for uri, label, score in res:
            print(f"  → [{score:.3f}] {label}  ({uri})")
            y_pred_doc.append(uri)
        print("true =", annotation)
        y_pred.append(y_pred_doc)
        y_true.append(annotation)
    else:
        print("  → No node above the threshold")
    total += 1
print_results(y_true, y_pred, "preprint", total)



##### ADS Helio ########

new_texts = []
new_annotations = []
for doc in corpus_loader.Reader().read_corpus(False, ADS_HELIO_CORPUS_DIR):
    new_texts.append(doc.text)
    new_annotations.append(doc.uats)

new_emb = onto.embed_training_corpus(corpus_name = "ADS_HELIO_CORPUS",
                                     texts = new_texts)

results = predict(
    model          = model,
    doc_embeddings = new_emb,
    graph          = graph,
    idx2node       = onto.idx2node,
    node_texts     = onto.node_texts,
    threshold      = 0.05,    # variable (use tau = 0.01 or tau = 0.05 for 2411 labels)
    top_k          = TOP_K,
    device         = DEVICE,
)
y_pred = []
y_true = []
total = 0
for i, (doc, annotation, res) in enumerate(zip(new_texts, new_annotations, results)):
    if res:
        y_pred_doc = []
        for uri, label, score in res:
            y_pred_doc.append(uri)
        y_pred.append(y_pred_doc)
        y_true.append(annotation)
    else:
        print("  → No node above the threshold")
    total += 1
print_results(y_true, y_pred, "ADS Helio", total)