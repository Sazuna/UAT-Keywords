"""
main.py

Load the UATs ontology, vectorize documents, train GNN and infer on our pre-print corpus.
"""

import torch
from config import BERT_MODEL, THRESHOLD, ADS_HELIO_CORPUS_DIR, ADS_CORPUS_DIR
from ontology_graph import OntologyGraph
import corpus_loader
from model import GNNOntologyClassifier
from train import (
    DocumentOntologyDataset,
    build_multihot,
    train,
    predict,
)


# ─────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────

TURTLE_PATH   = "../../corpus/UAT_v6.0.0.rdf" # Input graph
CORPUS_PATH   = ADS_HELIO_CORPUS_DIR #ADS_CORPUS_DIR # can use ADS_HELIOPHYSICS_CORPUS_DIR for HP-only
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
RGCN_HIDDEN   = 256
RGCN_OUT      = 128
NUM_LAYERS    = 2     # smoothing of the representations
DROPOUT       = 0.3
EPOCHS        = 10
BATCH_SIZE    = 16
LR            = 1e-3


# ─────────────────────────────────────────────────────────────────────
# 1. LOAD ONTOLOGY
# ─────────────────────────────────────────────────────────────────────

print("=== Loading ontology ===")
onto = OntologyGraph(turtle_path=TURTLE_PATH, bert_model_name=BERT_MODEL)
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
for document, annotation in reader.read_corpus(ignore_kailas = False,
                                               corpus_folder = CORPUS_PATH):
    documents.append(document)
    annotations.append(annotation)

# ─────────────────────────────────────────────────────────────────────
# 3. VECTORIZATION OF DOCUMENTS
# ─────────────────────────────────────────────────────────────────────

print("\n=== Vectorization of documents ===")
doc_embeddings = onto.embed_training_corpus(corpus_name = "ADS_corpus",
                                            texts = documents)    # [D, 768]
print(f"Shape embeddings documents: {doc_embeddings.shape}")

# ─────────────────────────────────────────────────────────────────────
# 4. BUILDING DATASET
# ─────────────────────────────────────────────────────────────────────

labels_multihot = build_multihot(annotations, onto.node2idx)   # [D, N]
labels_smoothed = onto.labels_smoothing(labels_multihot,
                                        alpha = 0.9,
                                        steps = 2,
                                        edges = {0, 1, 2, 3}) # broader, narrower, related, self (for weight conservation)
print("labels multihot std:", labels_multihot.std(dim=0).mean())
print("labels smoothed std:", labels_smoothed.std(dim=0).mean())

dataset = DocumentOntologyDataset(doc_embeddings, labels_smoothed)#labels_multihot)
print(f"Labels density : {labels_multihot.mean():.4f} (ratio of 1s in the matrix)")
print(f"Labels smoothed density : {labels_smoothed.mean():.4f} (ratio of 1s in the matrix)")


# ─────────────────────────────────────────────────────────────────────
# 5. MODEL
# ─────────────────────────────────────────────────────────────────────

model = GNNOntologyClassifier(
    bert_dim      = doc_embeddings.shape[1],
    rgcn_hidden   = RGCN_HIDDEN,
    rgcn_out      = RGCN_OUT,
    num_relations = 4,   # broader / narrower / related / self # TODO add a global relation
    num_layers    = NUM_LAYERS,
    dropout       = DROPOUT,
)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal of trainable parameters: {total_params:,}")


# ─────────────────────────────────────────────────────────────────────
# 6. TRAINING
# ─────────────────────────────────────────────────────────────────────

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
    checkpoint_path = "best_model.pt",
)


# ─────────────────────────────────────────────────────────────────────
# 7. INFERENCE
# ─────────────────────────────────────────────────────────────────────

print("\n=== Inference (exemples) ===")

# Load the best model
model.load_state_dict(torch.load("best_model.pt", map_location=DEVICE))

new_docs = [
    "Impact of deforestation on global warming and species extinction.",
    "Deep learning architectures for text classification tasks.",
] # TODO load


new_docs = []
new_annotations = []
for text, uats in corpus_loader.Reader().read_pre9forADS():
    new_docs.append(text)
    new_annotations.append(uats)


new_emb = onto.embed_training_corpus(corpus_name = "pre9forADS",
                                     texts = new_docs)

results = predict(
    model          = model,
    doc_embeddings = new_emb,
    graph          = graph,
    idx2node       = onto.idx2node,
    node_texts     = onto.node_texts,
    threshold      = THRESHOLD,    # variable (use tau = 0.01 or tau = 0.05 for 2411 labels)
    top_k          = 5,
    device         = DEVICE,
)

for i, (doc, res) in enumerate(zip(new_docs, results)):
    print(f"\nDocument {i+1}: {doc}")
    if res:
        for uri, label, score in res:
            print(f"  → [{score:.3f}] {label}  ({uri})")
    else:
        print("  → No node above the threshold")
