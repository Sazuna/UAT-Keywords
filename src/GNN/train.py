"""
train.py
Dataset + boucle d'entraînement pour la classification multi-label
document → nœuds d'ontologie.
"""

import torch
import torch.nn as nn
import math
from torch.utils.data import Dataset, DataLoader, random_split
from torch_geometric.data import Data
from typing import List, Tuple, Dict
from config import THRESHOLD
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

from model import GNNOntologyClassifier


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────

class DocumentOntologyDataset(Dataset):
    """
    Each example = (document_embedding, label_vector_multi_hot).

    Args:
        doc_embeddings : [D, bert_dim]   documents embeddings (pre-computed)
        labels_multihot: [D, N]          multi-hot vectors of the N nodes
    """

    def __init__(self, doc_embeddings: torch.Tensor, labels_multihot: torch.Tensor):
        assert doc_embeddings.shape[0] == labels_multihot.shape[0]
        self.doc_emb = doc_embeddings.float()
        self.labels  = labels_multihot.float()

    def __len__(self):
        return len(self.doc_emb)

    def __getitem__(self, idx):
        return self.doc_emb[idx], self.labels[idx]


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
        for uri in ann_list:
            if uri in node2idx:
                multihot[i, node2idx[uri]] = 1.0
                # TODO label smoothing (https://www.mdpi.com/2079-9292/13/15/2944)
            else:
                print(f"[Warning] Unknown URI in node2idx : {uri}")
    return multihot


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def evaluate(
    model: GNNOntologyClassifier,
    loader: DataLoader,
    graph: Data,
    criterion: nn.Module,
    top_k: int = 5,
    threshold: float = THRESHOLD,
    device: str = "cpu",
) -> Dict[str, float]:
    model.eval()
    all_preds, all_labels = [], []

    x          = graph.x.to(device)
    edge_index = graph.edge_index.to(device)
    edge_type  = graph.edge_type.to(device)
    
    total_val_loss = 0
    true_labels_by_doc = 0
    pred_labels_by_doc = 0
    with torch.no_grad():
        for doc_emb, labels in loader:
            doc_emb = doc_emb.to(device)
            logits  = model(doc_emb, x, edge_index, edge_type)
            labels = labels.to(device)
            #preds   = (torch.sigmoid(logits) >= threshold).cpu().numpy()
            
            # ---- get top_k preds -------
            topk = torch.topk(logits, top_k, dim=1)
            preds = torch.zeros_like(logits)
            preds.scatter_(1, topk.indices, 1)
            #print(preds)
            #print(preds.sum())
            #print(preds.shape)
            # ---- get top_k labels -------
            # (if smoothed, convert back to onehot)
            ### Fix ?
            onehot = torch.zeros_like(labels)
            for i in range(labels.shape[0]):
                k = int(labels[i].sum().item())

                if k > 0:
                    topk = torch.topk(labels[i], k)
                    onehot[i, topk.indices] = 1
            # ------------------------------

            val_loss = criterion(logits, labels)
            total_val_loss += val_loss.item()

            pred_labels_by_doc += preds.sum() / preds.shape[0]
            true_labels_by_doc += onehot.sum() / onehot.shape[0]

            all_preds.append(preds)
            all_labels.append(onehot.numpy())

    avg_val_loss = total_val_loss / len(loader)
    pred_labels_by_doc /= len(loader)
    true_labels_by_doc /= len(loader)

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_labels)

    return {
        "f1_micro":        f1_score(y_true, y_pred, average="micro", zero_division=0),
        "f1_macro":        f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "recall_micro":    recall_score(y_true, y_pred, average="micro", zero_division=0),
        "validation_loss": avg_val_loss,
        "true_labels_by_doc": true_labels_by_doc,
        "pred_labels_by_doc": pred_labels_by_doc
    }


# ─────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────

def train(
    model: GNNOntologyClassifier,
    graph: Data,
    dataset: DocumentOntologyDataset,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_ratio: float = 0.15,
    pos_weight_factor: float = 1.0,   # compensate desequilibrum multi-label (TODO remove this)
    device: str = "cpu",
    checkpoint_path: str = "best_model.pt",
):
    # Split train / val
    val_size   = max(1, int(len(dataset) * val_ratio))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

    model.to(device)
    graph_x           = graph.x.to(device)
    graph_edge_index  = graph.edge_index.to(device)
    graph_edge_type   = graph.edge_type.to(device)

    # BCE with positive class weighting (multi-label)
    # num_nodes = graph.num_nodes
    # pos_weight = torch.ones(num_nodes, device=device) # * pos_weight_factor
    """
    num_nodes = graph.num_nodes
    pos_counts = torch.zeros(num_nodes)
    total_samples = 0
    for _, y in train_ds:
        pos_counts += y
        total_samples += 1

    neg_counts = total_samples - pos_counts
    pos_weight = neg_counts / (pos_counts + 1e-6)
    pos_weight = pos_weight.to(device)
    #print("pos_weight:", pos_weight)
    """

    criterion = nn.BCEWithLogitsLoss()#pos_weight=pos_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5
    )

    best_f1 = 0.0
    best_val_loss = math.inf

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        for doc_emb, labels in train_loader:
            doc_emb = doc_emb.to(device)
            labels  = labels.to(device)

            optimizer.zero_grad()
            logits = model(doc_emb, graph_x, graph_edge_index, graph_edge_type)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation if not smoothed labels
        metrics = evaluate(model, val_loader, graph, threshold=THRESHOLD, criterion=criterion, device=device)
        f1 = metrics["f1_micro"]
        val_loss = metrics["validation_loss"]
        scheduler.step(f1)

        print(
            f"Epoch {epoch:03d}/{epochs} | training loss={avg_loss:.4f} | "
            f"validation loss={metrics['validation_loss']:.4f} | "
            f"f1_micro={f1:.4f} | f1_macro={metrics['f1_macro']:.4f} | "
            f"prec={metrics['precision_micro']:.4f} | recall={metrics['recall_micro']:.4f}"
        )
        print(
            f"true labels by doc={metrics['true_labels_by_doc']:.4f} | "
            f"pred labels by doc={metrics['pred_labels_by_doc']:.0f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Best model saved (val loss={best_val_loss:.4f})")
        """
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  ✓ Best model saved (f1={best_f1:.4f})")
        """

    print(f"\nTraining done. Best micro-F1: {best_f1:.4f}")
    return best_f1


# ─────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────

@torch.no_grad()
def predict(
    model: GNNOntologyClassifier,
    doc_embeddings: torch.Tensor,
    graph: Data,
    idx2node: Dict[int, str],
    node_texts: List[str],
    threshold: float = THRESHOLD,
    top_k: int = None,
    device: str = "cpu",
) -> List[List[Tuple[str, str, float]]]:
    """
    For each document, return un aliste of (uri, label, score) sorted by score.

    Args:
        doc_embeddings : [D, bert_dim]
        top_k          : if specified, return top_k nodes (ignore threshold)

    Returns:
        list of listes [(uri, label, score), ...]
    """
    model.eval().to(device)
    x          = graph.x.to(device)
    edge_index  = graph.edge_index.to(device)
    edge_type   = graph.edge_type.to(device)

    logits = model(doc_embeddings.to(device), x, edge_index, edge_type)  # [D, N]
    probs  = torch.sigmoid(logits).cpu()

    results = []
    for i in range(probs.shape[0]):
        p = probs[i]
        if top_k is not None:
            indices = torch.topk(p, k=top_k).indices.tolist()
        else:
            indices = (p >= threshold).nonzero(as_tuple=True)[0].tolist()

        doc_results = sorted(
            [(idx2node[j], node_texts[j], p[j].item()) for j in indices],
            key=lambda x: x[2], reverse=True
        )
        results.append(doc_results)

    return results
