"""
model.py
Modèle GNN pour la classification multi-label de documents sur une ontologie.

Architecture :
  1. R-GCN  : propage les embeddings BERT des nœuds en tenant compte des
              3 types d'arêtes (broader / narrower / related).
  2. Scorer : pour chaque document vectorisé, calcule un score d'affinité
              avec chaque nœud du graphe → sigmoid → multi-label.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv


class OntologyRGCN(nn.Module):
    """
    R-GCN that refines embeddings of the ontology's nodes

    Args:
        in_channels   : dim des features d'entrée (768 pour BERT-base)
        hidden_channels : dim des couches cachées
        out_channels  : dim de la représentation finale des nœuds
        num_relations : nombre de types d'arêtes (3 ici)
        num_layers    : profondeur du R-GCN
        dropout       : taux de dropout
    """

    def __init__(
        self,
        in_channels: int = 768,
        hidden_channels: int = 256,
        out_channels: int = 128,
        num_relations: int = 3,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout

        # Initial BERT projection
        self.input_proj = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.LayerNorm(hidden_channels),
            nn.ReLU(),
        )

        # R-GCN Layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        dims = [hidden_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for i in range(num_layers):
            self.convs.append(
                RGCNConv(dims[i], dims[i + 1], num_relations=num_relations)
            )
            self.norms.append(nn.LayerNorm(dims[i + 1]))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x          : [N, in_channels]  initial node features
            edge_index : [2, E]
            edge_type  : [E]  edge index in {0,1,2,3}

        Returns:
            node_emb : [N, out_channels]  modified representations
        """
        h = self.input_proj(x)

        for conv, norm in zip(self.convs, self.norms):
            h = conv(h, edge_index, edge_type)
            h = norm(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return h  # [N, out_channels]


class DocumentNodeScorer(nn.Module):
    """
    Projette un document dans le même espace que les nœuds du graphe,
    puis calcule des scores d'affinité → classification multi-label.

    Args:
        doc_in_dim    : dim de l'embedding document (768 pour BERT-base)
        node_out_dim  : out_channels de OntologyRGCN
        hidden_dim    : dim de la couche cachée du scorer
    """

    def __init__(self, doc_in_dim: int = 768, node_out_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.doc_proj = nn.Sequential(
            nn.Linear(doc_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, node_out_dim),
        )

    def forward(self, doc_emb: torch.Tensor, node_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            doc_emb  : [B, doc_in_dim]     embeddings des documents du batch
            node_emb : [N, node_out_dim]   embeddings des nœuds (figés pendant le forward)

        Returns:
            logits : [B, N]  un score par nœud par document (avant sigmoid)
        """
        d = self.doc_proj(doc_emb)           # [B, node_out_dim]
        d = F.normalize(d, dim=-1)
        n = F.normalize(node_emb, dim=-1)    # [N, node_out_dim]
        logits = d @ n.T                     # [B, N]

        return logits


class GNNOntologyClassifier(nn.Module):
    """
    Modèle complet : R-GCN + scorer document→nœuds.

    Usage typique :
        model = GNNOntologyClassifier(...)
        logits = model(doc_emb, graph.x, graph.edge_index, graph.edge_type)
        probs  = torch.sigmoid(logits)
    """

    def __init__(
        self,
        bert_dim: int = 768,
        rgcn_hidden: int = 256,
        rgcn_out: int = 128,
        num_relations: int = 5,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.rgcn = OntologyRGCN(
            in_channels=bert_dim,
            hidden_channels=rgcn_hidden,
            out_channels=rgcn_out,
            num_relations=num_relations,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.scorer = DocumentNodeScorer(
            doc_in_dim=bert_dim,
            node_out_dim=rgcn_out,
        )

    def forward(
        self,
        doc_emb: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
    ) -> torch.Tensor:
        node_emb = self.rgcn(node_features, edge_index, edge_type)   # [N, rgcn_out]
        logits = self.scorer(doc_emb, node_emb)                       # [B, N]
        return logits
