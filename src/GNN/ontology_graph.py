"""
ontology_graph.py
Chargement de l'ontologie Turtle et construction du graphe PyTorch Geometric.
"""

import torch
import re
import pickle
from rdflib import Graph, URIRef
from rdflib.namespace import SKOS, RDF
from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from typing import Dict, Tuple, List
from pathlib import Path
from config import BERT_UATS_EMBEDDINGS_FILE, BERT_DOCS_EMBEDDINGS_FILE


SKOS_BROADER = SKOS.broader
SKOS_NARROWER = SKOS.narrower
SKOS_RELATED = SKOS.related


class OntologyGraph:
    """
    Load an ontology and produce
      - node_features  : tensor [N, hidden_dim] (BERT embeddings of textual representation)
      - edge_index     : tensor [2, E]
      - edge_type      : tensor [E] (0=broader, 1=narrower, 2=related, 3=self)
      - node2idx / idx2node : mappings URI <-> integer
    """

    EDGE_TYPES = {"broader": 0, "narrower": 1, "related": 2, "self": 3}

    def __init__(self, turtle_path: str, bert_model_name: str = "bert-base-uncased"):
        self.turtle_path = turtle_path
        self.bert_model_name = bert_model_name

        self.tokenizer = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert.eval() # activate the inference mode (disable dropout and gradients)

        self.node2idx: Dict[str, int] = {}
        self.idx2node: Dict[int, str] = {}
        self.node_texts: List[str] = []

        self.graph_data: Data = None

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def load(self) -> Data:
        rdf = Graph()
        rdf.parse(self.turtle_path, format="xml")

        # 1. Collect all nodes (subjects or objects of a SKOS relation)
        relations = [
            (SKOS_BROADER, "broader"),
            (SKOS_NARROWER, "narrower"),
            (SKOS_RELATED, "related"),
        ] # "self" for self-attention (node to self edges)

        edges_raw: List[Tuple[str, str, str]] = []
        nodes_set = set()

        for predicate, etype in relations:
            for s, _, o in rdf.triples((None, predicate, None)):
                s_str, o_str = str(s), str(o)
                nodes_set.update([s_str, o_str])
                edges_raw.append((s_str, o_str, etype))

        # Fallback: add nodes that are concepts and self-attention edges
        for s, _, _ in rdf.triples((None, RDF.type, SKOS.Concept)):
            nodes_set.add(str(s))
            edges_raw.append((s_str, s_str, "self")) # self attention edges

        # TODO add a global node (whole graph) with edges between the global node and all nodes

        # 2. Node indexation
        sorted_nodes = sorted(nodes_set)
        self.node2idx = {n: i for i, n in enumerate(sorted_nodes)}
        self.idx2node = {i: n for n, i in self.node2idx.items()}

        # 3. Textual representation (prefLabel + altLabel + definition)
        self.node_texts = []
        for node_uri in sorted_nodes:
            uri_ref = URIRef(node_uri)
            pref_label = self._get_literal(rdf, uri_ref, SKOS.prefLabel)
            alt_label = self._get_literal(rdf, uri_ref, SKOS.altLabel)
            definition = self._get_literal(rdf, uri_ref, SKOS.definition)
            text = pref_label + " " + alt_label + " " + definition
            text = re.sub(r"\s+", " ", text)
            # node_uri.split("/")[-1].split("#")
            self.node_texts.append(text)

        # 4. BERT Embeddings of labels
        node_features = None
        print(BERT_UATS_EMBEDDINGS_FILE)
        if BERT_UATS_EMBEDDINGS_FILE.exists():
            with open(BERT_UATS_EMBEDDINGS_FILE, "rb") as file:
                node_features = pickle.load(file)
        if node_features is None:
            node_features = self._embed_texts(self.node_texts)  # [N, 768]
            with open(BERT_UATS_EMBEDDINGS_FILE, "wb") as file:
                pickle.dump(node_features, file)

        # 5. Build edge_index and edge_type
        src_list, dst_list, etype_list = [], [], []
        for s_str, o_str, etype in edges_raw:
            src_list.append(self.node2idx[s_str])
            dst_list.append(self.node2idx[o_str])
            etype_list.append(self.EDGE_TYPES[etype])

        edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
        edge_type = torch.tensor(etype_list, dtype=torch.long)

        self.graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_type=edge_type,
            num_nodes=len(sorted_nodes),
        )

        print(f"[OntologyGraph] {len(sorted_nodes)} nodes, {len(edges_raw)} edges loaded.")
        return self.graph_data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_literal(rdf, subject, predicate) -> str:
        for _, _, o in rdf.triples((subject, predicate, None)):
            return str(o)
        return ""


    @torch.no_grad()
    def _embed_texts(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        """CLS-token embedding via BERT (embed nodes)"""
        all_embeds = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=64, return_tensors="pt"
            )
            out = self.bert(**enc)
            cls = out.last_hidden_state[:, 0, :]   # [B, 768]
            all_embeds.append(cls.cpu())
        return torch.cat(all_embeds, dim=0)


    def embed_training_corpus(self, corpus_name: str, texts: List[str], batch_size: int = 16) -> torch.Tensor:
        """
        Vectorize a static corpus of documents.
        Save the embeddings to prevent re-vectorizing in the future.
        """
        filename = str(BERT_DOCS_EMBEDDINGS_FILE).removesuffix(".pkl")
        filename += f"{corpus_name}_{len(texts)}docs.pkl"
        filename = Path(filename)
        if filename.exists():
            with open(filename, "rb") as file:
                return pickle.load(file)
        embeds = self._embed_texts(texts, batch_size)
        with open(filename, "wb") as file:
            pickle.dump(embeds, file)
        return embeds
