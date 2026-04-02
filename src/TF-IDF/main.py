import numpy as np
import torch
import torch.nn as nn
from typing import List, Tuple, Dict
from rdflib import Graph, RDF, SKOS
from corpus_loader import Reader
from config import ADS_CORPUS_DIR

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD


# Hyperparameters
MAX_FEATURES: int = 30000
INPUT_DIM: int    = 512#2048
MAX_ITER: int     = 3000
batch_size: int   = 64

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

# 1. Load corpus
reader = Reader()
texts  = []
uats   = []

for text, uat in reader.read_corpus(ignore_kailas=False,
                                    corpus_folder=ADS_CORPUS_DIR):
    texts.append(text)
    uats.append(uat)

### Generate multihot
def load_nodes(uat_ontology_path) -> Tuple[Dict, Dict]:
    rdf = Graph()
    rdf.parse(uat_ontology_path, format="xml")

    # 1. Collect all nodes (subjects or objects of a SKOS relation)
    nodes_set = set()
    for s, _, _ in rdf.triples((None, RDF.type, SKOS.Concept)):
        s_str = str(s)
        nodes_set.add(s_str)

    # 2. Node indexation
    sorted_nodes = sorted(nodes_set, key = lambda x: int(x.split('/')[-1]))
    node2idx = {n: i for i, n in enumerate(sorted_nodes)}
    idx2node = {i: n for n, i in node2idx.items()}
    return node2idx, idx2node 
def build_multihot(
    annotations: List[List[str]],
    node2idx: Dict[str, int],
) -> np.array:#torch.Tensor:
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
    # multihot = torch.zeros(D, N)
    multihot = np.zeros([D, N])
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

node2idx, idx2node = load_nodes("../corpus/UAT_v6.0.0.rdf")
multihot = build_multihot(annotations=uats,
                          node2idx=node2idx)

#def tokenizer(text):
#    return regex.findall(r"[\w]+", text)

# 2. Learn TF-IDF
vectorizer = TfidfVectorizer(lowercase=True,
                             stop_words=stop_words,
                             ngram_range=(1, 3),
                             max_features=MAX_FEATURES,
                             dtype=np.float32)

vectorizer = vectorizer.fit(texts)

# 3. Transform texts

X = vectorizer.transform(texts)
Y = multihot

### Truncate to have less features
svd = TruncatedSVD(n_components=INPUT_DIM, random_state=42)
svd = svd.fit(X)
X = svd.fit_transform(X)

# 4. Logistic regression

### 2411 classifications (very slow) ###
"""
model = OneVsRestClassifier(
    LogisticRegression(
        solver="saga",        # Sparse + scalable
        class_weight="balanced",
        max_iter=MAX_ITER,
        n_jobs=-1,
    ),
    verbose=1,
)
model.fit(X, Y)
"""
"""
model = OneVsRestClassifier(
    SGDClassifier(
        loss="log_loss",
        max_iter=MAX_ITER,
        verbose=1,
        n_jobs=-1
    )
)
model.fit(X, Y)

# 5. Inference
text_test = []
uats_test = []
for text, uat in reader.read_pre9forADS():
    text_test.append(text)
    uats_test.append(uat)

X_test = vectorizer.transform(text_test)

Y_proba = model.predict_proba(X_test)  # shape [D_test, 2411]

def top_k_predictions(Y_proba, k=5):
    Y_pred = np.zeros_like(Y_proba)
    for i in range(Y_proba.shape[0]):
        top_k = np.argsort(Y_proba[i])[-k:]
        Y_pred[i, top_k] = 1
    return Y_pred

Y_pred = top_k_predictions(Y_proba, k=5)

print("Micro F1:", f1_score(uats_test, Y_pred, average="micro"))
print("Macro F1:", f1_score(uats_test, Y_pred, average="macro"))

"""

# 4. Model
class Model(nn.Module):
    def __init__(self, input_dim, n_labels):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_labels)

    def forward(self, x):
        return self.linear(x)  # logits

# X_tensor = torch.tensor(X)
#X_dense = X.toarray()
#X_tensor = torch.tensor(X_dense, dtype=torch.float32)
Y_tensor = torch.tensor(Y)

model = Model(input_dim=X.shape[1], n_labels=Y_tensor.shape[1])
pos_weight = (Y_tensor.shape[0] - Y_tensor.sum(axis=0)) / (Y_tensor.sum(axis=0) + 1e-6)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32)
#criterion = nn.BCEWithLogitsLoss()
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

n_samples = X.shape[0]
for epoch in range(10):
    total_loss = 0.0

    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size] #.toarray() #no need after TruncatedSVD
        Y_batch = Y[i:i+batch_size]

        X_tensor = torch.tensor(X_batch, dtype=torch.float32)
        Y_tensor = torch.tensor(Y_batch, dtype=torch.float32)
        optimizer.zero_grad()
        
        logits = model(X_tensor)   # [batch, 2411]
        loss = criterion(logits, Y_tensor)
        
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch}, loss={total_loss:.4f}")

# 5. Inference
text_test = []
uats_test = []
for text, uat in reader.read_pre9forADS():
    text_test.append(text)
    uats_test.append(uat)

X_test = vectorizer.transform(text_test)
X_test = svd.transform(X_test)
X_test_tensor = torch.tensor(X_test)

print(X_test_tensor.shape)
logits = model(X_test_tensor)
probs = torch.sigmoid(logits)
preds = (probs > 0.1).int()
top_k = torch.topk(probs, k=10, dim=1)
print(top_k)
print(idx2node.keys())
for idx, (text, uats) in zip(top_k.indices, zip(text_test, uats_test)):
    predicted_uats = [idx2node[int(uat)] for uat in idx]
    print("Text:", text[:512])
    print("paper UATs:", ' '.join(uats))
    print("predicted UATs:", ' '.join(predicted_uats))
