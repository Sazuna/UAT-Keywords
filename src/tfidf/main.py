import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import List, Tuple, Dict
from rdflib import Graph, URIRef, RDF, SKOS
from corpus_loader import Reader
from config import ADS_CORPUS_DIR

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.decomposition import TruncatedSVD


# Hyperparameters
MAX_FEATURES: int = 100000 # tf-idf vectorizer max features
INPUT_DIM: int    = 8192 # input dim of the classifier (truncatedSVD will select N features from the above max_features)
EPOCHS: int       = 10
TOP_K: int        = 10
batch_size: int   = 64

stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

# 0. Load ontology
def load_nodes(rdf) -> Tuple[Dict, Dict]:

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

def load_labels(node2idx,
                rdf: Graph) -> Dict:
    uri2label = dict()
    for uri in node2idx.keys():
        print(uri)
        for _, _, label in rdf.triples((URIRef(uri), SKOS.prefLabel, None)):
            uri2label[str(uri)] = str(label)
    return uri2label

rdf = Graph()
rdf.parse("../corpus/UAT_v6.0.0.rdf")
node2idx, idx2node = load_nodes(rdf)
node2label = load_labels(node2idx, rdf)

# 1. Load corpus
reader = Reader()
texts  = []
uats   = []

for text, uat in reader.read_corpus(ignore_kailas=False,
                                    corpus_folder=ADS_CORPUS_DIR):
    texts.append(text)
    uats.append(uat)

### Generate multihot
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

multihot = build_multihot(annotations=uats,
                          node2idx=node2idx)

#def tokenizer(text):
#    return regex.findall(r"[\w]+", text)

# 2. Learn TF-IDF
print("Training TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(lowercase=True,
                             stop_words=stop_words,
                             ngram_range=(1, 3),
                             max_features=MAX_FEATURES,
                             dtype=np.float32)

vectorizer = vectorizer.fit(texts)
print("Done")

# 3. Transform texts

print("Transforming text to features...")
X = vectorizer.transform(texts)
Y = multihot

### Truncate to have less features
svd = TruncatedSVD(n_components=INPUT_DIM, random_state=42)
svd = svd.fit(X)
X = svd.fit_transform(X)
print("Done")

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
print("Training linear regression model...")
for epoch in tqdm(range(EPOCHS)):
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
print("Done")

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
top_k = torch.topk(probs, k=TOP_K, dim=1)
print(top_k)
for idx, (scores, (text, uats)) in zip(top_k.indices, zip(top_k.values, zip(text_test, uats_test))):
    predicted_uats = [idx2node[int(uat)] for uat in idx]
    uats_labels = [node2label.get(uat, "UNKNOWN") for uat in uats]
    predicted_uats_labels = [node2label.get(uat, "UNKNOWN") for uat in predicted_uats]
    print("Text:", text[:512])
    print("paper UATs:", '\n\t'.join([f"{uat} ({label})" for uat, label in zip(uats, uats_labels)]))
    #print("predicted UATs:", ' '.join(predicted_uats))
    print("predicted UATs:", '\n\t'.join([f"{score:.4f}: {uat} ({label})" for score, (uat, label) in zip(scores, zip(predicted_uats, predicted_uats_labels))]))
    print("")
