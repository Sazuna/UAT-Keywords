import numpy as np
import torch
import torch.nn as nn
import argparse
import joblib

from tqdm import tqdm
from typing import List, Tuple, Dict
from rdflib import Graph, URIRef, RDF, SKOS
from src.tfidf.coherence import Coherence
from src.utils.corpus_loader import Reader
from src.utils.config import ADS_CORPUS_DIR, ADS_HELIO_CORPUS_DIR, UATS_RDF_V6, CACHE_DIR

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Hyperparameters
FROM_HF: bool     = False # Load corpus from HF instead of disk
MAX_FEATURES: int = 100000 # tf-idf vectorizer max features
INPUT_DIM: int    = 8192 # input dim of the classifier (truncatedSVD will select N features from the above max_features)
HIDDEN_DIM: int   = 4096
EPOCHS: int       = 10
TOP_K: int        = 10
LR: float         = 1e-4 # learning rate
WD: float         = 1e-5 # weight decay
batch_size: int   = 64
DTYPE             = torch.float32
VERBOSE: bool     = True

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--max_features")

if not  VERBOSE:
    print = lambda x: 0

print("Experiment hyperparameters:")
print(f"\tMAX_FEATURES = {MAX_FEATURES}")
print(f"\tINPUT_DIM    = {INPUT_DIM}")
print(f"\tHIDDEN_DIM   = {HIDDEN_DIM}")
print(f"\tEPOCHS       = {EPOCHS}")
print(f"\tLR           = {LR}")
print(f"\tWD           = {WD}")
print(f"\tTOP_K        = {TOP_K}")
print(f"\tCORPUS:      = {ADS_CORPUS_DIR}")

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
        for _, _, label in rdf.triples((URIRef(uri), SKOS.prefLabel, None)):
            uri2label[str(uri)] = str(label)
    return uri2label

rdf = Graph()
rdf.parse(UATS_RDF_V6)
node2idx, idx2node = load_nodes(rdf)
node2label = load_labels(node2idx, rdf)

# 1. Load corpus
reader = Reader()
if not FROM_HF:
    texts  = []
    uats   = []

    for doc in reader.read_corpus(ignore_kailas=False,
                                  corpus_folder=ADS_CORPUS_DIR):
        text = doc.text
        uat = doc.uats
        texts.append(text)
        uats.append(uat)

    coherence = Coherence(uats) # TODO only use uats in the train dataset to build coherence

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
else:
    # from HF
    dataset = reader.get_hf_corpus("Sazuna/UAT_keywords", split="train")
    texts = dataset["text"]
    uats = dataset["uat_uri"]
    multihot = dataset["multihot"]
#def tokenizer(text):
#    return regex.findall(r"[\w]+", text)

# 2. Learn TF-IDF
print("Training TF-IDF vectorizer...")
vectorizer = TfidfVectorizer(lowercase=True,
                             stop_words=stop_words,
                             ngram_range=(1, 3),
                             max_features=MAX_FEATURES,
                             dtype=np.float32)

# X = vectorizer.fit_transform(texts)
# Y = multihot
print("Done")

# 3. Transform texts

print("Transforming text to features...")
### Truncate to have less features
svd = TruncatedSVD(n_components=INPUT_DIM, random_state=42)
pipeline = Pipeline([
    ("tfidf", vectorizer),
    ("svd", svd)
])

# Saving vectorizer & svd
joblib.dump(pipeline, CACHE_DIR / "featurizer.joblib")

# 4. Model
class Model(nn.Module):
    def __init__(self, input_dim, n_labels):
        super().__init__()
        self.linear = nn.Linear(input_dim, n_labels)

    def forward(self, x):
        return self.linear(x)  # logits

# X_tensor = torch.tensor(X)
#X_dense = X.toarray()
#X_tensor = torch.tensor(X_dense, dtype=DTYPE)
device = "cuda" if torch.cuda.is_available else "cpu"

# Train & validation split
X_train, X_val, Y_train, Y_val = train_test_split(
    texts,
    multihot,
    test_size=0.05,
    random_state=42,
    shuffle=True
)

X_train = pipeline.fit_transform(X_train)
X_val = pipeline.transform(X_val)
print("Done")

X_train = torch.tensor(X_train, dtype=DTYPE, device=device)
Y_train = torch.tensor(Y_train, dtype=DTYPE, device=device)
X_val   = torch.tensor(X_val, dtype=DTYPE, device=device)
Y_val   = torch.tensor(Y_val, dtype=DTYPE, device=device)

# Y_tensor = torch.tensor(Y)
model = Model(input_dim=X_train.shape[1], n_labels=Y_train.shape[1])
model.to(device)
pos_weight = (Y_train.shape[0] - Y_train.sum(axis=0)) / (Y_train.sum(axis=0) + 1e-6)
pos_weight = torch.clamp(pos_weight, 1.0, 1000.0) # np.clip(pos_weight, 1.0, 100.0)  # Maximum to 100
pos_weight = torch.tensor(pos_weight, dtype=DTYPE, device=device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)

n_samples = X_train.shape[0]
print(f"\tTRAIN SIZE = {n_samples}")
print(f"\tTEST SIZE  = {X_val.shape[0]}")
print(Y_train.min(), Y_train.max())
print(Y_val.min(), Y_val.max())
print("Training linear regression model...")
best_loss = 1000000

for epoch in tqdm(range(EPOCHS)):
    total_loss = 0.0
    total_loss_count = 0
    model.train() # train mode
    for i in range(0, n_samples, batch_size):
        X_train_batch = X_train[i:i+batch_size]
        Y_train_batch = Y_train[i:i+batch_size]

        #X_train_batch = torch.tensor(X_train_batch, dtype=DTYPE).to(device)
        #Y_train_batch = torch.tensor(Y_train_batch, dtype=DTYPE).to(device)

        optimizer.zero_grad()
        
        logits = model(X_train_batch)   # [batch, 2411]
        train_loss = criterion(logits, Y_train_batch)
        
        train_loss.backward()
        optimizer.step()

        total_loss += train_loss.item()
        total_loss_count += 1

    # Validation
    model.eval() # eval mode
    total_val_loss = 0.0
    total_val_count = 0
    with torch.no_grad():
        precision_macro, precision_micro = 0, 0
        recall_macro, recall_micro = 0, 0
        f1_macro, f1_micro = 0, 0
        for i in range(0, X_val.shape[0], batch_size):
            X_batch = X_val[i:i+batch_size]
            Y_batch = Y_val[i:i+batch_size]

            X_tensor = X_batch # torch.tensor(X_batch, dtype=DTYPE).to(device)
            Y_tensor = Y_batch # torch.tensor(Y_batch, dtype=DTYPE).to(device)

            logits = model(X_tensor)
            loss = criterion(logits, Y_tensor)

            total_val_loss += loss.item()
            total_val_count += 1


            probs = torch.sigmoid(logits)
            top_k = torch.topk(probs, k=TOP_K, dim=1)


            y_pred = torch.zeros_like(probs, dtype=torch.int)
            y_pred.scatter_(1, top_k.indices, 1)
            y_pred = y_pred.detach().cpu().numpy()
            y_true = Y_tensor.detach().cpu().numpy()

            precision_macro += precision_score(y_true, y_pred, average="macro", zero_division=0)
            recall_macro += recall_score(y_true, y_pred, average="macro", zero_division=0)
            f1_macro += f1_score(y_true, y_pred, average="macro", zero_division=0)

            precision_micro = precision_score(y_true, y_pred, average="micro", zero_division=0)
            recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=0)
            f1_micro = f1_score(y_true, y_pred, average="micro", zero_division=0)

    precision_macro /= total_val_count
    recall_macro /= total_val_count
    f1_macro /= total_val_count
    print(f"Precision (macro): {precision_macro:.4f}")
    print(f"Recall    (macro): {recall_macro:.4f}")
    print(f"F1-score  (macro): {f1_macro:.4f}")

    precision_micro /= total_val_count
    recall_micro /= total_val_count
    f1_micro /= total_val_count
    print(f"Precision (micro): {precision_micro:.4f}")
    print(f"Recall    (micro): {recall_micro:.4f}")
    print(f"F1-score  (micro): {f1_micro:.4f}")
    avg_train_loss = total_loss / total_loss_count
    avg_val_loss = total_val_loss / total_val_count
    print(f"Epoch {epoch} | train loss={avg_train_loss:.4f} | val loss={avg_val_loss:.4f}")
    if avg_val_loss < best_loss:
        print("Saving best model...")
        torch.save(model.state_dict(), CACHE_DIR / f"classifier_{INPUT_DIM}.pt")
        best_loss = avg_val_loss
print("Done")


def classify(text: str,
             TOP_K: int) -> list[str]:
    x = pipeline.transform(text)
    x_tensor = torch.tensor(x, device = device)
    logits = model(x)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.1).int()
    top_k = torch.topk(probs, k=TOP_K, dim = 1)
    predicted_uats = [idx2node[int(uat)] for uat in idx]
    return predicted_uats

# 5. Inference

if __name__ == "__main__":
    ### Eval on preprint corpus ###
    text_test = []
    uats_test = []
    bibcodes  = []
    for document in reader.read_pre9forADS():
        text_test.append(document.text)
        uats_test.append(document.uats)
        bibcodes.append(document.bibcode)

    X_test = pipeline.transform(text_test)
    X_test_tensor = torch.tensor(X_test, device=device)

    Y_test = build_multihot(uats_test, node2idx=node2idx)
    Y_test_tensor = torch.tensor(Y_test, device=device)


    logits = model(X_test_tensor)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.1).int()
    top_k = torch.topk(probs, k=TOP_K, dim=1)
    for idx, scores, bibcode, text, uats in zip(
        top_k.indices,
        top_k.values,
        bibcodes,
        text_test,
        uats_test
    ):
        predicted_uats = [idx2node[int(uat)] for uat in idx]
        uats_labels = [node2label.get(uat, "UNKNOWN") for uat in uats]
        predicted_uats_labels = [node2label.get(uat, "UNKNOWN") for uat in predicted_uats]
        print("Bibcode:", bibcode, "Text:", text)
        print("paper UATs:", '\n\t'.join([f"{uat} ({label})" for uat, label in zip(uats, uats_labels)]))
        print("predicted UATs:")
        for i, (score, uat, label) in enumerate(zip(scores, predicted_uats, predicted_uats_labels), start=1):
            print(f"\tTop {i} | {score:.4f}: {uat} ({label})")
        if coherence:
            print(f"Mean npmi of top {TOP_K} labels:", coherence.mean_npmi(uats))
        print("")


    y_pred = torch.zeros_like(probs, dtype=torch.int)
    y_pred.scatter_(1, top_k.indices, 1)
    y_pred = y_pred.detach().cpu().numpy()
    y_true = Y_test_tensor.detach().cpu().numpy()

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall    (macro): {recall:.4f}")
    print(f"F1-score  (macro): {f1:.4f}")

    precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    print(f"Precision (micro): {precision:.4f}")
    print(f"Recall    (micro): {recall:.4f}")
    print(f"F1-score  (micro): {f1:.4f}")



    ### Eval on HP corpus ###
    text_test = []
    uats_test = []
    bibcodes  = []
    for document in reader.read_corpus(ignore_kailas=False,
                                       corpus_folder=ADS_HELIO_CORPUS_DIR):
        text_test.append(document.text)
        uats_test.append(document.uats)
        bibcodes.append(document.bibcode)

    X_test = pipeline.transform(text_test)
    X_test_tensor = torch.tensor(X_test, device=device)

    Y_test = build_multihot(uats_test, node2idx=node2idx)
    Y_test_tensor = torch.tensor(Y_test, device=device)


    logits = model(X_test_tensor)
    probs = torch.sigmoid(logits)
    preds = (probs > 0.1).int()
    top_k = torch.topk(probs, k=TOP_K, dim=1)

    """
    for idx, scores, bibcode, text, uats in zip(
        top_k.indices,
        top_k.values,
        bibcodes,
        text_test,
        uats_test
    ):
        predicted_uats = [idx2node[int(uat)] for uat in idx]
        uats_labels = [node2label.get(uat, "UNKNOWN") for uat in uats]
        predicted_uats_labels = [node2label.get(uat, "UNKNOWN") for uat in predicted_uats]
        print("Bibcode:", bibcode, "Text:", text)
        print("paper UATs:", '\n\t'.join([f"{uat} ({label})" for uat, label in zip(uats, uats_labels)]))
        print("predicted UATs:")
        for i, (score, uat, label) in enumerate(zip(scores, predicted_uats, predicted_uats_labels), start=1):
            print(f"\tTop {i} | {score:.4f}: {uat} ({label})")
        print(f"Mean npmi of top {TOP_K} labels:", coherence.mean_npmi(uats))
        print("")
    """


    y_pred = torch.zeros_like(probs, dtype=torch.int)
    y_pred.scatter_(1, top_k.indices, 1)
    y_pred = y_pred.detach().cpu().numpy()
    y_true = Y_test_tensor.detach().cpu().numpy()

    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall    (macro): {recall:.4f}")
    print(f"F1-score  (macro): {f1:.4f}")

    precision = precision_score(y_true, y_pred, average="micro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="micro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)

    print(f"Precision (micro): {precision:.4f}")
    print(f"Recall    (micro): {recall:.4f}")
    print(f"F1-score  (micro): {f1:.4f}")
