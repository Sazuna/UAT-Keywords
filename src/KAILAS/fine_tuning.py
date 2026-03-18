from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset
import torch
import numpy as np
import corpus_loader
import config
import ontology_graph

# Load model and tokenizer
MODEL_NAME = "adsabs/KAILAS"
CORPUS_PATH = config.ADS_CORPUS_DIR
NUM_LABELS = 2372 #2411
THRESHOLD = 0.5
NUM_EPOCHS = 1

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(batch):
    tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    # tokens["labels"] = [list(map(float, build_multihot(l))) for l in batch["labels"]]
    tokens["labels"] = build_multihot(batch["labels"]).tolist()
    return tokens

onto = ontology_graph.OntologyGraph(config.CORPUS_DIR / "UAT_v6.0.0.rdf")
node2idx = onto.node2idx
idx2node = onto.idx2node
UAT_NAMESPACE = "http://astrothesaurus.org/uat/"

def build_multihot(
    annotations: List[List[str]],
) -> torch.Tensor:
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


def compute_pos_weight(train_labels: List[List[str]], node2idx: dict, num_labels: int) -> torch.Tensor:
    """Compute pos_weight = (nb négatifs) / (nb positifs) pour chaque label."""
    counts = torch.zeros(num_labels)
    n_docs = len(train_labels)
    for ann_list in train_labels:
        for uri in ann_list:
            if uri in node2idx:
                counts[node2idx[uri]] += 1.0
    # Évite la division par zéro pour les labels jamais vus
    counts = torch.clamp(counts, min=1.0)
    pos_weight = (n_docs - counts) / counts
    return pos_weight


class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight: torch.Tensor = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # shape (batch, NUM_LABELS) — pas de softmax

        loss_fn = torch.nn.BCEWithLogitsLoss(
            pos_weight=self.pos_weight.to(logits.device)
        )
        loss = loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=False,  # important if we keep the same amount of labels
    problem_type="multi_label_classification"
)

# Dataset loading
train_docs = []
train_labels = []

reader = corpus_loader.Reader()
for document, annotation in reader.read_corpus(ignore_kailas = False,
                                               corpus_folder = CORPUS_PATH):
    train_docs.append(document)
    train_labels.append(annotation)

train_data = {
    "text": train_docs,
    "labels": train_labels
}

train_dataset = Dataset.from_dict(train_data)
train_dataset = train_dataset.map(tokenize, batched=True)
train_dataset.set_format("torch")


validation_docs = []
validation_labels = []

for document, annotation in reader.read_pre9forADS():
    validation_docs.append(document)
    validation_labels.append(annotation)

validation_data = {
    "text": validation_docs,
    "labels": validation_labels
}
validation_dataset = Dataset.from_dict(validation_data)
validation_dataset = validation_dataset.map(tokenize, batched=True)
validation_dataset.set_format("torch")

# Training parameters
training_args = TrainingArguments(
    output_dir="./kailas-finetuned",
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=8,
    learning_rate=2e-4,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    eval_strategy="epoch",
    report_to="none",
    use_cpu = not torch.cuda.is_available(),  # True if no GPU
)


# Train
"""
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)
"""
pos_weight = compute_pos_weight(train_labels, node2idx, NUM_LABELS)
print(f"pos_weight min/max/mean: {pos_weight.min():.1f} / {pos_weight.max():.1f} / {pos_weight.mean():.1f}")

trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    pos_weight=pos_weight,
)

trainer.train()

# Save
model.save_pretrained("./kailas-finetuned")
tokenizer.save_pretrained("./kailas-finetuned")
print("✅ Fine-tuning done!")





# Test on Dataset validation
print(f"\n{'='*60}")
print(f"Test on {len(validation_dataset)} documents (threshold={THRESHOLD})")
print(f"{'='*60}\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval()
model.to(device)

all_probs = []  # pour diagnostiquer la distribution des scores

for i, sample in enumerate(validation_dataset):
    input_ids      = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    probs     = torch.sigmoid(logits).squeeze(0).cpu().numpy()
    all_probs.append(probs)

    pred_idxs  = np.where(probs >= THRESHOLD)[0]
    pred_nodes = [idx2node[idx] for idx in pred_idxs]

    true_idxs  = np.where(sample["labels"].numpy() == 1.0)[0]
    true_nodes = [idx2node[idx] for idx in true_idxs]

    print(f"── Document {i} ──────────────────────────────────────────")
    print(f"  Text      : {validation_docs[i][:120]}...")
    print(f"  probs max  : {probs.max():.4f}  mean: {probs.mean():.4f}")  # diagnostic
    print(f"  Preds   : {pred_nodes}")
    print(f"  True   : {true_nodes}")
    print()

# Diagnostic global : distribution des probs sur tout le dataset
all_probs = np.concatenate(all_probs)
print(f"── Diagnostic global ────────────────────────────────────────")
print(f"  probs mean  : {all_probs.mean():.4f}")
print(f"  probs max   : {all_probs.max():.4f}")
print(f"  probs > 0.1 : {(all_probs > 0.1).sum()}")
print(f"  probs > 0.3 : {(all_probs > 0.3).sum()}")
print(f"  probs > 0.5 : {(all_probs > 0.5).sum()}")