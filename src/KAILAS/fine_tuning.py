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
NUM_LABELS = 2411

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
def tokenize(batch):
    tokens = tokenizer(batch["text"], truncation=True, padding="max_length", max_length=512)
    # tokens["labels"] = [list(map(float, build_multihot(l))) for l in batch["labels"]]
    tokens["labels"] = build_multihot(batch["labels"]).tolist()
    return tokens

onto = ontology_graph.OntologyGraph(config.CORPUS_DIR / "UAT_v6.0.0.rdf")
node2idx = onto.node2idx
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


model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    ignore_mismatched_sizes=True,  # important si vous changez le nb de labels
    problem_type="multi_label_classification"
)

# Dataset loading
train_data = {
    "text": ["Document 1...", "Document 2..."],
    "labels": [[1, 0, 1, ...], [0, 1, 0, ...]]
}

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
    num_train_epochs=3,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    weight_decay=0.01,
    save_strategy="epoch",
    logging_strategy="steps",
    logging_steps=50,
    eval_strategy="epoch",
    report_to="none",
    use_cpu = not torch.cuda.is_available(),  # True if no GPU
)


# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
)

trainer.train()

# Save
model.save_pretrained("./kailas-finetuned")
tokenizer.save_pretrained("./kailas-finetuned")
print("✅ Fine-tuning terminé !")