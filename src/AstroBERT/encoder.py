import torch
from transformers import BertModel, BertTokenizer
from src.utils.config import BERT_MODEL, ENCODER_MAX_LENGTH
import numpy as np

tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
model = BertModel.from_pretrained(BERT_MODEL)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)
model.eval()

def astrobert_encode(texts: list[str]) -> np.ndarray:
    """
    Encode a list of texts into embeddings using a BERT model.

    Args:
        texts (list[str]): list of input strings.

    Returns:
        np.ndarray: Array of shape (len(texts, hidden_size) with embeddings.
    """
    batch_size = 32
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(batch,
                            padding = True,
                            truncation = True,
                            return_tensors = "pt",
                            max_length = ENCODER_MAX_LENGTH)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            last_hidden_states = outputs.last_hidden_state
            batch_embeddings = last_hidden_states[:, 0, :]
            batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.cpu().numpy())
    all_embeddings = np.concatenate(all_embeddings, axis = 0)
    return all_embeddings

