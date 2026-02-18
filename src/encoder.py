
import torch
### AstroLlama
from transformers import AutoConfig, AutoModelForCausalLM
from transformers import AutoTokenizer
from config import SENTENCE_TRANSFORMERS_MODEL, ENCODER_MAX_LENGTH

def encode_batch(texts: list[str]):
    """
    Get the encoded tensors of the entities' textual informations.
    Those informations include the label, alternate labels, definition,
    location...
    Return a list of tensors for each entity.

    Keyword arguments:
    texts -- the list of entities' string representations to encode
    """
    ### Astrollama ###
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # https://huggingface.co/UniverseTBD/astrollama
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path = SENTENCE_TRANSFORMERS_MODEL,
        use_fast = False
        # device=device
    )
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=SENTENCE_TRANSFORMERS_MODEL,
        # device_map="auto",
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path = SENTENCE_TRANSFORMERS_MODEL),
        use_safetensors = True,
        trust_remote_code = True,
        # load_in_4bit = True,
        torch_dtype = torch.bfloat16,
        # device = device
        )
    print(f"Encoding the entities with {model}")
    # no need to normalize embeddings as we compute a cosine similarity.
    inputs = tokenizer(texts,
                       return_tensors = "pt",
                       return_token_type_ids = False,
                       padding = True,
                       truncation = True,
                       max_length = ENCODER_MAX_LENGTH
                      )
    inputs.to(model.device)
    return model(**inputs, output_hidden_states = True)
    """
    # Bert:
    return CosineSimilarityScorer.model.encode(texts,
                                               batch_size = BATCH_SIZE,
                                               show_progress_bar = True,
                                               convert_to_tensor = True,
                                               normalize_embeddings = False)
    """
