"""
Configuration script. Use to change the project's parameters.

Author:
    Liza Fretel (liza.fretel@obspm.fr)
"""
from pathlib import Path
import os
import asyncio

USERNAME = os.environ.get("USER") or os.environ.get("USERNAME") or os.getlogin()

# directories
ROOT = Path(__file__).parent.parent.parent
CONF_DIR = ROOT / "conf"
CORPUS_DIR = ROOT / "corpus"
OUTPUT_DIR = ROOT / "src" / "KAILAS" / "output"
if "SSH_CONNECTION" in os.environ or "SSH_CLIENT" in os.environ:
    # tycho
    CACHE_DIR = Path("/scratch2") / USERNAME / "cache"
    DATA_DIR = Path("/data") / USERNAME / "cache"
else:
    # local
    CACHE_DIR = ROOT / "cache"
    DATA_DIR = ROOT / "corpus"

# mkdir
DATA_DIR.mkdir(parents = True, exist_ok = True)
CACHE_DIR.mkdir(parents = True, exist_ok = True)

# HuggingFace, sentence transformers environment variables
##os.environ["TOKENIZERS_PARALLELISM"] = "false"
##os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
##os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface" ) # Must import before transformers

# File for saving the UATs in json format
UATS_JSON = CORPUS_DIR / "output.json"
UATS_JSON_VERBALIZED = CORPUS_DIR / "output_verbalized.json"
UATS_LABELS_JSON = CORPUS_DIR / "uats_labels.json"

BERT_MODEL = "adsabs/astroBERT" # "UniverseTBD/astrollama"
ENCODER_MAX_LENGTH = 512

# File for saving the UAT's embeddings
BERT_UATS_EMBEDDINGS_FILE = CACHE_DIR / f"embeddings_{BERT_MODEL.replace('/', '-')}"
BERT_UATS_EMBEDDINGS_FILE_VERBALIZED = CACHE_DIR / f"embeddings_{BERT_MODEL.replace('/', '-')}_verbalized"
# File for saving the UAT's embeddings of documents
BERT_DOCS_EMBEDDINGS_FILE = CACHE_DIR / f"embeddings_{BERT_MODEL.replace('/', '-')}_documents.pkl"

# Folder for saving the ADS HelioPhysics corpus
ADS_HELIO_CORPUS_DIR = CACHE_DIR / "ADS_HelioPhysics_corpus"
ADS_HELIO_CORPUS_DIR.mkdir(parents = True, exist_ok = True)
ADS_CORPUS_DIR = CACHE_DIR / "ADS_corpus"
ADS_CORPUS_DIR.mkdir(parents = True, exist_ok = True)

# Our data file
TEST_CORPUS_FILE = CORPUS_DIR / "pre9forADS_all_updated.dat" #"pre9forADS_all_annotated.dat"
UATS_RDF_V6 = CORPUS_DIR / "UAT_v6.0.0.rdf"
UATS_RDF_V5 = CORPUS_DIR / "UAT_v5.1.0.rdf"

### Ollama configuration

# Ollama Configuration
async def connect_to_ollama():
    """
    Selects ollama models to use depending on the device.
    This function can be modified depending on the models and
    devices (ssh remote / local) that you are using.

    The function should be called once before LLM calls to prevent
    connecting to LLM in a non-LLM run (like during an update).
    """
    global OLLAMA_HOST
    global OLLAMA_MODEL
    global OLLAMA_MODEL_NAME
    global SUMMARIZE_MODEL
    if "SSH_CONNECTION" in os.environ or "SSH_CLIENT" in os.environ:
        OLLAMA_HOST = os.environ["OLLAMA_HOST"]
        if "127.0.0.1" in OLLAMA_HOST:
            # tycho91
            OLLAMA_MODEL = "phi4:latest" # 14GB
            OLLAMA_MODEL_NAME = "phi4:14b"
            SUMMARIZE_MODEL = "phi4:latest"
            CONNECTION_MODE = "tycho91 ollama"
        else:
            # another tycho
            if not OLLAMA_HOST:
                print("OLLAMA_HOST is not set. Please add to your ~/.bashrc:")
                print("export OLLAMA_HOST=\"http://{armstrong_IPV4}:11434\"")
                raise EnvironmentError("OLLAMA_HOST not set")
            OLLAMA_MODEL = "deepseek-v3:latest" # 400 GB (~12s)
            OLLAMA_MODEL_NAME = "DeepSeek-v3:671b"
            SUMMARIZE_MODEL = "deepseek-v3:latest"
            CONNECTION_MODE = "armstrong ollama"
    else:
        # local
        port = 11434
        OLLAMA_HOST = f"http://localhost:{port}"
        OLLAMA_MODEL = "phi4:latest"#"gemma3:12b"#"orca2:7b"#"ministral-3:14b"
        OLLAMA_MODEL_NAME = "phi4:14b"#"gemma3:12b"#"orca2:7b"#"ministral-3:14b"
        SUMMARIZE_MODEL = "phi4:latest"
        CONNECTION_MODE = "local ollama"
    return OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_MODEL_NAME, SUMMARIZE_MODEL, CONNECTION_MODE

OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_MODEL_NAME, SUMMARIZE_MODEL = None, None, None, None
def configure_ollama():
    global OLLAMA_HOST
    if OLLAMA_HOST is not None:
        # Only configure once
        return
    global OLLAMA_MODEL
    global OLLAMA_MODEL_NAME
    global SUMMARIZE_MODEL
    OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_MODEL_NAME, SUMMARIZE_MODEL, CONNECTION_MODE = asyncio.run(connect_to_ollama())
    print(f"Connected to {CONNECTION_MODE}. Using model {OLLAMA_MODEL}")

OLLAMA_TEMPERATURE = 0 # Higher temperature = less determinist
