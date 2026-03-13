"""
Configuration script. Use to change the project's parameters.

Author:
    Liza Fretel (liza.fretel@obspm.fr)
"""
from pathlib import Path
import os

USERNAME = os.environ.get("USER") or os.environ.get("USERNAME") or os.getlogin()

# directories
ROOT = Path(__file__).parent.parent.parent
CONF_DIR = ROOT / "conf"
CORPUS_DIR = ROOT / "corpus"
OUTPUT_DIR = ROOT / "src" / "KAILAS" / "output"
if "SSH_CONNECTION" in os.environ or "SSH_CLIENT" in os.environ:
    # tycho
    CACHE_DIR = Path("/scratch") / USERNAME / "cache"
    DATA_DIR = Path("/scratch2") / USERNAME / "cache"
    # TMP_DIR = Path("/scratch") / USERNAME / "tmp"
else:
    # local
    CACHE_DIR = ROOT / "cache"
    DATA_DIR = ROOT / "corpus"
    # TMP_DIR = ROOT / "tmp"

# mkdir
DATA_DIR.mkdir(parents = True, exist_ok = True)
CACHE_DIR.mkdir(parents = True, exist_ok = True)
# TMP_DIR.mkdir(parents = True, exist_ok = True)

# HuggingFace, sentence transformers environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["HF_HOME"] = str(CACHE_DIR / "huggingface" ) # Must import before transformers

# File for saving the UATs in json format
UATS_JSON = CORPUS_DIR / "output.json"
UATS_LABELS_JSON = CORPUS_DIR / "uats_labels.json"

BERT_MODEL = "adsabs/astroBERT" # "UniverseTBD/astrollama"
ENCODER_MAX_LENGTH = 512

# File for saving the UAT's embeddings of UAT nodes
BERT_UATS_EMBEDDINGS_FILE = CACHE_DIR / f"embeddings_{BERT_MODEL.replace('/', '-')}_uat_nodes.pkl"
# File for saving the UAT's embeddings of documents
BERT_DOCS_EMBEDDINGS_FILE = CACHE_DIR / f"embeddings_{BERT_MODEL.replace('/', '-')}_documents.pkl"

# Folder for saving the ADS HelioPhysics corpus
ADS_HELIO_CORPUS_DIR = DATA_DIR / "ADS_HelioPhysics_corpus"
ADS_HELIO_CORPUS_DIR.mkdir(parents = True, exist_ok = True)
ADS_CORPUS_DIR = DATA_DIR / "ADS_corpus"
ADS_CORPUS_DIR.mkdir(parents = True, exist_ok = True)

# Our data file
TEST_CORPUS_FILE = CORPUS_DIR / "pre9forADS_all_annotated.dat"

# tau (threshold for selection of candidate labels)
THRESHOLD = 0.1