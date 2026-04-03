
import atexit
import json
import os
import requests
import hashlib
from src.utils.config import configure_ollama, OLLAMA_TEMPERATURE, BERT_MODEL
import src.utils.config

cache = dict()
loaded = False

def load_cache():
    global loaded
    if loaded:
        return
    responses_cache_file = config.CACHE_DIR / f"{config.OLLAMA_MODEL_NAME}-responses.json"
    if os.path.exists(responses_cache_file):
        with open(responses_cache_file, "r") as file:
            global cache
            cache = json.load(file)
    loaded = True
    atexit.register(save_cache)

def save_cache():
    global cache
    responses_cache_file = config.CACHE_DIR / f"{config.OLLAMA_MODEL_NAME}-responses.json"
    with open(responses_cache_file, "w") as file:
        json.dump(cache, file, indent = 2)

def get_cached(cache_key):
    global cache
    return cache.get(cache_key, None)

def save_to_cache(response, cache_key):
    global cache
    cache[cache_key] = response

def generate(prompt: str,
             num_predict: int = 256,
             cache_key: str = None) -> str:
    """
    Send a simple generate query to the Ollama API.

    Args:
        prompt: the prompt to send to the LLM.
        num_predict: maximum length of the predicted message.
        cache_key: key where to save the LLM's response in the cache.
                   By default, a key is generated from the prompt.
    """
    configure_ollama()
    load_cache()
    if not cache_key:
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    response_cache = get_cached(cache_key)
    if response_cache:
        return response_cache
        
    response = requests.post(
        f'{config.OLLAMA_HOST}/api/generate',
        json={
            'model': config.OLLAMA_MODEL_NAME,
            'prompt': prompt,
            'stream': False,
            'temperature': OLLAMA_TEMPERATURE,
            'num_predict': num_predict,
            "options": {
                "temperature": OLLAMA_TEMPERATURE,
                "num_predict": num_predict
            }
        }
    )
    if response.ok:
        response = response.json()['response'].strip()
        save_to_cache(response, cache_key)
        return response
    else:
        raise requests.ConnectionError(response.json()["error"])


def encode(text: list[str]):
    """
    Generate embeddings from the model.
    An embedding model might be used instead.
    """
    configure_ollama()
    response = requests.post(
        f'{config.OLLAMA_HOST}/api/embed',
        json={
            'model': BERT_MODEL,
            'input': text,
        }
    )
    return response.json()["embedding"]
    """
    response = requests.post(
        f'{config.OLLAMA_HOST}/api/generate',
        json={
            'model': config.OLLAMA_MODEL,#EMBEDDING_LLM,
            'input': text,
            'embedding': True
        }
    )

    response.raise_for_status()
    return response.json()["response"]
    """
