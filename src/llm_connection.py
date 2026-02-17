
import atexit
import json
import pickle
import re
import os
import requests
import config
from config import OLLAMA_TEMPERATURE
from collections import defaultdict
import hashlib

cache = dict()
loaded = False

def load_cache():
    if loaded:
        return
    responses_cache_file = config.CACHE_DIR / f"{config.OLLAMA_MODEL_NAME}-responses.json"
    if os.path.exists(responses_cache_file):
        with open(responses_cache_file, "r") as file:
            global cache
            cache = json.load(file)
    atexit.register(save_cache)

def save_cache():
    responses_cache_file = config.CACHE_DIR / f"{config.OLLAMA_MODEL_NAME}-responses.json"
    with open(responses_cache_file, "w") as file:
        json.dump(cache, file, indent = 2)

def get_cached(prompt):
    global cache
    return cache.get(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), None)

def save_to_cache(prompt, response):
    global cache
    cache[hashlib.sha256(prompt.encode("utf-8")).hexdigest()] = response

def generate(prompt: str,
             num_predict: int = 256) -> str:

    """
    Send a simple generate query to the Ollama API.

    Args:
        prompt: the prompt to send to the LLM.
        num_predict: maximum length of the predicted message.
    """
    load_cache()
    response_cache = get_cached(prompt)
    if response_cache:
        return response_cache
        
    response = requests.post(
        f'{config.OLLAMA_HOST}/api/generate',
        json={
            'model': config.OLLAMA_MODEL,
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
        save_to_cache(prompt, response)
        return response
    else:
        raise requests.ConnectionError(response.json()["error"])
