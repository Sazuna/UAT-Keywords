
from transformers import pipeline
from transformers import AutoTokenizer

MODEL    = "adsabs/KAILAS"
tokenizer = AutoTokenizer.from_pretrained(MODEL)

classifier = pipeline("text-classification", model = MODEL, tokenizer = tokenizer)#, return_all_scores = True)
# zero_shot_classifier = pipeline("zero-shot-classification", model = "adsabs/KAILAS", tokenizer = tokenizer)
