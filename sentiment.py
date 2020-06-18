"""client for the roberta sentiment model hosted on allennlp demo"""
import pickle
import re
import requests
import time
import os
from string import whitespace


def predict_sentiment(text: str, rtok=None, sleep_time=0.1):
    sanitized_text = re.sub(r"\<.*?\>", "", text)

    if all([c in whitespace for c in sanitized_text]):
        return None

    from transformers.tokenization_roberta import RobertaTokenizer
    if rtok is None:
        rtok = RobertaTokenizer.from_pretrained("roberta-large")
    sanitized_text = rtok.convert_tokens_to_string(rtok.tokenize(sanitized_text)[:200])  # empirical

    time.sleep(sleep_time)

    url = 'https://demo.allennlp.org/api/roberta-sentiment-analysis/predict'
    r=requests.post(url, json={"model":"RoBERTa","sentence":sanitized_text})

    if r.status_code != 200:
        return None

    data = r.json()
    return {"label": data["label"], "prob": max(data["probs"]), "logits": data["logits"]}


class SentimentCache:
    def __init__(self, path: str="sentiment_cache.pkl.gz", cache: dict=None):
        self.path = path
        self.cache = cache
        from transformers.tokenization_roberta import RobertaTokenizer
        self.rtok = RobertaTokenizer.from_pretrained("roberta-large")
        if self.cache is None:
            self.cache = {}

    def query(self, text: str, sleep_time: float=0.2):
        if text not in self.cache:
            response = predict_sentiment(text, rtok=self.rtok, sleep_time=sleep_time)
            if response is not None:
                self.cache[text] = response
            return response
        else:
            return self.cache[text]

    def save(self, verbose=True, do_backup=True):
        with open(self.path, "wb") as f:
            pickle.dump(self.cache, f)
        if do_backup:
            # TODO: better path handling
            with open(self.path[:-len(".pkl.gz")] + "_backup.pkl.gz", "wb") as f:
                pickle.dump(self.cache, f)
        if verbose:
            print(f"saved sentiment cache with length {len(self.cache)}")

    @staticmethod
    def load(path: str="sentiment_cache.pkl.gz", verbose=True) -> 'SentimentCache':
        cache = None
        if os.path.exists(path):
            with open(path, "rb") as f:
                cache = pickle.load(f)
            if verbose:
                print(f"loaded sentiment cache with length {len(cache)}")
        else:
            print(f"initialized response cache")
        return SentimentCache(path, cache)
