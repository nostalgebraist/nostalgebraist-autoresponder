import pickle
import re
import requests
import time
import os
from string import whitespace
from datetime import datetime, timedelta

from tqdm import tqdm

from mood import logit_diff_to_pos_sent
from experimental.ml_connector import side_judgments_from_gpt2_service

SELECT_VIA_GENERATOR = True
SENTIMENT_VIA_GENERATOR = True


def get_multi_side_judgments(
    texts: list,
    v8_timestamps=None,
    v10_timestamps=None,
    wait_first_time=1.5,
    wait_recheck_time=1,
    use_allen_payload_schema=True,
):
    response = side_judgments_from_gpt2_service(
        texts,
        v8_timestamps=v8_timestamps,
        v10_timestamps=v10_timestamps,
        wait_first_time=wait_first_time,
        wait_recheck_time=wait_recheck_time,
    )
    result = [
        munge_side_judgment_payload(
            response, extract_ix=ix, use_allen_payload_schema=use_allen_payload_schema
        )
        for ix in range(len(texts))
    ]
    return result


def get_side_judgments(
    text: str,
    v8_timestamp=None,
    v10_timestamp=None,
    wait_first_time=1.5,
    wait_recheck_time=1,
    use_allen_payload_schema=True,
):
    return get_multi_side_judgments(
        [text],
        v8_timestamps=None if v8_timestamp is None else [v8_timestamp],
        v10_timestamps=None if v10_timestamp is None else [v10_timestamp],
        wait_first_time=wait_first_time,
        wait_recheck_time=wait_recheck_time,
        use_allen_payload_schema=use_allen_payload_schema,
    )[0]


def munge_side_judgment_payload(response, extract_ix, use_allen_payload_schema=True):
    result = {
        "selection_proba": response["selection_proba"][extract_ix],
        "sentiment": None,
    }
    if use_allen_payload_schema:  # stuff like reversed logits etc.
        if response["sentiment_logit_diffs"] is not None:
            allen_schema = logit_diff_to_allen_schema(
                response["sentiment_logit_diffs"][extract_ix]
            )
            result["sentiment"] = {
                "raw": response["sentiment_logit_diffs"][extract_ix],
                "allen_schema": allen_schema,
                "obtaine_via": "generator",
            }
    else:
        raise ValueError("not implemented yet")
    return result


def logit_diff_to_allen_schema(logit_diff: float):
    label = "1" if logit_diff > 0 else "0"

    pos_logit = logit_diff / 2
    neg_logit = -1 * logit_diff / 2
    logits = [pos_logit, neg_logit]

    prob_pos = logit_diff_to_pos_sent(logit_diff)
    prob = prob_pos if logit_diff > 0 else (1.0 - prob_pos)

    entry = {"label": label, "prob": prob, "logits": logits}
    return entry


def predict_sentiment_legacy(text: str, rtok=None, sleep_time=0.1):
    sanitized_text = re.sub(r"\<.*?\>", "", text)

    if all([c in whitespace for c in sanitized_text]):
        return None

    from transformers.tokenization_roberta import RobertaTokenizer

    if rtok is None:
        rtok = RobertaTokenizer.from_pretrained("roberta-large")
    sanitized_text = rtok.convert_tokens_to_string(
        rtok.tokenize(sanitized_text)[:200]
    )  # empirical

    time.sleep(sleep_time)

    url = "https://demo.allennlp.org/api/roberta-sentiment-analysis/predict"
    r = requests.post(url, json={"model": "RoBERTa", "sentence": sanitized_text})

    if r.status_code != 200:
        return None

    data = r.json()
    return {
        "label": data["label"],
        "prob": max(data["probs"]),
        "logits": data["logits"],
    }


class SideJudgmentCache:
    def __init__(
        self, path: str = "data/side_judgment_cache.pkl.gz", cache: dict = None
    ):
        self.path = path
        self.cache = cache
        # from transformers.tokenization_roberta import RobertaTokenizer
        # self.rtok = RobertaTokenizer.from_pretrained("roberta-large")
        self.rtok = None
        if self.cache is None:
            self.cache = {}

    def remove_oldest(self, max_hours=12, dryrun=False):
        last_allowed_time = datetime.now() - timedelta(hours=max_hours)

        existing_cache = self.cache

        allowed_keys = {
            k
            for k, v in existing_cache.items()
            if "last_accessed_time" in v
            and v["last_accessed_time"] >= last_allowed_time
        }

        new_cache = {
            k: existing_cache[k] for k in existing_cache.keys() if k in allowed_keys
        }

        before_len = len(existing_cache)
        delta_len = before_len - len(new_cache)

        if dryrun:
            print(
                f"remove_oldest: would drop {delta_len} of {before_len} side judgments"
            )
        else:
            print(f"remove_oldest: dropping {delta_len} of {before_len} side judgments")
            self.cache = new_cache

    def record(self, text: str, payload: dict):
        self.cache[text] = munge_side_judgment_payload(payload, extract_ix=0)

    def query(
        self, text: str, v8_timestamp=None, v10_timestamp=None, sleep_time: float = 0.2
    ):
        if text not in self.cache:
            response = get_side_judgments(
                text, v8_timestamp=v8_timestamp, v10_timestamp=v10_timestamp
            )
            if response is None:
                return response
            if response["sentiment"] is None:
                # fallback to allen
                if SELECT_VIA_GENERATOR:
                    print(f"SELECT_VIA_GENERATOR FAILURE ON {text}")
                if self.rtok is None:
                    from transformers.tokenization_roberta import RobertaTokenizer

                    self.rtok = RobertaTokenizer.from_pretrained("roberta-large")
                allen_response = predict_sentiment_legacy(
                    text, rtok=self.rtok, sleep_time=sleep_time
                )
                if allen_response is not None:
                    response["sentiment"] = {
                        "raw": allen_response,
                        "allen_schema": allen_response,
                        "obtaine_via": "allen",
                    }

            if (
                response["selection_proba"] is not None
                and response["sentiment"] is not None
            ):
                self.cache[text] = response
                self.cache[text]["last_accessed_time"] = datetime.now()
            return response
        else:
            self.cache[text]["last_accessed_time"] = datetime.now()
            return self.cache[text]

    def query_multi(
        self,
        texts: list,
        v8_timestamps=None,
        v10_timestamps=None,
        sleep_time: float = 0.2,
        batch_size: int = 4,
        verbose=True,
        progbar=True,
    ):
        now = datetime.now()

        results = [self.cache.get(t, None) for t in texts]
        fill_ixs = [ix for ix, r in enumerate(results) if r is None]

        if verbose:
            print(f"query_multi:\n\tfill_ixs={fill_ixs}\n")

        batch_ixs = [
            fill_ixs[j : j + batch_size] for j in range(0, len(fill_ixs), batch_size)
        ]
        if len(batch_ixs) * batch_size < len(fill_ixs):
            batch_ixs.append(fill_ixs[len(fill_ixs) // batch_size :])
        if verbose:
            print(f"query_multi:\n\tbatch_ixs={batch_ixs}\n")
        batch_texts = [[texts[ix] for ix in b] for b in batch_ixs]
        if verbose:
            print(f"query_multi:\n\tbatch_texts={batch_texts}\n")
        batch_responses = [
            get_multi_side_judgments(
                bt, v8_timestamps=v8_timestamps, v10_timestamps=v10_timestamps
            )
            for bt in batch_texts
        ]
        if verbose:
            print(f"query_multi:\n\tbatch_responses={batch_responses}\n")

        iter_ = zip(batch_ixs, batch_responses, batch_texts)

        if progbar:
            iter_ = tqdm(iter_, total=len(batch_ixs), unit="judg")

        for b, br, bt in iter_:
            for ix, r, t in zip(b, br, bt):
                if verbose:
                    print(f"query_multi:\n\tfilling {ix}:\n\t{t}\nwith\n\t{r}\n")
                results[ix] = r
                self.cache[t] = r
                self.cache[t]["last_accessed_time"] = now
        return results

    def save(self, verbose=True, do_backup=True):
        self.remove_oldest()
        with open(self.path, "wb") as f:
            pickle.dump(self.cache, f)
        if do_backup:
            # TODO: better path handling
            with open(self.path[: -len(".pkl.gz")] + "_backup.pkl.gz", "wb") as f:
                pickle.dump(self.cache, f)
        if verbose:
            print(f"saved side judgment cache with length {len(self.cache)}")

    @staticmethod
    def load(
        path: str = "data/side_judgment_cache.pkl.gz", verbose=True
    ) -> "SideJudgmentCache":
        cache = None
        if os.path.exists(path):
            with open(path, "rb") as f:
                cache = pickle.load(f)
            if verbose:
                print(f"loaded side judgment cache with length {len(cache)}")
        else:
            print(f"initialized side judgment cache")
        loaded = SideJudgmentCache(path, cache)
        loaded.remove_oldest()
        return loaded
