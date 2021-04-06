import os
import json
import time
from typing import Tuple, NamedTuple

import requests

from bridge_shared import bridge_service_unique_id, bridge_service_url
from autoresponder_config import (
    model_name,
    ckpt_select,
    ckpt_sentiment,
    ckpt_autoreviewer,
)


ModelVersion = NamedTuple("ModelVersion", model_identifier=str, version_identifier=str)

LATEST_MODEL_VERSIONS = (
    ModelVersion("model_name", model_name),
    ModelVersion("ckpt_select", ckpt_select),
    ModelVersion("ckpt_sentiment", ckpt_sentiment),
    ModelVersion("ckpt_autoreviewer", ckpt_autoreviewer),
)

BridgeCacheKey = NamedTuple(
    "BridgeCacheKey", bridge_id=str, model_versions=Tuple[ModelVersion, ...]
)


def make_bridge_cache_key(data: dict) -> BridgeCacheKey:
    bridge_id = bridge_service_unique_id(bridge_service_url, data)
    return BridgeCacheKey(bridge_id=bridge_id, model_versions=LATEST_MODEL_VERSIONS)


class BridgeCache:
    def __init__(self, cache: dict):
        self.cache = cache

    def query(self, data: dict):
        key = make_bridge_cache_key(data)

        if key not in self.cache:
            response_data = self.call_bridge(data, bridge_id=key.bridge_id)

            true_key = self.key_from_response_data(
                bridge_id=key.bridge_id, response_data=response_data
            )
            key = true_key

            self.cache[key] = response_data

        return self.cache[key]

    @staticmethod
    def key_from_response_data(response_data: dict, bridge_id: str):
        entry = response_data[0]
        model_info = entry["model_info"]

        model_versions = (
            ModelVersion("model_name", model_info.get("model_name", "")),
            ModelVersion("ckpt_select", model_info.get("ckpt_select", "")),
            ModelVersion("ckpt_sentiment", model_info.get("ckpt_sentiment", "")),
            ModelVersion("ckpt_autoreviewer", model_info.get("ckpt_autoreviewer", "")),
        )

        return BridgeCacheKey(bridge_id=bridge_id, model_versions=model_versions)

    @staticmethod
    def call_bridge(data: dict, bridge_id: str):
        data_to_send = dict()
        data_to_send.update(data)
        data_to_send["id"] = bridge_id

        requests.post(bridge_service_url + "/requestml", json=data_to_send)

        response_data = []
        while len(response_data) == 0:
            time.sleep(1)
            response_data = requests.post(
                bridge_service_url + "/getresult", data={"id": bridge_id}
            ).json()

        requests.post(bridge_service_url + "/done", json={"id": bridge_id})
        return response_data

    def to_json(self):
        return [{"key": k, "value": v} for k, v in self.cache.items()]

    @staticmethod
    def from_json(entries: list) -> "BridgeCache":
        def _parse_key(key_json: list) -> BridgeCacheKey:
            bridge_id = key_json[0]
            model_versions = tuple(
                [
                    ModelVersion(model_identifier=mv[0], version_identifier=mv[1])
                    for mv in key_json[1]
                ]
            )
            return BridgeCacheKey(bridge_id=bridge_id, model_versions=model_versions)

        cache = {_parse_key(entry["key"]): entry["value"] for entry in entries}
        return BridgeCache(cache=cache)

    def save(self, path="data/bridge_cache.json"):
        t1 = time.time()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json(), f)
        delta = time.time() - t1
        print(f"saved bridge cache with length {len(self.cache)} in {delta:.2f}s")

    @staticmethod
    def load(path="data/bridge_cache.json") -> "BridgeCache":
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    entries = json.load(f)
                return BridgeCache.from_json(entries)
            except json.decoder.JSONDecodeError:
                print("cache file found, but could not load it")

        print("initializing bridge cache")
        return BridgeCache(cache=dict())
