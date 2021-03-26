import json
import time
from typing import Tuple, NamedTuple

import requests

from bridge_shared import bridge_service_unique_id, bridge_service_url
from util.util import typed_namedtuple_to_dict
from autoresponder_config import model_version, ckpt_select, ckpt_sentiment, ckpt_autoreviewer


ModelVersion = NamedTuple("ModelVersion",
                          model_identifier=str,
                          version_identifier=str)

LATEST_MODEL_VERSIONS = (
    ModelVersion('model_name', model_name),
    ModelVersion('ckpt_select', ckpt_select),
    ModelVersion('ckpt_sentiment', ckpt_sentiment),
    ModelVersion('ckpt_autoreviewer', ckpt_autoreviewer),
)

BridgeCacheKey = NamedTuple("BridgeCacheKey",
                            bridge_id=str,
                            model_versions=Tuple[ModelVersion, ...]
                            )


def make_bridge_cache_key(data: dict) -> BridgeCache:
    bridge_id = bridge_service_unique_id(data)
    return BridgeCacheKey(bridge_id=bridge_id, model_versions=LATEST_MODEL_VERSIONS)


class BridgeCache:
    def __init__(self, cache: dict):
        self.cache = cache

    def query(self, data: dict):
        key = make_bridge_cache_key(data)

        if key not in self.cache:
            response_data = self.call_bridge(data, bridge_id=key.bridge_id)
            self.cache[key] = response_data

        return self.cache[key]

    @staticmethod
    def call_bridge(self, data: dict, bridge_id: str):
        response_data = []
        while len(response_data) == 0:
            time.sleep(1)
            response_data = requests.post(
                bridge_service_url + "/getresult", data={"id": bridge_id}
            ).json()

        requests.post(bridge_service_url + "/done", json={"id": bridge_id})
        return response_data

    def save(self, path="data/bridge_cache.json"):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f)

    @staticmethod
    def load(path="data/bridge_cache.json") -> 'BridgeCache':
        if not os.path.exists(path):
            print("initializing bridge cache")
            return BridgeCache(cache=dict())

        with open(path, "r", encoding="utf-8") as f:
            cache = json.load(f)
        return BridgeCache(cache=cache)
