import time
import json
import urllib.parse
# from datetime import datetime

import requests

from util.error_handling import LogExceptionAndSkip
from response_cache import PostIdentifier   # ResponseCache, CachedResponseType


class RawNotification(dict):
    def __hash__(self):
        return hash(json.dumps(self))


def latest_note_ts(private_client, post_identifier: PostIdentifier):
    # TODO: caching
    n = private_client.notes(post_identifier.blog_name, id=post_identifier.id_)['notes']
    return max([nn.get('timestamp', -1) for nn in n])


def post_created_ts(private_client, post_identifier: PostIdentifier):
    # TODO: caching
    p = private_client.posts(post_identifier.blog_name, id=post_identifier.id_)['posts'][0]
    return p['timestamp']


# copy/paste from tumbl.py
# TODO: DRY
def check_notifications(private_client, blogName, n_to_check=250, after_ts=0, before_ts=None, dump_to_file=False):
    base_url = private_client.request.host + f"/v2/blog/{blogName}/notifications"
    if before_ts is not None:
        # TODO: verify this is compatible with pagination
        base_url = base_url + "?" + urllib.parse.urlencode({"before": before_ts})

    request_kwargs = dict(
        allow_redirects=False,
        headers=private_client.request.headers,
        auth=private_client.request.oauth,
    )

    getter = lambda url: requests.get(url, **request_kwargs).json()["response"]
    updater = lambda page: [
        item for item in page["notifications"] if item["timestamp"] > after_ts
    ]
    n = []

    with LogExceptionAndSkip("check notifications"):
        page = getter(base_url)
        delta = updater(page)

        while len(n) < n_to_check and len(delta) > 0:
            n += delta
            print(f"{len(n)}/{n_to_check}")
            time.sleep(0.1)
            url = private_client.request.host + page["_links"]["next"]["href"]
            page = getter(url)
            delta = updater(page)

    if dump_to_file:
        # for now, just make sure these are saved somehow
        # once i know how i'm using them, i'll set up something more formal w/ deduping etc
        with open("data/notification_dump.jsonl", "a", encoding="utf-8") as f:
            for nn in n:
                json.dump(nn, f)
                f.write('\n')

    return n
