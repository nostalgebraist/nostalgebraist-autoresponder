import time
import json
import urllib.parse
from collections import defaultdict, Counter
# from datetime import datetime

import requests

from util.error_handling import LogExceptionAndSkip
from persistence.response_cache import PostIdentifier   # ResponseCache, CachedResponseType


class RawNotification(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sorted_keys = sorted(self.keys())
        self._hash = hash(
            str(
                [(k, self[k])
                 for k in
                 self._sorted_keys
                 ]
            )
        )

    def __hash__(self):
        return self._hash


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


def load_notification_dump(dedup=True):
    with open("data/notification_dump.jsonl", "r") as f:
        nots = [json.loads(line) for line in f]

    if dedup:
        nots = list({json.dumps(no): no for no in nots}.values())

    return nots


# note: this was unnecessary
# TODO: why did i think this was necessary??
def collect_naked_reblogs(nots):
    post_id_to_equivalent_post_id = {}

    for no in nots:
        if no['type'] == 'reblog_naked':
            post_id_to_equivalent_post_id[no['post_id']] = no['target_post_id']

    return post_id_to_equivalent_post_id


def assign_to_targets(nots, blogName, valid_only=True):
    post_id_to_notifications = defaultdict(set)

    post_id_to_equivalent_post_id = collect_naked_reblogs(nots)

    n_per_route = Counter()
    blognames = Counter()
    for no in nots:
        target_post_id = no.get('target_post_id')
        target_post_id = post_id_to_equivalent_post_id.get(target_post_id, target_post_id)
        if target_post_id:
            blognames[no.get('target_tumblelog_name')] += 1
            if no.get('target_tumblelog_name') == blogName:
                n_per_route['direct'] += 1
                post_id_to_notifications[target_post_id].add(RawNotification(**no))
            elif target_post_id in post_id_to_equivalent_post_id:
                n_per_route['indirect'] += 1
                target_post_id = post_id_to_equivalent_post_id[target_post_id]
                post_id_to_notifications[target_post_id].add(RawNotification(**no))
            else:
                print(no)

    print(n_per_route)
    print(blognames)
    # TODO: make this work?  it's too costly without caching
    #
    # if valid_only:
    #     min_ts =
    #     post_id_to_notifications = {pid: v for pid, v in post_id_to_notifications.items()
    #                                 if }

    return post_id_to_notifications
