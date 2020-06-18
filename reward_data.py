"""
Tools for scraping note counts as training data for the selector model.

Really shouldn't be called "reward"...

I run scrape_new_and_save manually once per ~7 days, then retrain selector model
"""
import os, textwrap, time, pickle

from bs4 import BeautifulSoup
from reblogs_v4 import *

from ratelimit_util import RateLimitClient
from response_cache import ResponseCache, PostIdentifier, CachedResponseType

# TODO: DRY
blogName = "nostalgebraist-autoresponder"

# TODO: define this only once (copy/paste from tumbl.py)
def _get_body(post):
    body_keys = ["body", "answer"]
    for k in body_keys:
        body = post.get(k, None)
        if body is not None:
            break
    if body is None:
        print(f"couldn't find body, tried keys {body_keys}, available keys are {post.keys()}")
    return body

def _get_reblog_data_body(post):
    tree_html, comment = post['reblog']['tree_html'], post['reblog']['comment']
    return tree_html, comment


def process_body(post):
    tree_html, comment = _get_reblog_data_body(post)
    if len(tree_html) == 0:
        body = _get_body(post)
    else:
        body = tree_html

    if post['type'] == "answer":
        body = f"<h2>{post['asking_name']} asked: {post['question']}</h2>" + body

    soup = BeautifulSoup(body, features='lxml')
    processed, post_metadata = process_post(soup, use_article=False)

    processed = processed.rstrip("<|endoftext|>")
    if len(tree_html) > 0:
        processed = processed + UNAME_CHAR + "nostalgebraist-autoresponder" + A_CHAR + comment
    return processed, post_metadata


def _autoresponse_to_prompt_and_continuation(processed):
    prompt = processed[::-1][processed[::-1].index(A_CHAR):][::-1]
    continuation = processed[::-1][:processed[::-1].index(A_CHAR)][::-1]
    return prompt, continuation


def get_prompt_and_continuation(post):
    processed, post_metadata = process_body(post)

    if post_metadata['is_reblog'] or post_metadata['is_ask']:
        prompt, continuation = _autoresponse_to_prompt_and_continuation(processed)
    elif post_metadata['is_orig']:
        # TODO: undo tight coupling to orig post making code
        prompt = "endoftext|>" + ORIG_POST_CHAR
        continuation = processed[processed.index(ORIG_POST_CHAR)+1:]
    else:
        raise ValueError(f"don't know how to deal with post")

    continuation += (T_CHAR + " ".join([f"#{tag}" for tag in post.get("tags", [])]) + "<|")

    return prompt, continuation


def show_response(ex: dict):
    for k, v in ex.items():
        print(k)
        print("\t" + "\n\t".join(textwrap.wrap(repr(v), width=100)))


def _review_extraction(posts):
    for ix in range(len(posts)):
         print(f"{ix}/{len(posts)}\n")
         ex = posts[ix]
         print(ex['id'])
         print(f"note_count: {ex.get('note_count')}")
         prompt, continuation = get_prompt_and_continuation(ex)
         data = {"prompt": prompt.replace("\n", "\\n"), "continuation": continuation.replace("\n", "\\n")}
         show_response(data)
         print("\n\n~~~~~~~~\n\n")


# count notes for posts --> reward

# TODO: delete this
def _find_previous_reblog_in_trail(post):
    # deprecated
    #
    # to attribute notes in a reblog chain, we look for earlier reblogs from AR, and only count notes between this one and that one
    prev_reblog_id = None

    prev_reblogs = post['trail'][:-1] # don't count _this_ one as "previous"
    for r in prev_reblogs:
        if r['blog']['name'] == blogName:
            prev_reblog_id = r['post']['id']

    return prev_reblog_id

def full_notes(response_cache, post):
    post_id = str(post['id'])
    time.sleep(0.33)
    post_identifier = PostIdentifier(blogName, post_id)
    notes = response_cache.query(CachedResponseType.NOTES, post_identifier, expected_notes=post['note_count'])
    return notes


def notes_since_last_reblog(response_cache, post, verbose=False):
    post_id = str(post['id'])
    notes = full_notes(response_cache, post)

    # notes come in descending temporal order, reverse to unconfuse myself
    notes_ascending = notes[::-1]
    # find index of *this* reblog
    this_ix = [ix for ix, n in enumerate(notes_ascending)
               if (n.get("post_id") == post_id) or
               n['type'] == "posted"][-1]
    other_reblog_ixs = [ix for ix, n in enumerate(notes_ascending)
                        if ix > this_ix and
                        n['type'] == 'reblog' and
                        n['blog_name'] == blogName]

    start_ix = this_ix+1 # don't count this as a note
    stop_ix = len(notes_ascending) # default
    if len(other_reblog_ixs) > 0:
        stop_ix = min(other_reblog_ixs)

    if verbose:
        print((this_ix, start_ix, stop_ix, len(notes_ascending)))

    subset_notes = notes_ascending[start_ix:stop_ix]
    return subset_notes


def count_notes_for_post(response_cache, post):
    return len(notes_since_last_reblog(response_cache, post))


def scrape_note_counts(client, max_to_scrape=None, offset=0, stop_on_seen_id=False,  ids_to_reward_data=dict(),
                       ignore_reblogs_from_dash=True):
    ratelimit_client = RateLimitClient.from_tumblr_rest_client(client, blogName=blogName)
    response_cache = ResponseCache.load(client)

    if max_to_scrape is None:
        max_to_scrape = 1e10

    def _on_new_post_batch(posts, ids_to_reward_data):
        for p in posts:
            try:
                post_id = p['id']
                prompt, continuation = get_prompt_and_continuation(p)
                note_count = count_notes_for_post(response_cache, p)

                if post_id in ids_to_reward_data:
                    existing_note_count = ids_to_reward_data[post_id].get("note_count")
                    if existing_note_count != note_count:
                        print(f"updating note_count from {existing_note_count} to {note_count} for {post_id}")

                ids_to_reward_data[post_id] = {"prompt": prompt,
                                               "continuation": continuation,
                                               "note_count": note_count}

            except Exception as e:
                print(f"hit {e} on {p['id']}")
        return ids_to_reward_data

    def _reward_post_screener(posts):
        screened = [p for p in posts]

        # ignore #quotes (very old behavior)
        screened = [p for p in screened if "quotes" not in p.get("tags", [])]

        # ignore mood graphs
        screened = [p for p in screened if "mood" not in p.get("caption", "")]

        if ignore_reblogs_from_dash:
            screened = [p for p in screened if p.get('source_title', blogName) == blogName]

        if len(screened) < len(posts):
            print(f"Ignoring {len(posts)-len(screened)} of {len(posts)} posts")
        return screened

    # scraping routine
    count_check_requests_start = ratelimit_client.get_ratelimit_data()['day']['remaining']
    limit_ = 15
    offset_ = offset
    posts = []
    next_ = response_cache.client.posts(blogName, limit=limit_, offset=offset_)["posts"]
    seen_id_count = 0
    new_id_count = None
    starting_ids = set(ids_to_reward_data.keys())

    while (len(next_) != 0) and len(posts) < max_to_scrape and (seen_id_count!=new_id_count or (not stop_on_seen_id)):
        print("\n")
        posts.extend(next_)
        print(f"got {len(next_)}, starting with {next_[0]['id']}")

        offset_ += len(next_)

        new_post_batch = _reward_post_screener(next_)
        ids_to_reward_data = _on_new_post_batch(new_post_batch, ids_to_reward_data)

        new_ids = {p['id'] for p in new_post_batch}
        new_id_count = len(new_ids)
        seen_id_count = len(new_ids.intersection(starting_ids))

        if len(new_ids) > 0 and len(starting_ids) > 0:
            print(f"seen_id_count: {seen_id_count}, new_id_count: {new_id_count}")
            if (seen_id_count > 0) and stop_on_seen_id:
                print(f"ids {new_ids.intersection(starting_ids)} ({seen_id_count} ids) in both new and starting sets")

        count_check_requests_end = ratelimit_client.get_ratelimit_data()['day']['remaining']
        count_check_requests_diff = count_check_requests_start - count_check_requests_end
        print(f"used {count_check_requests_diff} requests so far")

        print(ratelimit_client.get_ratelimit_data())
        time.sleep(0.33)
        next_ = response_cache.client.posts(blogName, limit=limit_, offset=offset_)["posts"]

    return ids_to_reward_data, offset_


def scrape_and_save(client, max_to_scrape=50, reset_offset=False, save_path="reward/reward.pkl.gz", ignore_reblogs_from_dash=True):
    if not os.path.exists(save_path):
        ids_to_reward_data = {}
        offset = 0
    else:
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        ids_to_reward_data = data["ids_to_reward_data"]
        offset = data["offset"]

    if reset_offset:
        offset = 0

    print(f"starting with {len(ids_to_reward_data)} posts of data")
    ids_to_reward_data, offset = scrape_note_counts(client, max_to_scrape, offset, ids_to_reward_data=ids_to_reward_data, ignore_reblogs_from_dash=ignore_reblogs_from_dash)

    print(f"ids_to_reward_data: {len(ids_to_reward_data)} items")
    print(f"offset: {offset}")

    data = {"ids_to_reward_data": ids_to_reward_data, "offset": offset}
    with open(save_path, "wb") as f:
        pickle.dump(data, f)
    print(f"saved {len(ids_to_reward_data)} posts of data")


def scrape_new_and_save(client, save_path="reward/reward.pkl.gz", offset=0, max_to_scrape=None, stop_on_seen_id=True, dry_run=False, ignore_reblogs_from_dash=True):
    if not os.path.exists(save_path):
        ids_to_reward_data = {}
    else:
        with open(save_path, "rb") as f:
            data = pickle.load(f)
        ids_to_reward_data = data["ids_to_reward_data"]

    print(f"starting with {len(ids_to_reward_data)} posts of data\n")

    ids_to_reward_data, offset = scrape_note_counts(client, max_to_scrape=max_to_scrape, offset=offset, stop_on_seen_id=stop_on_seen_id, ids_to_reward_data=ids_to_reward_data, ignore_reblogs_from_dash=ignore_reblogs_from_dash)

    print(f"ids_to_reward_data: {len(ids_to_reward_data)} items")
    print(f"offset: {offset}")

    if not dry_run:
        data = {"ids_to_reward_data": ids_to_reward_data, "offset": offset}
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"saved {len(ids_to_reward_data)} posts of data")
