import os
import re
import argparse
import pickle
from collections import defaultdict
from functools import partial

from tqdm.autonotebook import tqdm

from bs4 import BeautifulSoup

from reblogs_v5 import *
from autoresponder_static_v8 import *
from munging_shared import *

import image_analysis_singleton
image_analysis_cache = image_analysis_singleton.IMAGE_ANALYSIS_CACHE

tqdm.pandas()


# TODO (out of date): get rid of this once all images are analyzed
# TODO (less out of date): stop analyzing new mood graphs?
CACHE_HITS = 0
CACHE_MISSES = 0


def cached_image_analysis_fn(
    elem,
    image_formatter=V9_IMAGE_FORMATTER,
    cached_only=False,
    verbose=False,
):
    global CACHE_HITS
    global CACHE_MISSES
    url_attr = "href" if elem.name == "a" else "src"

    if elem.attrs.get(url_attr) is None:
        return None

    is_hit = elem.attrs.get(url_attr) in image_analysis_cache.cache

    if is_hit:
        CACHE_HITS += 1
    else:
        CACHE_MISSES += 1

    if is_hit or (not cached_only):
        return IMAGE_ANALYSIS_FN(
            elem,
            image_formatter=image_formatter,
            verbose=verbose,
        )
    return ""


def fix_p_in_h2_bug(raw_html):
    return re.sub(
        r"(<h2>.*)<p>(.*)</p>(.*</h2>)", lambda m: "".join(m.groups()), raw_html
    )


def get_all_posts(
    posts_dir,
    cached_images_only=False,
    return_metadata_per_post=True,
):
    global CACHE_HITS
    global CACHE_MISSES

    posts = []
    post_fns = []
    image_urls = set()
    reply_urls_to_fns = defaultdict(set)
    metadata_per_post = {}

    all_fns = os.listdir(posts_dir)

    iter_ = tqdm(
        sorted(all_fns),
        mininterval=1,
        miniters=1,
        smoothing=0.3,
    )

    for ix, fn in enumerate(iter_):
        if not fn.endswith(".html"):
            continue

        with open(os.path.join(posts_dir, fn), "r") as f:
            raw_html = f.read()
            fixed_html = fix_p_in_h2_bug(raw_html)
            soup = BeautifulSoup(fixed_html)

        user_defined_image_analysis = partial(
            cached_image_analysis_fn, cached_only=cached_images_only
        )

        processed, post_metadata = process_post(
            soup,
            uname_config="frank_v10_1_operate",
            get_image_urls=True,
            user_defined_image_analysis=user_defined_image_analysis,
            V10=True,
            debug=False,
        )

        metadata_per_post[fn] = post_metadata
        image_urls.update(post_metadata["image_urls"])

        posts.append(processed)
        post_fns.append(fn)

        if post_metadata["reply_post_url"] is not None:
            reply_urls_to_fns[post_metadata["reply_post_url"]].add(fn)

        iter_.set_postfix(hits=CACHE_HITS, misses=CACHE_MISSES, refresh=False)

    return (
        posts,
        post_fns,
        image_urls,
        reply_urls_to_fns,
        metadata_per_post,
    )


def _autoresponse_to_prompt_and_continuation(processed):
    prompt = processed[::-1][processed[::-1].index(A_CHAR) :][::-1]
    continuation = processed[::-1][: processed[::-1].index(A_CHAR)][::-1]
    return prompt, continuation


def get_prompt_and_continuation_from_processed(processed: str):
    if A_CHAR in processed:
        prompt, continuation = _autoresponse_to_prompt_and_continuation(processed)
    elif ORIG_POST_CHAR in processed:
        prompt = ORIG_POST_CHAR
        continuation = processed[processed.index(ORIG_POST_CHAR) + 1 :]
    else:
        raise ValueError(f"don't know how to deal with post")

    if "<|endoftext|>" in continuation:
        continuation = continuation[: continuation.index("<|endoftext|>") + 2]

    if UNAME_CHAR in continuation:
        raise ValueError("bad parse")

    return prompt, continuation


def load_scraped_bot_posts(posts_dir, cached_images_only=False):
    (
        posts,
        post_fns,
        image_urls,
        reply_urls_to_fns,
        metadata_per_post,
    ) = get_all_posts(
        posts_dir=posts_dir,
        cached_images_only=cached_images_only,
        return_metadata_per_post=True,
    )

    post_ids = [int(fn[: -len(".html")]) for fn in post_fns]

    ids_to_posts = {pid: post for pid, post in zip(post_ids, posts)}

    ids_to_timestamps = {
        pid: get_ts_from_fn(os.path.join(posts_dir, fn))
        for pid, fn in zip(post_ids, post_fns)
    }

    ids_to_metas = {
        int(fn[: -len(".html")]): meta for fn, meta in metadata_per_post.items()
    }

    ids_to_note_counts = {
        pid: 0 if meta.get("note_count") is None else meta["note_count"]
        for pid, meta in ids_to_metas.items()
    }

    ids_to_loaded_data = {
        pid: {
            "post": ids_to_posts.get(pid),
            "timestamp": ids_to_timestamps.get(pid),
            "meta": ids_to_metas.get(pid),
            "note_count": ids_to_note_counts.get(pid),
        }
        for pid in post_ids
    }

    other_data = {"image_urls": image_urls}
    return ids_to_loaded_data, other_data


def fill_in_selector_training_data(ids_to_loaded_data, include_reblogs=False):
    ids_to_selector_training_data_rows = {}

    for id_ in tqdm(
        list(ids_to_loaded_data.keys()),
        mininterval=1,
        miniters=1,
        smoothing=0,
    ):
        meta = ids_to_loaded_data[id_]["meta"]
        is_reply = meta["reply_post_url"] is not None

        if (not include_reblogs) and (meta["is_reblog"] and not is_reply):
            continue

        post = ids_to_loaded_data[id_]["post"]
        try:
            prompt, continuation = get_prompt_and_continuation_from_processed(post)
        except Exception as e:
            print(f"skipping {id_}: {e}")

        note_count = ids_to_loaded_data[id_]["note_count"]
        timestamp = ids_to_loaded_data[id_]["timestamp"]

        new_row = {}
        new_row["prompt"] = prompt
        new_row["continuation"] = continuation
        new_row["note_count"] = note_count
        new_row["timestamp"] = timestamp
        new_row["v8_timestamp"] = timestamp_to_v8_format(timestamp)
        new_row["v10_timestamp"] = timestamp_to_v10_format(timestamp)
        new_row["is_ask"] = meta["is_ask"]
        new_row["is_orig"] = meta["is_orig"]
        new_row["is_reblog"] = meta["is_reblog"] and not is_reply
        new_row["is_reply"] = is_reply

        ids_to_selector_training_data_rows[id_] = new_row

    return ids_to_selector_training_data_rows


def selector_data_prep_pipeline(
    posts_dir,
    save_path,
    save_path_image_urls,
    cached_images_only=False,
    include_reblogs=False,
    save_image_analysis_cache=False,
    save_image_urls=False,
):
    ids_to_loaded_data, other_data = load_scraped_bot_posts(
        posts_dir, cached_images_only=cached_images_only
    )
    ids_to_selector_training_data_rows = fill_in_selector_training_data(
        ids_to_loaded_data, include_reblogs=include_reblogs
    )
    with open(save_path, "wb") as f:
        pickle.dump(ids_to_selector_training_data_rows, f)

    if save_image_analysis_cache:
        image_analysis_cache.save()

    if save_image_urls:
        with open(save_path_image_urls, "wb") as f:
            pickle.dump(other_data["image_urls"], f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--posts-dir", type=str, default="data/scraped_blog/posts/", required=False
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="data/selector_training_data.pkl.gz",
        required=False,
    )
    parser.add_argument(
        "--include-reblogs", default=False, action="store_true", required=False
    )
    parser.add_argument(
        "--cached-images-only", default=False, action="store_true", required=False
    )
    parser.add_argument(
        "--save-image-analysis-cache",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--save-image-urls",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--save-path-image-urls",
        type=str,
        default="data/selector_training_data_image_urls.pkl.gz",
        required=False,
    )

    args = parser.parse_args()

    selector_data_prep_pipeline(
        posts_dir=args.posts_dir,
        save_path=args.save_path,
        save_path_image_urls=args.save_path_image_urls,
        cached_images_only=args.cached_images_only,
        include_reblogs=args.include_reblogs,
        save_image_analysis_cache=args.save_image_analysis_cache,
        save_image_urls=args.save_image_urls,
    )
