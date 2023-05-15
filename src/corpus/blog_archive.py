# TODO: scrape format v2 here?
import json
import argparse
import random
import pickle
from typing import Optional
from datetime import datetime
from functools import partial

from tqdm.autonotebook import tqdm

from api_tumblr.tumblr_parsing import TumblrThread
from tumblr_to_text.nwo import npf_thread_to_formatted_text
from tumblr_to_text.nwo_munging import make_nwo_prompts

from api_tumblr.client_pool import ClientPool
from api_tumblr.paging import fetch_posts

from multimodal.image_analysis import ImageAnalysisCache

import multimodal.image_analysis_singleton

import config.bot_config_singleton
bot_specific_constants = config.bot_config_singleton.bot_specific_constants
bot_name = bot_specific_constants.blogName

from util.times import now_pst, fromtimestamp_pst

UNUSED_TYPES = {"mood", "review", "manual"}


def roll_head_timestamp(base_head_timestamp: datetime, actual_timestamp: datetime, n_future_months: int = 10):
    year = base_head_timestamp.year
    month = base_head_timestamp.month + random.randint(0, n_future_months)
    if month > 12:
        month = month - 11  # starts at 1, not 0
        year = year + 1
    return actual_timestamp.replace(year=year, month=month, day=2)


def sub_prompt_timestamp(base_head_timestamp, actual_timestamp, prompt):
    before, sep, seg = prompt.rpartition("\n\n Written ")
    timeseg, sep2, after = seg.partition(" | ")

    head_ts = roll_head_timestamp(
        base_head_timestamp=base_head_timestamp, actual_timestamp=actual_timestamp
    )

    return before + sep + head_ts.strftime("%-I %p %B %Y") + sep2 + after


# TODO: better handling of fic override
def construct_head_training_texts(thread: TumblrThread, base_head_timestamp: datetime, blog_name: str = bot_name, caption_fn=None):
    head_timestamp = roll_head_timestamp(base_head_timestamp=base_head_timestamp,
                                         actual_timestamp=fromtimestamp_pst(thread.timestamp))
    _, text_selector, text_autoreviewer = make_nwo_prompts(thread,
                                                           head_timestamp=head_timestamp,
                                                           blog_name=blog_name,
                                                           include_image_urls=True,
                                                           include_image_urls_for_heads=True,
                                                           ml_prompt_format=False)
    if caption_fn is not None:
        text_selector = caption_fn(text_selector)
        text_autoreviewer = caption_fn(text_autoreviewer)
    return text_selector, text_autoreviewer


def determine_post_type(thread: TumblrThread, blog_name: str = bot_name):
    if len(thread.posts[-1].content.blocks) == 0:
        # manual reblog from me
        return "manual"

    if len(thread.posts) == 1:
        if thread.ask_content:
            return "ask"
        text = thread.posts[0].to_html()
        if "replied to your post" in text.split("\n")[0]:
            return "reply"
        if "This is a graph of my mood over the last" in text:
            return "mood"
        if "I wrote this review by request of" in text:
            return "review"
        return "orig"

    if len(thread.posts) == 2:
        return "reblog_dash"

    second_to_last_name = thread.posts[-3].blog_name
    if second_to_last_name == blog_name:
        return "reblog_response"
    return "reblog_dash"


def post_to_line_entry(
    post_payload: dict,
    base_head_timestamp: datetime,
    blog_name: str = bot_name,
    include_unused_types=False,
    caption_fn=None,
):
    thread = TumblrThread.from_payload(post_payload)

    post_type = determine_post_type(thread, blog_name)

    if post_type in UNUSED_TYPES and not include_unused_types:
        text_full, text_selector, text_autoreviewer = "", "", ""
    else:
        text_full = npf_thread_to_formatted_text(thread, include_image_urls=True)
        if caption_fn is not None:
            text_full = caption_fn(text_full)
        text_selector, text_autoreviewer = construct_head_training_texts(
            thread, base_head_timestamp, blog_name, caption_fn=caption_fn
        )

    return {
        "id": post_payload["id"],
        "genesis_post_id": thread.posts[-1].genesis_post_id,
        "timestamp_posix": post_payload["timestamp"],
        "note_count": post_payload["note_count"],
        "post_type": post_type,
        "text_full": text_full,
        "text_selector": text_selector,
        "text_autoreviewer": text_autoreviewer,
    }


def fetch_and_process(blog_name: str = bot_name,
                      n: Optional[int] = None,
                      offset : int = 0,
                      include_unused_types=False,
                      fetch_only=False,
                      process_only=False,
                      save_processed_every=-1,
                      save_image_cache=False,
                      write_captions=False,
                      before=None,
                      processed_path="data/head_training_data.json"):
    with open("data/head_training_data_raw_posts.pkl.gz", "rb") as f:
        posts = pickle.load(f)

    max_ts_posix = max(pp["timestamp"] for pp in posts)
    max_ts = fromtimestamp_pst(max_ts_posix).isoformat()
    print(f"loaded {len(posts)} raw posts, max ts {max_ts}")

    lines = load(path=processed_path)
    max_processed_id = max(line["id"] for line in lines)
    print(f"loaded {len(lines)} existing records, max id {max_processed_id}")

    fetched_ids = {pp['id'] for pp in posts}
    processed_ids = {line['id'] for line in lines}

    if process_only:
        new_posts = posts
    else:
        pool = ClientPool()

        new_posts = fetch_posts(pool, blog_name, n, offset, needs_private_client=True, stop_at_id=max_processed_id, before=before)

        new_posts = [pp for pp in new_posts if pp["id"] not in fetched_ids]

        posts.extend(new_posts)

        print(f"saving {len(posts)} raw posts")

        with open("data/head_training_data_raw_posts.pkl.gz", "wb") as f:
            pickle.dump(posts, f)

    if fetch_only:
        return lines

    new_posts = [pp for pp in posts if pp["id"] not in processed_ids]
    new_posts = sorted(new_posts, key=lambda pp: pp['id'])
    new_ids = {pp['id'] for pp in new_posts}
    print(f"processing subset of length {len(new_posts)}, min id {min(new_ids)}, max id {max(new_ids)}")

    base_head_timestamp = now_pst()

    caption_fn = None
    if write_captions:
        from api_ml.ml_connector import caption_images_in_post_html
        caption_fn = partial(caption_images_in_post_html, verbose=False)

    for i, pp in enumerate(tqdm(new_posts, mininterval=0.3, smoothing=0)):
        line = post_to_line_entry(
            pp,
            base_head_timestamp,
            blog_name=blog_name,
            include_unused_types=include_unused_types,
            caption_fn=caption_fn
        )
        lines.append(line)
        if (save_processed_every > 0) and (i > 0) and (i % save_processed_every == 0):
            save(lines, path=processed_path)
            if save_image_cache:
                multimodal.image_analysis_singleton.IMAGE_ANALYSIS_CACHE.save()

    return lines


def reroll_head_timestamps(processed_path):
    lines = load(path=processed_path)

    base_head_timestamp = now_pst()

    for row in lines:
        actual_timestamp = fromtimestamp_pst(row["timestamp_posix"])
        for key in ['text_selector', 'text_autoreviewer']:
            subbed = sub_prompt_timestamp(base_head_timestamp, actual_timestamp, row[key])
            row[key] = subbed

    return lines


def save(lines, path="data/head_training_data.json"):
    print(f"saving {len(lines)} processed entries")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=1)


def load(path="data/head_training_data.json"):
    with open(path, "r", encoding="utf-8") as f:
        lines = json.load(f)
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=None)
    parser.add_argument("-offset", type=int, default=0)
    parser.add_argument("--blog-name", type=str, default=bot_name)
    parser.add_argument("--include-unused-types", action="store_true")
    parser.add_argument("--save-image-cache", action="store_true")
    parser.add_argument("--log-image-cache-misses", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--save-processed-every", type=int, default=-1)
    parser.add_argument("--aux-image-cache-path", type=str, default=None)
    parser.add_argument("--write-captions", action="store_true")
    parser.add_argument("--reroll-head-timestamps", action="store_true")
    parser.add_argument("--before", type=int, default=None)
    parser.add_argument("--processed-path", type=str, default="data/head_training_data.json")
    args = parser.parse_args()

    args = parser.parse_args()

    if args.log_image_cache_misses:
        multimodal.image_analysis_singleton.IMAGE_ANALYSIS_CACHE.log_cache_miss = True

    if args.reroll_head_timestamps:
        lines = reroll_head_timestamps(processed_path=args.processed_path)
    else:
        if args.aux_image_cache_path is not None:
            aux_image_cache = ImageAnalysisCache.load(args.aux_image_cache_path)
            multimodal.image_analysis_singleton.IMAGE_ANALYSIS_CACHE.aux_image_cache = aux_image_cache

        lines = fetch_and_process(blog_name=args.blog_name, n=args.n, offset=args.offset,
                                  include_unused_types=args.include_unused_types,
                                  fetch_only=args.fetch_only, process_only=args.process_only,
                                  write_captions=args.write_captions,
                                  save_processed_every=args.save_processed_every,
                                  save_image_cache=args.save_image_cache,
                                  before=args.before,
                                  processed_path=args.processed_path)
        if args.save_image_cache:
            multimodal.image_analysis_singleton.IMAGE_ANALYSIS_CACHE.save()

    if not args.fetch_only:
        save(lines, path=args.processed_path)


if __name__ == "__main__":
    main()
