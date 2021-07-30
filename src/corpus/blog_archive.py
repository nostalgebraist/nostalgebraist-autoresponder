import json
import argparse
import random
from typing import Optional
from datetime import datetime
import pickle

from tqdm.autonotebook import tqdm

from api_tumblr.tumblr_parsing import TumblrThread
from tumblr_to_text.nwo import npf_thread_to_formatted_text
from tumblr_to_text.nwo_munging import make_nwo_prompts

from api_tumblr.client_pool import ClientPool
from api_tumblr.paging import fetch_posts

import multimodal.image_analysis_singleton

import config.bot_config_singleton
bot_specific_constants = config.bot_config_singleton.bot_specific_constants
bot_name = bot_specific_constants.blogName

UNUSED_TYPES = {"mood", "review", "manual"}


def roll_head_timestamp(base_head_timestamp: datetime, actual_timestamp: datetime, n_future_months: int = 2):
    month = base_head_timestamp.month + random.randint(0, n_future_months)
    return actual_timestamp.replace(year=base_head_timestamp.year, month=month, day=base_head_timestamp.day)


# TODO: better handling of fic override
def construct_head_training_texts(thread: TumblrThread, base_head_timestamp: datetime, blog_name: str = bot_name):
    head_timestamp = roll_head_timestamp(base_head_timestamp=base_head_timestamp,
                                         actual_timestamp=datetime.fromtimestamp(thread.timestamp))
    _, text_selector, text_autoreviewer = make_nwo_prompts(thread,
                                                           head_timestamp=head_timestamp,
                                                           blog_name=blog_name,
                                                           ml_prompt_format=False)
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


def post_to_line_entry(post_payload: dict, base_head_timestamp: datetime,
                       blog_name: str = bot_name, include_unused_types=False):
    thread = TumblrThread.from_payload(post_payload)

    post_type = determine_post_type(thread, blog_name)

    if post_type in UNUSED_TYPES and not include_unused_types:
        text_full, text_selector, text_autoreviewer = "", "", ""
    else:
        text_full = npf_thread_to_formatted_text(thread)
        text_selector, text_autoreviewer = construct_head_training_texts(thread, base_head_timestamp, blog_name)

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
                      process_only=False):
    with open("data/head_training_data_raw_posts.pkl.gz", "rb") as f:
        posts = pickle.load(f)

    max_ts_posix = max(pp["timestamp"] for pp in posts)
    max_ts = datetime.fromtimestamp(max_ts_posix).isoformat()
    print(f"loaded {len(posts)} raw posts, max ts {max_ts}")

    lines = load()
    max_processed_id = max(line["id"] for line in lines)
    print(f"loaded {len(lines)} existing records, max id {max_processed_id}")

    if process_only:
        new_posts = [pp for pp in posts if pp["id"] > max_processed_id]
    else:
        pool = ClientPool()

        new_posts = fetch_posts(pool, blog_name, n, offset, needs_private_client=True, stop_at_id=max_processed_id)

        posts.extend(new_posts)

        print(f"saving {len(posts)} raw posts")

        with open("data/head_training_data_raw_posts.pkl.gz", "wb") as f:
            pickle.dump(posts, f)

    if fetch_only:
        return lines

    base_head_timestamp = datetime.now()

    lines_new = [post_to_line_entry(pp,
                                    base_head_timestamp,
                                    blog_name=blog_name,
                                    include_unused_types=include_unused_types)
                 for pp in tqdm(new_posts, mininterval=0.3, smoothing=0)]
    lines.extend(lines_new)
    return lines


def save(lines, path="data/head_training_data.json"):
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
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    args = parser.parse_args()

    lines = fetch_and_process(blog_name=args.blog_name, n=args.n, offset=args.offset,
                              include_unused_types=args.include_unused_types,
                              fetch_only=args.fetch_only, process_only=args.process_only)
    if args.save_image_cache:
        multimodal.image_analysis_singleton.IMAGE_ANALYSIS_CACHE.save()

    if not args.fetch_only:
        save(lines)


if __name__ == "__main__":
    main()
