import json
import argparse
from typing import Optional

from tqdm.autonotebook import tqdm

from api_tumblr.tumblr_parsing import TumblrThread
from tumblr_to_text.nwo import npf_thread_to_formatted_text
from tumblr_to_text.nwo_munging import make_nwo_prompts

from api_tumblr.client_pool import ClientPool
from api_tumblr.paging import fetch_posts

import multimodal.image_analysis_singleton

from config.bot_config import BotSpecificConstants
bot_name = BotSpecificConstants.load().blogName

UNUSED_TYPES = {"mood", "review", "manual"}


# TODO: better handling of fic override
def construct_head_training_texts(thread: TumblrThread, blog_name: str = bot_name):
    _, text_selector, text_autoreviewer = make_nwo_prompts(thread, blog_name=blog_name, ml_prompt_format=False)
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


def post_to_line_entry(post_payload: dict, blog_name: str = bot_name, include_unused_types=False):
    thread = TumblrThread.from_payload(post_payload)

    post_type = determine_post_type(thread, blog_name)

    if post_type in UNUSED_TYPES and not include_unused_types:
        text_full, text_selector, text_autoreviewer = "", "", ""
    else:
        text_full = npf_thread_to_formatted_text(thread)
        text_selector, text_autoreviewer = construct_head_training_texts(thread, blog_name)

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
                      include_unused_types=False):
    pool = ClientPool()

    posts = fetch_posts(pool, blog_name, n, offset)

    lines = [post_to_line_entry(pp, blog_name, include_unused_types=include_unused_types)
             for pp in tqdm(posts, mininterval=1)]

    return lines


def save(lines, path="data/head_training_data.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lines, f, indent=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--blog-name", type=str, default=bot_name)
    parser.add_argument("--include-unused-types", action="store_true")
    parser.add_argument("--save-image-cache", action="store_true")
    args = parser.parse_args()

    lines = fetch_and_process(args.blog_name, args.n, args.offset, args.include_unused_types)
    save(lines)
    if args.save_image_cache:
        multimodal.image_analysis_singleton.IMAGE_ANALYSIS_CACHE.save()


if __name__ == "__main__":
    main()
