from copy import deepcopy
from datetime import datetime

from api_tumblr.tumblr_parsing import TumblrThread
from munging.year_munging import sample_year
from experimental.nwo import post_payload_to_formatted_text, npf_thread_to_formatted_text


def replace_payload_timestamp(post_payload: dict, timestamp: int) -> dict:
    post_payload = deepcopy(post_payload)
    post_payload["timestamp"] = timestamp
    return post_payload


def sample_year_and_set_payload_timestamp(post_payload: dict) -> dict:
    timestamp = datetime.fromtimestamp(post_payload["timestamp"])

    timestamp = timestamp.replace(year=int(sample_year()))

    timestamp_posix = int(timestamp.timestamp())

    return replace_payload_timestamp(post_payload, timestamp_posix)


def cut_to_final_exchange(thread: TumblrThread) -> TumblrThread:
    posts_reversed = thread.posts[::-1]

    posts_reversed_cut = []
    n_retained = 0

    for post in posts_reversed:
        posts_reversed_cut.append(post)

        if len(post.content.blocks) > 0:
            n_retained += 1

        if n_retained >= 2:
            break

    posts = posts_reversed_cut[::-1]
    return TumblrThread(posts=posts, timestamp=thread.timestamp)


def cut_to_new_since_last_post_by_user(thread: TumblrThread, user_name: str) -> TumblrThread:
    posts_reversed = thread.posts[::-1]

    posts_reversed_cut = []
    n_by_user = 0

    for post in posts_reversed:
        posts_reversed_cut.append(post)

        if post.blog_name == user_name:
            n_by_user += 1

        if n_by_user >= 2:
            break

    posts = posts_reversed_cut[::-1]
    return TumblrThread(posts=posts, timestamp=thread.timestamp)


def make_nwo_prompts(post_payload, blogName, debug=True):
    prompt = post_payload_to_formatted_text(
        sample_year_and_set_payload_timestamp(post_payload), ml_prompt_format=True
    )

    thread = TumblrThread.from_payload(post_payload)

    thread_selector = cut_to_final_exchange(thread)
    prompt_selector = npf_thread_to_formatted_text(thread_selector, ml_prompt_format=True)

    thread_autoreviewer = cut_to_new_since_last_post_by_user(thread, blogName)
    prompt_autoreviewer = npf_thread_to_formatted_text(thread_autoreviewer, ml_prompt_format=True)

    if debug:
        print(f"prompt: {repr(prompt)}")
        print(f"prompt_selector: {repr(prompt_selector)}")
        print(f"prompt_autoreviewer: {repr(prompt_autoreviewer)}")

    return prompt, prompt_selector, prompt_autoreviewer


