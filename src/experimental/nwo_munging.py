from copy import deepcopy
from datetime import datetime

from api_tumblr.tumblr_parsing import NPFContent, TumblrPost, TumblrThread
from munging.year_munging import sample_year
from munging.autoresponder_static import DEFAULT_CSC
from experimental.nwo import post_payload_to_formatted_text, npf_thread_to_formatted_text


def replace_payload_timestamp(post_payload: dict, timestamp: int) -> dict:
    post_payload = deepcopy(post_payload)
    post_payload["timestamp"] = timestamp
    return post_payload


def sample_year_and_set(timestamp: datetime):
    return timestamp.replace(year=int(sample_year()))


def sample_year_and_set_payload_timestamp(post_payload: dict) -> dict:
    timestamp = datetime.fromtimestamp(post_payload["timestamp"])

    timestamp = sample_year_and_set(timestamp)

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


def make_nwo_textpost_prompts(blogName, timestamp, control_seg_config=DEFAULT_CSC, debug=True):
    prompts, prompts_selector, prompts_autoreviewer = [], {}, {}
    probs = []

    # regular
    probs.append(0.7)
    fake_post = TumblrPost(blog_name=blogName,
                           content=NPFContent(blocks=[], layout=[], blog_name=blogName),
                           tags=[])

    timestamp_posix = int(timestamp.timestamp())
    timestamp_sampled_posix = int(sample_year_and_set(timestamp).timestamp())

    fake_thread_real_ts = TumblrThread(posts=[fake_post], timestamp=timestamp_posix)
    fake_thread_sampled_ts = TumblrThread(posts=[fake_post], timestamp=timestamp_sampled_posix)

    prompt_regular = npf_thread_to_formatted_text(fake_thread_sampled_ts,
                                                  ml_prompt_format=True)  # generator
    prompts.append(prompt_regular)
    prompts_selector[prompt_regular] = npf_thread_to_formatted_text(fake_thread_real_ts,
                                                                    ml_prompt_format=True)  # selector sees real ts
    prompts_autoreviewer[prompt_regular] = npf_thread_to_formatted_text(fake_thread_real_ts,
                                                                        ml_prompt_format=True)  # autoreviewer sees real ts

    # fic
    probs.append(0.15)
    prompt_fic = control_seg_config["ORIG_FICTION_CHAR_FORUMLIKE"]
    prompts.append(prompt_fic)
    prompts_selector[prompt_fic] = control_seg_config["ORIG_POST_CHAR_FORUMLIKE"]  # selector sees regular post char
    prompts_autoreviewer[prompt_fic] = control_seg_config[
        "ORIG_POST_CHAR_FORUMLIKE"]  # autoreviewer sees regular post char

    # review
    probs.append(0.15)
    prompt_review = control_seg_config["REVIEW_CHAR_FORUMLIKE"]
    prompts.append(prompt_review)
    prompts_selector[prompt_review] = control_seg_config["ORIG_POST_CHAR_FORUMLIKE"]  # selector sees regular post char
    prompts_autoreviewer[prompt_review] = control_seg_config[
        "ORIG_POST_CHAR_FORUMLIKE"]  # autoreviewer sees regular post char

    if debug:
        print(f"prompts: {repr(prompts)}")
        print(f"prompts_selector: {repr(prompts_selector)}")
        print(f"prompts_autoreviewer: {repr(prompts_autoreviewer)}")
        print(f"probs: {repr(probs)}")

    return prompts, prompts_selector, prompts_autoreviewer, probs