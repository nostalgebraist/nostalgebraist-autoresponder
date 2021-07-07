from typing import List
from datetime import datetime

from api_tumblr.tumblr_parsing import NPFTextBlock, NPFContent, TumblrPost, TumblrThread
from munging.year_munging import sample_year
from munging.autoresponder_static import DEFAULT_CSC
from munging.autoresponder_static_v8 import construct_fic_override_v2
from munging.nwo.nwo import npf_thread_to_formatted_text, format_and_normalize_post_html


def sample_year_and_set(timestamp: datetime):
    return timestamp.replace(year=int(sample_year()))


def sample_year_and_set_timestamp(thread: TumblrThread) -> TumblrThread:
    timestamp = datetime.fromtimestamp(thread.timestamp)

    timestamp = sample_year_and_set(timestamp)

    timestamp_posix = int(timestamp.timestamp())

    new_thread = TumblrThread(posts=thread.posts, timestamp=timestamp_posix)
    return new_thread


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


def cut_to_n_most_recent_by_user(thread: TumblrThread, user_name: str, n_most_recent: int, keep_first=True) -> TumblrThread:
    posts_reversed = thread.posts[::-1]

    posts_reversed_cut = []
    n_by_user = 0

    for post in posts_reversed:
        posts_reversed_cut.append(post)

        if post.blog_name == user_name:
            n_by_user += 1

        if n_by_user >= n_most_recent:
            break

    posts = posts_reversed_cut[::-1]
    if not keep_first:
        posts = posts[1:]
    return TumblrThread(posts=posts, timestamp=thread.timestamp)


def pop_reblog_without_commentary(thread: TumblrThread):
    if len(thread.posts[-1].content.blocks) > 0:
        return thread

    return TumblrThread(posts=thread.posts[:-1], timestamp=thread.timestamp)


def set_tags(thread: TumblrThread, tags: List[str]):
    final_post = thread.posts[-1]
    final_post = TumblrPost(blog_name=final_post.blog_name, content=final_post.content, tags=tags)
    return TumblrThread(posts=thread.posts[:-1] + [final_post], timestamp=thread.timestamp)


def fake_tumblr_post(blog_name: str, text_blocks: List[str], tags: List[str]):
    blocks = [NPFTextBlock(text=text) for text in text_blocks]
    content = NPFContent(blocks=blocks, layout=[], blog_name=blog_name)

    fake_post = TumblrPost(blog_name=blog_name,
                           content=content,
                           tags=tags)
    return fake_post


def add_reblog(thread: TumblrThread,
               blog_name: str,
               text_blocks: List[str],
               tags: List[str],
               timestamp: datetime):
    fake_post = fake_tumblr_post(blog_name=blog_name, text_blocks=text_blocks, tags=tags)

    timestamp_posix = int(timestamp.timestamp())
    fake_thread = TumblrThread(posts=thread.posts + [fake_post], timestamp=timestamp_posix)

    return fake_thread


def add_empty_reblog(thread: TumblrThread, blog_name: str, timestamp: datetime):
    return add_reblog(thread, blog_name=blog_name, text_blocks=[], tags=[], timestamp=timestamp)


def insert_reply_before_final_post(
    thread: TumblrThread, reply_blog_name: str, reply_body: str
) -> TumblrThread:
    fake_post = fake_tumblr_post(blog_name=reply_blog_name, text_blocks=[reply_body], tags=[])

    new_posts = thread.posts[:-1] + [fake_post] + thread.posts[-1:]

    fake_thread = TumblrThread(posts=new_posts, timestamp=thread.timestamp)

    return fake_thread


def get_normalized_ask_text(thread: TumblrThread):
    if not thread.ask_content:
        raise ValueError(f"get_normalized_ask_text: no ask_content on thread")

    ask_text = thread.ask_content.to_html()
    ask_text = format_and_normalize_post_html(ask_text)

    return ask_text


def make_nwo_prompts(thread: TumblrThread, blog_name: str, debug=True):
    prompt = npf_thread_to_formatted_text(
        sample_year_and_set_timestamp(thread), ml_prompt_format=True
    )

    thread_selector = cut_to_final_exchange(thread)
    prompt_selector = npf_thread_to_formatted_text(thread_selector, ml_prompt_format=True)

    thread_autoreviewer = cut_to_n_most_recent_by_user(thread, blog_name, n_most_recent=2)
    prompt_autoreviewer = npf_thread_to_formatted_text(thread_autoreviewer, ml_prompt_format=True)

    if debug:
        print(f"prompt: {repr(prompt)}")
        print(f"prompt_selector: {repr(prompt_selector)}")
        print(f"prompt_autoreviewer: {repr(prompt_autoreviewer)}")

    return prompt, prompt_selector, prompt_autoreviewer


def make_nwo_fic_override_prompts(thread: TumblrThread, control_seg_config=DEFAULT_CSC, debug=True):
    ask_text = get_normalized_ask_text(thread)

    prompt = construct_fic_override_v2(ask_text, control_seg_config=control_seg_config)
    prompt_selector = control_seg_config["ORIG_POST_CHAR_FORUMLIKE"]
    prompt_autoreviewer = control_seg_config["ORIG_POST_CHAR_FORUMLIKE"]

    if debug:
        print(f"prompt: {repr(prompt)}")
        print(f"prompt_selector: {repr(prompt_selector)}")
        print(f"prompt_autoreviewer: {repr(prompt_autoreviewer)}")

    return prompt, prompt_selector, prompt_autoreviewer


def make_nwo_textpost_prompts(blog_name, timestamp, control_seg_config=DEFAULT_CSC, debug=True):
    prompts, prompts_selector, prompts_autoreviewer = [], {}, {}
    probs = []

    # regular
    probs.append(0.7)
    fake_post = fake_tumblr_post(blog_name=blog_name, text_blocks=[], tags=[])

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
