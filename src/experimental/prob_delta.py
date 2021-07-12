from datetime import datetime

from api_tumblr.tumblr_parsing import TumblrThread
from tumblr_to_text.nwo import npf_thread_to_formatted_text, expand_asks
from tumblr_to_text.nwo_munging import add_empty_reblog

from api_ml.ml_connector import prob_delta_from_gpt


def construct_prob_delta_prompts(thread: TumblrThread):
    thread = add_empty_reblog(thread, 'DUMMYUSER', datetime.now())

    prompt = npf_thread_to_formatted_text(thread, prob_delta_format=True)

    prompt_ref = prompt.splitlines()[-1]

    _, posts = expand_asks(thread)
    forbidden_strings = [" " + post.blog_name for post in posts[:-1]]

    return prompt, prompt_ref, forbidden_strings


def get_prob_delta_for_payload(post_payload: dict, blog_name: str):
    thread = TumblrThread.from_payload(post_payload)

    prompt, prompt_ref, forbidden_strings = construct_prob_delta_prompts(thread)

    token_str = " " + blog_name

    if token_str in forbidden_strings:
        msg = f"get_prob_delta_for_payload: skipping, token_str={repr(token_str)} in"
        msg += f" forbidden_strings={repr(forbidden_strings)}"
        print(msg)
        return 0.

    return prob_delta_from_gpt(text=prompt,
                               text_ref=prompt_ref,
                               token_str=token_str,
                               forbidden_strings=forbidden_strings)
