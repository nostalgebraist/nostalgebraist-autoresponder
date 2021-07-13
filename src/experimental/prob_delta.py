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


def get_prob_delta_for_payloads(post_payloads: list, blog_name: str):
    token_str = " " + blog_name
    kwargs = {"text": [], "text_ref": [], "token_str": token_str, "forbidden_strings": []}

    for pp in post_payloads:
        thread = TumblrThread.from_payload(pp)

        text, text_ref, forbidden_strings = construct_prob_delta_prompts(thread)
        kwargs["text"].append(text)
        kwargs["text_ref"].append(text_ref)
        kwargs["forbidden_strings"].append(forbidden_strings)

    return prob_delta_from_gpt(**kwargs)
