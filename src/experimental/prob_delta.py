from datetime import datetime

from api_tumblr.tumblr_parsing import TumblrThread
from tumblr_to_text.nwo import npf_thread_to_formatted_text, expand_asks
from tumblr_to_text.nwo_munging import add_empty_reblog


def construct_prob_delt_probs(thread: TumblrThread):
    thread = add_empty_reblog(thread, 'DUMMYUSER', datetime.now())

    prompt = npf_thread_to_formatted_text(thread, prob_delta_format=True)

    prompt_ref = prompt.splitlines()[-1]

    _, posts = expand_asks(thread)
    forbidden_strings = [" " + post.blog_name for post in posts[:-1]]

    return prompt, prompt_ref, forbidden_strings
