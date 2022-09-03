from datetime import datetime

from api_tumblr.tumblr_parsing import TumblrThread
from tumblr_to_text.nwo import npf_thread_to_formatted_text, expand_asks
from tumblr_to_text.nwo_munging import add_empty_reblog

from api_ml.ml_connector import prob_delta_from_gpt


def construct_prob_delta_prompts_for_post(
    thread: TumblrThread,
    cut_to_last_and_skip_username=False,  # like ask username skipping, but for full posts
):
    # cut to last
    _, posts = expand_asks(thread)
    nposts = len(posts)

    thread = add_empty_reblog(thread, 'DUMMYUSER', datetime.now())

    prompt = npf_thread_to_formatted_text(thread, prob_delta_format=True)

    if cut_to_last_and_skip_username:
        # cut to last
        # TODO: maybe put this on the NPF -> text side? not sure which is better
        _, _, prompt = prompt.partition(f"#{nposts}")
        firstline, sep, rest = prompt.partition("\n")
        firstline = " " + firstline.split(" ")[-1]
        prompt = firstline + sep + rest

    prompt_ref = prompt.splitlines()[-1]

    if cut_to_last_and_skip_username:
        forbidden_strings = []  # ok to predict name if it's not in the prompt
    else:
        forbidden_strings = [" " + post.blog_name for post in posts[:-1]]

    return prompt, prompt_ref, forbidden_strings


def construct_prob_delta_prompts_for_ask(
    thread: TumblrThread,
    skip_asking_name=True,
):
    prompt = npf_thread_to_formatted_text(thread, prob_delta_format=True)

    if skip_asking_name:
        _, seg, suffix = prompt.partition(' asked')
        prompt = seg + suffix

    prompt_ref = prompt.splitlines()[-1]

    _, posts = expand_asks(thread)

    if skip_asking_name:
        posts = posts[1:]  # ok to predict asking name if it's not in the prompt

    forbidden_strings = [" " + post.blog_name for post in posts[:-1]]

    return prompt, prompt_ref, forbidden_strings


def get_prob_delta_for_payloads(
    post_payloads: list,
    blog_name: str,
    is_ask: bool,
    skip_asking_name=True,
    cut_to_last_and_skip_username=False,
):
    token_str = " " + blog_name
    kwargs = {"text": [], "text_ref": [], "token_str": token_str, "forbidden_strings": []}

    for pp in post_payloads:
        thread = TumblrThread.from_payload(pp)

        if is_ask:
            text, text_ref, forbidden_strings = construct_prob_delta_prompts_for_ask(
                thread,
                skip_asking_name=skip_asking_name,
            )
        else:
            text, text_ref, forbidden_strings = construct_prob_delta_prompts_for_post(
                thread,
                cut_to_last_and_skip_username=cut_to_last_and_skip_username,
            )
        kwargs["text"].append(text)
        kwargs["text_ref"].append(text_ref)
        kwargs["forbidden_strings"].append(forbidden_strings)

    return prob_delta_from_gpt(**kwargs)
