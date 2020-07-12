"""tumblr API <--> preprocessed text tools I need in more than one place"""
import re

import reblogs_v5
from bs4 import BeautifulSoup

VERBOSE_LOGS = False

MAY_2020_TRAIL_HACK = True
JUNE_2020_TRAIL_HACK_UPDATE = True
FORCE_TRAIL_HACK_IDS = {621279018200760320}
JUNE_2020_LINKPOST_HACK = True

Q_CHAR = "会"
A_CHAR = "域"
T_CHAR = "职"

UNAME_CHAR = "友"
ORIG_POST_CHAR = "翰"


def format_post_for_api(post):
    post = "<p>" + post + "</p>"
    post = re.sub("\n", "</p><p>", post)
    return post

def inverse_format_post_for_api(post):
    if post.startswith("<p>"):
        post = post[len("<p>"):]
    if post.endswith("</p>"):
        post = post[:-len("</p>")]
    post = re.sub(r"</p><p>", "\n", post)
    post = re.sub(r"<br>", "\n", post)
    if VERBOSE_LOGS:
        print(post)
    return post

def get_body(post: dict):
    if post.get("type") == "blocks":
        # can't ignore NPF forever, apparently :( :(
        # TODO: figure out if this ever triggers (bootstrap drafts are never NPF -- but sometimes malformed)
        def _parse_block(block):
            if block["type"] != "text":
                return ""
            formatted_text = "<p>" + block.get("text", "") + "</p>"
            if "heading" in block.get("subtype", ""):
                formatted_text = "<h2>" + formatted_text + "</h2>"
            if "indented" in block.get("subtype", ""):
                formatted_text = "<blockquote>" + formatted_text + "</blockquote>"
            if "formatting" in block:
                pass  # TODO: rollup tumblr's terrible re-implementation of basic text markup
        npf_parsed = "".join(_parse_block(block) for block in post.get("content", []))
        print(f"\nparsed NPF\n\t{post.get('content', [])} \n to \n\t{npf_parsed}\n")
        return npf_parsed
    if JUNE_2020_LINKPOST_HACK and post.get("type") == "link":
        try:
            return post['reblog']['tree_html'] + post['reblog']['comment']
        except Exception as e:
            print(f"june 2020 hack failed with {e} for {post}")
            pass
    else:
        body_keys = ["body", "answer", "caption", "description"]
        for k in body_keys:
            body = post.get(k, None)
            if body is not None:
                break
        if body is None:
            print(f"couldn't find body, tried keys {body_keys}, available keys are {post.keys()}")
            return None
    if MAY_2020_TRAIL_HACK and len(post.get('trail', [])) > 0:
        # check if body looks malformed
        if JUNE_2020_TRAIL_HACK_UPDATE:
            diagnostic_substring = "".join(body.rpartition("</blockquote>")[:2])
            is_malformed_body = diagnostic_substring.endswith("</blockquote></blockquote>")
            malformed_explainer = f"likely malformed body:\ndiagnostic_substring\n\t{diagnostic_substring}\n\nends with </blockquote></blockquote>"
        else:
            body_users = re.findall(r"\/\/(.+?).tumblr.com", body)
            trail_users = [item['blog']['name'] for item in post['trail']]
            is_malformed_body = any([(t!=b) for t, b in zip(trail_users, body_users)])
            malformed_explainer = f"likely malformed body:\n\tbody_users {body_users} vs\n\ttrail_users {trail_users}\nwith body {body}"

        if is_malformed_body:
            if VERBOSE_LOGS:
                print(malformed_explainer)
            body = body_via_trail_hack(post)
    if any([item.get('post', {}).get('id') in FORCE_TRAIL_HACK_IDS
            for item in post.get('trail', [])]):
        body = body_via_trail_hack(post)
    return body


def body_via_trail_hack(post: dict):
    if "trail" not in post:
        print(f"couldn't find trail, available keys are {post.keys()}")
        return None

    final_unit = ""
    uname_units, content_units = [], []
    for item in post['trail']:
        if not item.get('is_current_item', False):
            uname_units.append(f"<p><a class=\"tumblr_blog\">{item['blog']['name']}</a><blockquote></p>")
            content_units.append(item['content_raw'] + "<" + "/" + "blockquote>")
        else:
            final_unit = item['content_raw']
    my_units = uname_units[::-1] + content_units + [final_unit]
    return "".join(my_units)


def process_post_from_html_body(body: str) -> str:
    # warning: doesn't handle ask prefixes
    # if you want to go from an API payload to text, use process_post_from_post_payload

    if body is None or len(body) == 0:
        return body

    soup = BeautifulSoup(body, features="lxml")
    processed, _ = reblogs_v5.process_post(soup, use_article=False)

    return processed

def process_post_from_post_payload(post: dict) -> str:
    body = get_body(post)
    if body is None:
        return None

    processed = process_post_from_html_body(body)

    if len(processed) == 0:
        # assume we should use A_CHAR here, we should never write a textpost of length 0
        processed = A_CHAR + "<|endoftext|>"

    if "question" in post and "asking_name" in post:
        ask_prefix = UNAME_CHAR + post["asking_name"] + Q_CHAR + "\n" + inverse_format_post_for_api(post["question"]) + "\n"
        if ORIG_POST_CHAR in processed:
            processed = processed.replace(ORIG_POST_CHAR, A_CHAR)
        processed = ask_prefix + processed
        if VERBOSE_LOGS:
            print(f"ask_prefix: {ask_prefix}")
    else:
        if VERBOSE_LOGS:
            print(f"didn't find ask keys; have keys {post.keys()}")
    return processed
