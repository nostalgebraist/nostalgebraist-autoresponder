"""tumblr API <--> preprocessed text tools I need in more than one place"""
import re
import json
from copy import deepcopy
import html as html_lib

import pytumblr
from wcwidth import wcwidth

from tumblr_to_text.classic.autoresponder_static import ORIG_POST_CHAR_CHINESE, EOT

from multimodal.image_analysis_static import IMAGE_DELIMITER

from api_tumblr.tumblr_parsing import TumblrThread

from api_tumblr.pytumblr_wrapper import RateLimitClient

VERBOSE_LOGS = False

MAY_2020_TRAIL_HACK = True
JUNE_2020_TRAIL_HACK_UPDATE = True
AUGUST_2020_TRAIL_HACK_ADDON = True
FORCE_TRAIL_HACK_IDS = {621279018200760320, 624899848589688832, 645400204942704640, 645854610131714048, 645861728145555456}
JUNE_2020_LINKPOST_HACK = True


def sanitize_user_input_outer_shell(text):
    # to be applied to stuff near the tumblr API level
    sanitized_text = text

    # zero-width joiners etc
    sanitized_text = "".join([c for c in sanitized_text if wcwidth(c) != 0])

    for delimiter in [EOT, IMAGE_DELIMITER]:
        sanitized_text = sanitized_text.replace(delimiter, "")

    return sanitized_text


def format_post_for_api(post):
    # temporary hack
    post = (
        post.replace(ORIG_POST_CHAR_CHINESE, "")
        .replace("Blog post by Frank", "")
        .replace("Book review by Frank", "")
        .replace("Original fiction by Frank", "")
        .lstrip("\n")
    )

    post = html_lib.escape(post)
    post = "<p>" + post + "</p>"
    post = re.sub("\n", "</p><p>", post)
    return post


def is_npf(post_payload: dict) -> bool:
    # goals:
    #   - always check whether a paylod is npf in the same way
    #   - edits that "simulate legacy" should not change is_npf(payload)==True
    return "content" in post_payload


def simulate_legacy_payload(post_payload):
    # TODO: is this idempotent?
    payload_is_npf = is_npf(post_payload)

    if payload_is_npf:
        # npf branch
        sim_payload = deepcopy(post_payload)

        if "original_type" in post_payload:
            orig_type = post_payload["original_type"]

            # normalize types
            preferred_type_names = {
                "note": "answer",
                "regular": "text"
            }
            orig_type = preferred_type_names.get(orig_type, orig_type)

            sim_payload["type"] = orig_type
        else:
            print(f"no original_type key in payload, have type {post_payload.get('type')}, keys {sorted(post_payload.keys())}")

        thread = TumblrThread.from_payload(post_payload)
        op_content = thread.posts[0].content

        if op_content.has_ask:
            ask_content = op_content.ask_content
            sim_payload["asking_name"] = ask_content.asking_name
            sim_payload["question"] = ask_content.to_html()
            sim_payload["answer"] = op_content.to_html()

        if len(thread.posts) > 1:
            this_reblog = thread.posts[-1]
            comment = this_reblog.to_html()
            sim_payload["reblog"] = {
                "comment": comment,
                # real payloads also have a field "tree_html" but who cares
            }
    else:
        # legacy branch: return as is
        sim_payload = post_payload

    # validate
    if is_npf(sim_payload) != payload_is_npf:
        raise ValueError(f"simulated payload switched the value of is_npf: payload {repr(post_payload)} sim_payload {repr(sim_payload)}")
    return sim_payload


def get_body(post: dict):
    if is_npf(post):
        return TumblrThread.from_payload(post).to_html()
    if JUNE_2020_LINKPOST_HACK and post.get("type") == "link":
        try:
            return post["reblog"]["tree_html"] + post["reblog"]["comment"]
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
            print(
                f"couldn't find body, tried keys {body_keys}, available keys are {post.keys()}"
            )
            return None
    if MAY_2020_TRAIL_HACK and len(post.get("trail", [])) > 0:
        # check if body looks malformed
        is_malformed_body = False
        malformed_explainer = ""
        if JUNE_2020_TRAIL_HACK_UPDATE:
            (
                is_malformed_body_june,
                malformed_explainer_june,
            ) = malformed_diagnostic_june_2020(body)
            if not is_malformed_body and is_malformed_body_june:
                is_malformed_body = True
                malformed_explainer = malformed_explainer_june
        else:
            body_users = re.findall(r"\/\/(.+?).tumblr.com", body)
            trail_users = [item["blog"]["name"] for item in post["trail"]]
            is_malformed_body = any([(t != b) for t, b in zip(trail_users, body_users)])
            malformed_explainer = f"likely malformed body:\n\tbody_users {body_users} vs\n\ttrail_users {trail_users}\nwith body {body}"

        if AUGUST_2020_TRAIL_HACK_ADDON:
            (
                is_malformed_body_aug,
                malformed_explainer_aug,
            ) = malformed_diagnostic_august_2020(body)
            if not is_malformed_body and is_malformed_body_aug:
                is_malformed_body = True
                malformed_explainer = malformed_explainer_aug

        if is_malformed_body:
            if VERBOSE_LOGS:
                print(malformed_explainer)
            body = body_via_trail_hack(post)
    if any(
        [
            int(item.get("post", {}).get("id", -1)) in FORCE_TRAIL_HACK_IDS
            for item in post.get("trail", [])
        ]
    ):
        print('activated body_via_trail_hack')
        body = body_via_trail_hack(post)

    body = sanitize_user_input_outer_shell(body)

    return body


def malformed_diagnostic_june_2020(body):
    diagnostic_substring = "".join(body.rpartition("</blockquote>")[:2])
    is_malformed_body = diagnostic_substring.endswith("</blockquote></blockquote>")
    malformed_explainer = f"likely malformed body:\ndiagnostic_substring\n\t{diagnostic_substring}\n\nends with </blockquote></blockquote>"
    return is_malformed_body, malformed_explainer


def malformed_diagnostic_august_2020(body):
    body_re = re.compile(r"<p><a.*?class=\"tumblr_blog\".*?>.+?</a>:</p><blockquote>")
    diagnostic_segments = re.split(body_re, body.replace("\n", ""))
    is_malformed_body = not all([len(item) == 0 for item in diagnostic_segments[:-1]])
    malformed_explainer = f"likely malformed body:\ndiagnostic_segments\n\t{diagnostic_segments}\n\nsuggest text is interleaved with early blockquotes"
    return is_malformed_body, malformed_explainer


def body_via_trail_hack(post: dict):
    if "trail" not in post:
        print(f"couldn't find trail, available keys are {post.keys()}")
        return None

    final_unit = ""
    uname_units, content_units = [], []
    for item in post["trail"]:
        if not item.get("is_current_item", False):
            uname_units.append(
                f"<p><a class=\"tumblr_blog\">{item['blog']['name']}</a><blockquote></p>"
            )
            content_units.append(item["content_raw"] + "<" + "/" + "blockquote>")
        else:
            final_unit = item["content_raw"]
    my_units = uname_units[::-1] + content_units + [final_unit]
    return "".join(my_units)


# profanity tools
# currently unused


def keep_only_nonenglish_script(wash_lists):
    lists = {}
    for locale, words in wash_lists.items():
        words_filtered = [w for w in words if any([ord(c) > 256 for c in w])]
        if len(words_filtered) > 0:
            lists[locale] = words_filtered
    return lists


def flatten_wash_lists(wash_lists):
    entries = []
    for locale, words in wash_lists.items():
        for w in words:
            entries.append({"word": w, "locale": locale})
    return entries


def load_wash_lists():
    with open(
        "washyourmouthoutwithsoap_multilingual_profanity.json", "r", encoding="utf-8"
    ) as f:
        wash_lists = json.load(f)
    wash_lists = keep_only_nonenglish_script(wash_lists)
    wash_lists = flatten_wash_lists(wash_lists)
    return wash_lists


# npf --> legacy client

class LegacySimulatingClient(RateLimitClient):
    def send_api_request(
        self, method, url, params={}, valid_parameters=[], needs_api_key=False
    ):
        response = super().send_api_request(
            method,
            url,
            params=params,
            valid_parameters=valid_parameters,
            needs_api_key=needs_api_key,
        )
        if "posts" in response:
            response["posts"] = [simulate_legacy_payload(p) for p in response["posts"]]
        return response

    @staticmethod
    def from_tumblr_rest_client(client: pytumblr.TumblrRestClient, blogName) -> 'LegacySimulatingClient':
        return LegacySimulatingClient(
            consumer_key=client.request.consumer_key,
            consumer_secret=client.request.oauth.client.client_secret,
            oauth_token=client.request.oauth.client.resource_owner_key,
            oauth_secret=client.request.oauth.client.resource_owner_secret,
            blogName=blogName,
        )

    @staticmethod
    def from_rate_limit_client(client: RateLimitClient) -> 'LegacySimulatingClient':
        return LegacySimulatingClient(
            consumer_key=client.request.consumer_key,
            consumer_secret=client.request.oauth.client.client_secret,
            oauth_token=client.request.oauth.client.resource_owner_key,
            oauth_secret=client.request.oauth.client.resource_owner_secret,
            blogName=client.blogName,
            using_npf_consumption=client.using_npf_consumption
        )
