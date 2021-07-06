"""tumblr API <--> preprocessed text tools I need in more than one place"""
import re
import json
from copy import deepcopy
from datetime import datetime

import pytumblr
from bs4 import BeautifulSoup
from wcwidth import wcwidth

from munging import reblogs_v5
from munging.autoresponder_static import find_all_control_chars_chinese, CHINESE_CHAR_DELIMITERS, EOT, Q_CHAR, A_CHAR, \
    T_CHAR, UNAME_CHAR, ORIG_POST_CHAR_CHINESE
from munging.autoresponder_static_v8 import TIME_SIDECHANNEL_CHAR, timestamp_to_v10_format, join_time_sidechannel
from config.autoresponder_config import final_munge_before_neural
from util.error_handling import LogExceptionAndSkip

from multimodal.image_analysis import (
    V9_IMAGE_FORMATTER,
    IMAGE_DIR,
    IMAGE_DELIMITER,
    IMAGE_DELIMITER_WHITESPACED
)

from multimodal import image_analysis_singleton

image_analysis_cache = image_analysis_singleton.IMAGE_ANALYSIS_CACHE

from multimodal.text_segmentation import make_image_simple

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

    for delimiter in CHINESE_CHAR_DELIMITERS + [IMAGE_DELIMITER, TIME_SIDECHANNEL_CHAR]:
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

    post = "<p>" + post + "</p>"
    post = re.sub("\n", "</p><p>", post)
    return post


def inverse_format_post_for_api(post):
    if post.startswith("<p>"):
        post = post[len("<p>") :]
    if post.endswith("</p>"):
        post = post[: -len("</p>")]
    post = re.sub(r"</p><p>", "\n", post)
    post = re.sub(r"<br>", "\n", post)
    post = sanitize_user_input_outer_shell(post)
    if VERBOSE_LOGS:
        print(post)
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


def process_post_from_html_body(
    body: str, debug=False, V10=True,
) -> str:
    # warning: doesn't handle ask prefixes
    # if you want to go from an API payload to text, use process_post_from_post_payload

    if body is None or len(body) == 0:
        return body

    soup = BeautifulSoup(body, features="lxml")
    processed, _ = reblogs_v5.process_post(
        soup,
        use_article=False,
        debug=debug,
        V10=V10,
    )

    return processed


def process_post_from_post_payload(
    post: dict, debug=False, V10=True,
) -> str:
    post = simulate_legacy_payload(post)

    body = get_body(post)
    if body is None:
        return None

    processed = process_post_from_html_body(
        body, debug=debug, V10=V10,
    )

    if len(processed) == 0:
        # assume we should use A_CHAR here, we should never write a textpost of length 0
        processed = A_CHAR + "<|endoftext|>"

    if "question" in post and "asking_name" in post:
        ask_char = reblogs_v5.V10_ASK_CHAR if V10 else Q_CHAR
        ask_prefix = (
            UNAME_CHAR
            + post["asking_name"]
            + ask_char
            + "\n"
            + inverse_format_post_for_api(post["question"])
            + "\n"
        )
        if ORIG_POST_CHAR_CHINESE in processed:
            processed = processed.replace(ORIG_POST_CHAR_CHINESE, A_CHAR)
        processed = ask_prefix + processed
        if VERBOSE_LOGS:
            print(f"ask_prefix: {ask_prefix}")
    else:
        if VERBOSE_LOGS:
            print(f"didn't find ask keys; have keys {post.keys()}")
    if post.get("title") is not None and len(post["title"]) > 0:
        # TODO: fix this to put the title after the initial control char(s)
        title_prefix = f"<h2>{post['title']}</h2>\n"
        processed = title_prefix + processed
        if VERBOSE_LOGS:
            print(f"title_prefix: {title_prefix}")
    return processed


def screener_string_from_bootstrap_draft(d):
    processed = process_post_from_post_payload(d)
    cchars = find_all_control_chars_chinese(processed)

    if len(cchars) == 0:
        msg = f"screener_string_from_bootstrap_draft:\n\tweirdness: couldn't find control chars in {processed} for {d}"
        print(msg)
        return ""
    if cchars[-1][0] == A_CHAR:
        # should always be true for bootstrap drafts, but just for generality
        cchars = cchars[:-1]

    start_ix = 0
    for cc1, cc2 in zip(cchars[:-1], cchars[1:]):
        if cc1[0] == A_CHAR:
            # if we wrote the post in cc1, we know everything's good until the start of cc2
            start_ix = cc2[1]

    screener_string = processed[start_ix:]
    return screener_string


# image stuff


def find_images_and_sub_text(
    text: str,
    image_formatter=V9_IMAGE_FORMATTER,
    verbose=False,
):
    text_subbed = text

    for match in re.finditer(r"(<img src=\")([^\"]+)(\"[^>]*>)", text):
        imtext = image_analysis_cache.extract_and_format_text_from_url(
            match.group(2)
        )

        text_subbed = text_subbed.replace(
            match.group(0),
            imtext,
        )

    text_subbed = re.sub(r"<figure[^>]*>", "", text_subbed)
    text_subbed = text_subbed.replace("</figure>", "")

    return text_subbed


def upload_images_to_tumblr_urls(images, keys, client, blogname):
    if len(images) != len(keys):
        print(
            f"!! warning: in upload_images_to_tumblr_urls, got {len(images)} images but {len(keys)} keys"
        )
        return {}
    if len(images) == 0:
        return {}
    paths = [f"{IMAGE_DIR}/temp{i}.jpg" for i, im in enumerate(images)]
    for p, im in zip(paths, images):
        im.save(p, format="jpeg")

    # TODO: figure out how to do this in NPF consumption world
    orig_npf_flag = client.using_npf_consumption
    client.npf_consumption_off()

    r = client.create_photo(blogname, state="draft", data=paths)

    r2 = client.posts(blogname, id=r["id"])["posts"][0]
    urls = [ph["original_size"] for ph in r2["photos"]]

    client.delete_post(blogname, id=r["id"])

    if orig_npf_flag:
        client.npf_consumption_on()

    return {k: url for k, url in zip(keys, urls)}


def find_text_images_and_sub_real_images(
    text,
    client,
    blogname,
    verbose=False,
    dryrun=False,
):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    orig_text = text

    text = text.strip(" \n")

    if text.startswith(IMAGE_DELIMITER):
        text = "\n\n" + text

    if text.endswith(IMAGE_DELIMITER):
        text = text + "\n"

    # imtext_regex = r"(\n=======\n)(.+?)(=======\n)"
    escaped_delim = IMAGE_DELIMITER.encode('unicode_escape').decode()
    escaped_delim_ws = IMAGE_DELIMITER_WHITESPACED.encode('unicode_escape').decode()
    imtext_regex = rf"({escaped_delim_ws})(.+?)({escaped_delim}\n)"
    figure_format = """<figure data-orig-height="{h}" data-orig-width="{w}"><img src="{url}" data-orig-height="{h}" data-orig-width="{w}"/></figure>"""
    imtexts = set()
    ims_checksum = 0

    for match in re.finditer(
        imtext_regex,
        text,
        flags=re.DOTALL,
    ):
        imtext = match.group(2).rstrip("\n")
        imtexts.add(imtext)
        ims_checksum += 1

    vprint(f"find_text_images_and_sub_real_images: found {len(imtexts)} imtexts")

    if dryrun:
        return text, ims_checksum

    images = []
    keys = []
    for imtext in imtexts:
        if len(imtext) > 0:
            images.append(make_image_simple(imtext))
            keys.append(imtext)

    imtexts_to_tumblr_images = upload_images_to_tumblr_urls(
        images, keys, client, blogname
    )

    vprint(f"find_text_images_and_sub_real_images: uploaded {len(imtexts)} images")

    def _replace_with_figure(match):
        imtext = match.group(2).rstrip("\n")
        if imtext in imtexts_to_tumblr_images:
            tumblr_image = imtexts_to_tumblr_images[imtext]
            vprint(
                f"find_text_images_and_sub_real_images: subbing {repr(tumblr_image)} for {repr(imtext)}"
            )
            return figure_format.format(
                url=tumblr_image["url"],
                h=tumblr_image["height"],
                w=tumblr_image["width"],
            )
        else:
            vprint(
                f"find_text_images_and_sub_real_images: nothing to sub for {repr(imtext)}"
            )
            return ""

    text_subbed = re.sub(
        imtext_regex,
        _replace_with_figure,
        text,
        flags=re.DOTALL,
    )

    happened = len(imtexts_to_tumblr_images) > 0
    if not happened:
        text_subbed = orig_text  # ensure no munging for matching went through
    return text_subbed, happened


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


# "pretending the post is by me" tools


def write_text_for_side_judgment(
    post_payload,
    chop_on_a_char=False,
    add_tags=False,
    swap_in_frank=False,
    add_empty_response=True,
    keep_orig_post_char=False,
):
    processed = process_post_from_post_payload(post_payload)
    if processed is None:
        return False

    if ORIG_POST_CHAR_CHINESE in processed:
        if keep_orig_post_char:
            text = processed
        else:
            text = processed[processed.index(ORIG_POST_CHAR_CHINESE) + 1 :]
            if not swap_in_frank:
                text = UNAME_CHAR + post_payload["blog_name"] + Q_CHAR + text
    elif A_CHAR in processed:
        if chop_on_a_char:
            text = processed[
                [ix for ix, c in enumerate(processed) if c == A_CHAR][-1] + 1 :
            ]
            if not swap_in_frank:
                text = UNAME_CHAR + post_payload["blog_name"] + Q_CHAR + text
        else:
            text = processed
            if not swap_in_frank:
                text = text.replace(
                    A_CHAR, UNAME_CHAR + post_payload["blog_name"] + Q_CHAR
                )
    else:
        print("\trejecting: parse fail")
        return False
    if T_CHAR in text:
        text = text.partition(T_CHAR)[0]
    if text.endswith("<|endoftext|>"):
        text = text[: -len("<|endoftext|>")].rstrip("\n")

    if add_empty_response:
        text = text + A_CHAR

    tags = post_payload.get("tags", [])
    if not add_tags:
        tags = []
    text += T_CHAR + " ".join(["#" + t for t in tags])
    if ORIG_POST_CHAR_CHINESE not in text and A_CHAR not in text and UNAME_CHAR not in text:
        text = ORIG_POST_CHAR_CHINESE + text
    return text


# corpus (re)construction

# TODO: remove (currently used for validation)
def corpus_doc_from_post_payload(post_payload):
    ts = datetime.fromtimestamp(post_payload['timestamp'])
    v10_timestamp = timestamp_to_v10_format(ts)

    doc_chinese_format = write_text_for_side_judgment(
        post_payload,
        chop_on_a_char=False,
        add_tags=True,
        swap_in_frank=True,
        add_empty_response=False,
        keep_orig_post_char=True
    )

    user_name = post_payload['blog_name']

    doc_chinese_format = join_time_sidechannel(doc_chinese_format, v10_timestamp)

    kwargs = dict(mode='train', user_name=user_name)

    doc = final_munge_before_neural(doc_chinese_format, **kwargs)

    return doc


# TODO: remove (currently used for validation)
def archive_to_corpus(post_payload, path="data/dash_post_dump.txt", separator=EOT):
    with LogExceptionAndSkip("archive post to corpus"):
        doc = corpus_doc_from_post_payload(post_payload)

        if separator in doc:
            raise ValueError(f"separator in doc: {repr(doc)}")

        with open(path, "a", encoding="utf-8") as f:
            line = doc + EOT
            f.write(line)


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