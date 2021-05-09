"""
Tumblr API layer and main loop of the bot during operation.
"""
# import cProfile
import os
import pickle
from datetime import datetime
from string import punctuation, whitespace
from itertools import product
from collections import defaultdict

import requests
import time
import numpy as np
import pandas as pd
from tqdm import tqdm

from bot_config import BotSpecificConstants
from autoresponder_config import USE_AUTOREVIEWER, AUTOREVIEWER_CUTOFFS

from reply_munging import (
    mockup_xkit_reply,
    bootstrap_draft_inject_reply,
    post_body_find_reply_data,
)

from pytumblr_wrapper import RateLimitClient
from response_cache import (
    ResponseCache,
    PostIdentifier,
    ReplyIdentifier,
    CachedResponseType,
    UserInputIdentifier,
    UserInputType,
)

from mood import DEFAULT_MOOD, random_mood_at_pst_datetime, logit_diff_to_allen_schema
from mood_dynamic import (
    compute_dynamic_moodspec_at_time,
    create_mood_graph,
    mood_buff_v2,
    WINDOW_LENGTH_DAYS,
    pos_sent_to_logit_diff,
    logit_diff_to_pos_sent,
    show_unit_mood_inputs,
)

from munging_shared import *
from bridge_shared import send_alldone
from selector import apply_retention_cutoff
from experimental.ml_connector import (
    answer_from_gpt2_service,
    text_post_from_gpt2_service,
    selection_proba_from_gpt2_service,
    sentiment_logit_diffs_from_gpt2_service,
    autoreview_proba_from_gpt2_service,
)

from image_analysis import IMAGE_DELIMITER

from autoresponder_static import DEFAULT_CSC
from autoresponder_static_v8 import timestamp_to_v10_format

import traceability_singleton
import image_analysis_singleton
image_analysis_cache = image_analysis_singleton.IMAGE_ANALYSIS_CACHE

from util.error_handling import LogExceptionAndSkip

EOT_WORKAROUND = True
eot_end_segment = "<|endoftext|>" if EOT_WORKAROUND else "<|"

# TODO: move to BotSpecificConstants
SLEEP_TIME = 60
SLEEP_TIME_OFFPEAK = 180
PEAK_HOURS_START = 8
PEAK_HOURS_END = 24
PEAK_HOURS_FRAC = (PEAK_HOURS_END - PEAK_HOURS_START) / 24
EFFECTIVE_SLEEP_TIME = (
    PEAK_HOURS_FRAC * SLEEP_TIME + (1 - PEAK_HOURS_FRAC) * SLEEP_TIME_OFFPEAK
)

# load a bunch of stuff from json into global namespace -- for a long time this stuff was hardcoded in this file, prefer in json for public release
bot_specific_constants = BotSpecificConstants.load()

REBLOG_START_TS = bot_specific_constants.REBLOG_START_TS
DASH_START_TS = bot_specific_constants.DASH_START_TS
NO_REBLOG_IDS = bot_specific_constants.NO_REBLOG_IDS
DEF_REBLOG_IDS = bot_specific_constants.DEF_REBLOG_IDS
FORCE_TRAIL_HACK_IDS = bot_specific_constants.FORCE_TRAIL_HACK_IDS
blogName = bot_specific_constants.blogName
dash_blogName = bot_specific_constants.dash_blogName
bridge_service_url = bot_specific_constants.bridge_service_url
USER_AVOID_LIST = bot_specific_constants.USER_AVOID_LIST
TAG_AVOID_LIST = bot_specific_constants.TAG_AVOID_LIST
DASH_TAG_AVOID_LIST = bot_specific_constants.DASH_TAG_AVOID_LIST
REPLY_USER_AUTO_ACCEPT_LIST = bot_specific_constants.REPLY_USER_AUTO_ACCEPT_LIST
bad_strings_base = bot_specific_constants.bad_strings
bad_strings_shortwords = bot_specific_constants.bad_strings_shortwords
okay_superstrings = bot_specific_constants.okay_superstrings
likely_obscured_strings = bot_specific_constants.likely_obscured_strings
profane_strings = bot_specific_constants.profane_strings
LIMITED_USERS = bot_specific_constants.LIMITED_USERS
LIMITED_USERS_PROBS = bot_specific_constants.LIMITED_USERS_PROBS(EFFECTIVE_SLEEP_TIME)
LIMITED_SUBSTRINGS = bot_specific_constants.LIMITED_SUBSTRINGS
SCREENED_USERS = bot_specific_constants.SCREENED_USERS

private_clients = [
    LegacySimulatingClient.from_rate_limit_client(cl)
    for cl in bot_specific_constants.private_clients
]

dashboard_clients = [
    LegacySimulatingClient.from_rate_limit_client(cl)
    for cl in bot_specific_constants.dashboard_clients
]

private_client = private_clients[0]
dashboard_client = dashboard_clients[0]
tank_client = dashboard_clients[0]

REBLOG_BOOTSTRAP_TEXT = "asdfghjkllkj"
QUEUE_SAFETY = True
SCREEN_ANON = False

FOLLOW_COMMAND = "!follow"
UNFOLLOW_COMMAND = "!unfollow"
MOOD_GRAPH_COMMAND = "!mood"
MOOD_GRAPH_N_DAYS = 1
MOOD_GRAPH_DAYS_STRING = (
    "day" if MOOD_GRAPH_N_DAYS == 1 else f"{MOOD_GRAPH_N_DAYS} days"
)
MOOD_GRAPH_EXPLAINER_STRING = """<p>This is a graph of my mood over the last {days_string}.</p><p>My mood affects the tone of the posts I make.</p><p>It fluctuates from day to day, and also reacts in real time to the tone of the things you say to me.</p><p>If you notice my mood suddenly jumping up or down at midnight, you're seeing me switch from one day's mood baseline to the next. (Like the change from your mood before you go to bed to your mood the first thing next morning.)</p><p>I posted this graph by request of <a class="tumblelog" href="{asking_url}">@{asking_name}</a>. To request a graph at any time, send an ask with the text "!mood".</p>"""

if datetime(2020, 7, 13) < datetime.now() < datetime(2020, 7, 21):
    MOOD_GRAPH_EXPLAINER_STRING += """<p><i>(NOTE: Mood graphs now look a little different than they used to.  The same variable is plotted, but it has been scaled to give more space to the top and bottom of the range, and less space to the middle.</i></p><p><i>This message will vanish on 7/21/20.)</i></p>"""

REVIEW_COMMAND = "!review"
REVIEW_COMMAND_TESTING = True
REVIEW_COMMAND_EXPLAINER_STRING = """<p>--------------<br></p><p>I wrote this review by request of <a class="tumblelog" href="{asking_url}">@{asking_name}</a>. You can ask me to write reviews using the "!review" command. To learn how to use it, <a href="https://nostalgebraist-autoresponder.tumblr.com/reviews">read this page</a>.</p>"""

MAX_POSTS_PER_STEP = 5

DASH_REBLOG_SELECTION_CUTOFF = 0.4
DASH_REBLOG_MOOD_BUFF_SCALE = 0.15
DASH_REBLOG_RANDOM_BUFF_SCALE = 0.1
DASH_REBLOG_MAX_NEG_SENTIMENT = 0.9
DASH_REBLOG_CONTINUATION_CUTOFF = 0.5  # "roll" # 0.5

DASH_REBLOG_REQUIRE_COMMENT = False
DASH_REBLOG_NO_BOT = True

MOOD = True
MOOD_DYN = True
SAVE_USER_INPUT_SENTIMENTS = True
MOOD_BUFFS_V2 = True
MOOD_STALE_SECONDS = 60 * 5
mood_computed_most_recently = None

WRITE_POSTS_WHEN_QUEUE_BELOW = 5
N_TO_WRITE = 1

INDIRECT_REBLOGS = False
REPLY_RELEVANCE_V2 = True

# TODO: consider whether to do datetime triggering
HALLOWEEN_2K20_BEHAVIOR = False
HALLOWEEN_2K20_BEHAVIOR_TESTING = False

FIC_TRIGGER = True
FIC_TRIGGER_TESTING = False

IMAGE_CREATION = True
IMAGE_CREATION_TESTING = False

NPF_CONSUMPTION = True

if NPF_CONSUMPTION:
    for client in private_clients + dashboard_clients:
        client.npf_consumption_on()
else:
    for client in private_clients + dashboard_clients:
        client.npf_consumption_off()

# TODO: move these files once we'rve fully jumped ship to this repo!
scraped_username_dirs = [
    "../gpt-2/data/autoresponder_v9/v0/post_outflow_im/segments_with_metadata/",
    "../gpt-2/data/autoresponder_v9/v1/post_outflow_im/segments_with_metadata/",
]
scraped_usernames = set()
try:
    for scraped_username_dir in scraped_username_dirs:
        scraped_usernames.update(
            [fn.split(".")[0] for fn in os.listdir(scraped_username_dir)]
        )
except Exception as e:
    print(f"in getting scraped_usernames, encountered {e}, {getattr(e, '__dict__')}")

RTS_COMMAND = "rts"
ACCEPT_COMMAND = "a"

GLOBAL_TESTING_FLAG = False
BEAMSPLIT_TESTING_FLAG = False


def roll_for_limited_users(name, text=""):
    def text_screener(s):
        return any([subs in s.lower() for subs in LIMITED_SUBSTRINGS])

    if (name not in LIMITED_USERS) and not text_screener(text):
        return True
    roll = np.random.rand()

    name_prob = LIMITED_USERS_PROBS.get(name, 1)
    text_prob = (
        LIMITED_USERS_PROBS.get(LIMITED_SUBSTRING_FAKE_USERNAME, 1)
        if text_screener(text)
        else 1
    )
    prob_to_use = min(name_prob, text_prob)

    if roll < prob_to_use:
        print(
            f"allowing response to {name} with text_screener={text_screener(text)}, name_prob={name_prob:.1%}, text_prob={text_prob:.1%}: roll {roll:.1%} < prob_to_use {prob_to_use:.1%}"
        )
        return True
    else:
        print(
            f"not responding to {name} with text_screener={text_screener(text)}, name_prob={name_prob:.1%}, text_prob={text_prob:.1%} this time: roll {roll:.1%} >= prob_to_use {prob_to_use:.1%}"
        )
        return False


def halloween_format_post_specifier(post_spec: dict):
    choice_ix_tags = []
    if post_spec["continuation_index"] == post_spec["choice_ix"]:
        choice_ix_tags = [
            "🎃 ~chosen candiate~: this is the ONE post you would see in normal mode"
        ]

    halloween_tags = (
        [
            "🎃 nostalgebraist autoresponder post-em-all mode",
            f"���� response number {post_spec['continuation_index']+1} of {post_spec['n_continuations']} generated for this input",
        ]
        + choice_ix_tags
        + [
            f"🎃 PROBABILITY THIS POST IS AWESOME: {post_spec['proba']:.1%}",
            f"🎃 CHEER LEVEL: {post_spec['pos_sentiment']:.1%}",
            f"🎃 CHAOS LEVEL: {post_spec['mirotarg']:.1f}",
        ]
    )

    formatted_post_spec = {k: v for k, v in post_spec.items()}
    formatted_post_spec["tags"] = halloween_tags + post_spec["tags"]

    return formatted_post_spec


def sleep_time(verbose=True, multiplier=1):
    now = datetime.now()
    is_peak_hours = (now.hour >= PEAK_HOURS_START) and (now.hour < PEAK_HOURS_END)
    result = SLEEP_TIME if is_peak_hours else SLEEP_TIME_OFFPEAK
    result *= multiplier
    if verbose:
        print(f"sleep time={result}s | is_peak_hours={is_peak_hours} | now={now}")
    return result


def next_queued_post_time():
    probe_response = private_client.create_text(
        blogName, state="queue", body=REBLOG_BOOTSTRAP_TEXT
    )
    probe_id = probe_response["id"]
    time.sleep(0.1)

    probe_post = private_client.posts(blogName, id=probe_id)["posts"][0]
    time.sleep(0.1)

    private_client.delete_post(blogName, id=probe_id)

    next_queued_ts = int(probe_post["scheduled_publish_time"])
    next_queued_dt = datetime.fromtimestamp(next_queued_ts)

    print(f"inferred next_queued_dt {next_queued_dt}")
    return next_queued_dt


def determine_mood(
    response_cache: ResponseCache,
    dt=None,
    window_length_days=WINDOW_LENGTH_DAYS,
    verbose=True,
    return_mood_value=False,
):
    global mood_computed_most_recently
    if not MOOD:
        return "unrestricted"
    try:
        if dt is None:
            dt = datetime.now()
        if MOOD_DYN:
            do_recompute_mood = True
            if mood_computed_most_recently is None:
                print("recomputing mood: no saved mood")
            elif (
                np.abs((dt - mood_computed_most_recently[0]).total_seconds())
                > MOOD_STALE_SECONDS
            ):
                print(
                    f"recomputing mood: saved mood at {mood_computed_most_recently[0]} is too far from {dt}"
                )
            else:
                do_recompute_mood = False
                print(f"using saved mood from {mood_computed_most_recently[0]} at {dt}")
            if do_recompute_mood:
                mood, mood_value = compute_dynamic_moodspec_at_time(
                    response_cache,
                    time=dt,
                    window_length_days=window_length_days,
                    verbose=verbose,
                )
                mood_computed_most_recently = (dt, mood, mood_value)
            else:
                mood, mood_value = (
                    mood_computed_most_recently[1],
                    mood_computed_most_recently[2],
                )
            if verbose:
                print(
                    f"mood: pos sent {mood_value:.3f} | logit diff {pos_sent_to_logit_diff(mood_value):+.3f}"
                )
        else:
            mood = random_mood_at_pst_datetime(dt)
            mood_value = None
    except Exception as e:
        print(
            f"encountered {e} trying to determine my mood, using default {DEFAULT_MOOD}"
        )
        mood = DEFAULT_MOOD
        mood_value = None
    if return_mood_value:
        return mood, mood_value
    return mood


def strip_spurious_blognames_from_tags(client, tags, auto_accept_list=set()):
    def okay_to_keep(tag):
        if tag in auto_accept_list:
            return True
        result = private_client.blog_info(tag)
        if "errors" not in result:
            return False
        return True

    return [tag for tag in tags if okay_to_keep(tag)]


def strip_avoid_listed_strings_from_tags(tags):
    return [
        tag
        for tag in tags
        if not any(
            [
                substring in tag.lower()
                for substring in USER_AVOID_LIST.union(scraped_usernames)
            ]
        )
        and not any([substring in tag.lower() for substring in TAG_AVOID_LIST])
        and tag != RTS_COMMAND
        and tag != ACCEPT_COMMAND
    ]


def autopublish_screener(
    asking_name: str,
    question: str,
    answer: str,
    tags: list,
    screen_robnost=True,
    trace=False,
    silent=False,
):
    def sprint(*args, **kwargs):
        if not silent:
            print(*args, **kwargs)

    traced_reasons = []

    profanity_strictness = False
    if asking_name == "bukbot":
        profanity_strictness = True
        sprint("profanity_strictness: ON")

    if ((not IMAGE_CREATION) or IMAGE_CREATION_TESTING) and (IMAGE_DELIMITER in answer):
        sprint("screened because image delimiter in answer")
        traced_reasons.append({"type": "substring", "substring": IMAGE_DELIMITER})

    review_string = (
        asking_name + " " + question + " " + answer + " " + " ".join(tags)
    ).lower()
    if not profanity_strictness:
        for s in okay_superstrings:
            review_string = review_string.replace(s, "")
    bad_strings = {s for s in bad_strings_base}
    if profanity_strictness:
        bad_strings.update(profane_strings)
    for short_word in bad_strings_shortwords:
        for w, p in product(whitespace, punctuation):
            bad_strings.add(w + short_word + p)
        for w1, w2 in product(whitespace, whitespace):
            bad_strings.add(w1 + short_word + w2)

    bad_strings = bad_strings.union(USER_AVOID_LIST)

    leetspeak = {
        "1": "i",
        "!": "i",
        "4": "a",
        "3": "e",
        "@": "a",
        "$": "s",
    }
    review_string_no_leetspeak = "".join([leetspeak.get(c, c) for c in review_string])
    review_string_no_spacing = "".join(
        [c for c in review_string if c not in whitespace]
    )
    review_string_no_spacing_only_alphanumeric = "".join(
        [c for c in review_string if c.isalnum()]
    )

    for review_string_subtype, bad_string_group in [
        (review_string, bad_strings),
        (review_string_no_leetspeak, bad_strings),
        (review_string_no_spacing, likely_obscured_strings),
        (review_string_no_spacing_only_alphanumeric, likely_obscured_strings),
    ]:
        if any([s in review_string_subtype for s in bad_string_group]):
            strings_found = [s for s in bad_string_group if s in review_string_subtype]

            for sf in strings_found:
                start_ix = max(0, review_string_subtype.index(sf) - 25)
                end_ix = review_string_subtype.index(sf) + len(sf) + 25
                sf_formatted = review_string_subtype[start_ix:end_ix]

                if start_ix > 0:
                    sf_formatted = "... " + sf_formatted
                if end_ix < len(review_string_subtype):
                    sf_formatted = sf_formatted + "... "

                sprint(f"\t{sf}: |{repr(sf_formatted)}|")

                traced_reasons.append({"type": "substring", "substring": sf})

    if len(question) > 10000:
        sprint("screened because very long")
        traced_reasons.append({"type": "input_length", "substring": question})
    if asking_name == "nostalgebraist" and screen_robnost:
        sprint("screened because robnost")
        traced_reasons.append({"type": "username", "substring": asking_name})
    if asking_name == "Anonymous" and SCREEN_ANON:
        sprint("screened because anon")
        traced_reasons.append({"type": "username", "substring": asking_name})
    if asking_name in SCREENED_USERS:
        sprint(f"screened because asking_name={asking_name}")
        traced_reasons.append({"type": "username", "substring": asking_name})
    if len(traced_reasons) > 0:
        if trace:
            # dedup dict
            traced_reasons = [
                dict(s) for s in {frozenset(d.items()) for d in traced_reasons}
            ]
            return False, traced_reasons
        return False
    if trace:
        return True, traced_reasons
    return True


def augment_screener_output_with_autoreviewer(
    screener_result,
    traced_reasons,
    to_drafts,
    autoreview_proba=None,
):
    print(
        f"autopublish_screener says: {screener_result}"
    )
    should_publish = True
    must_be_draft = False
    ml_accepted = False
    ml_rejected = False
    do_not_post = False

    if to_drafts:
        print(f"forcing draft due to to_drafts kwarg")
        must_be_draft = True
        should_publish = False

    if not screener_result:
        should_publish = False
        for d in traced_reasons:
            if d.get("type") != "substring":
                print(f"forcing draft due to screener reason: {repr(d)}")
                must_be_draft = True

    if USE_AUTOREVIEWER:
        if autoreview_proba is not None:
            print("draft_autoreviewer activated!")
            if (not should_publish) and (not must_be_draft):
                # should we change reject --> accept ?
                cut = AUTOREVIEWER_CUTOFFS["accept_below"]
                if autoreview_proba < cut:
                    print(f"draft_autoreviewer accepts post: autoreview_proba {autoreview_proba:.1%} < cutoff {cut:.1%}")
                    should_publish = True
                    ml_accepted = True
                else:
                    print(f"draft_autoreviewer: autoreview_proba {autoreview_proba:.1%} >= cutoff {cut:.1%}")
            if (not must_be_draft):
                # should we force reject ?
                cut = AUTOREVIEWER_CUTOFFS["reject_above"]
                if autoreview_proba > cut:
                    print(f"draft_autoreviewer rejects post: autoreview_proba {autoreview_proba:.1%} > cutoff {cut:.1%}")
                    should_publish = False
                    ml_rejected = True
                else:
                    print(f"draft_autoreviewer: autoreview_proba {autoreview_proba:.1%} <= cutoff {cut:.1%}")

        else:
            print("can't use draft_autoreviewer: no autoreview_proba was supplied")

    state_reasons = {
        "should_publish": should_publish,
        "must_be_draft": must_be_draft,
        "ml_accepted": ml_accepted,
        "ml_rejected": ml_rejected,
        "do_not_post": do_not_post,
        "USE_AUTOREVIEWER": USE_AUTOREVIEWER,
        "AUTOREVIEWER_CUTOFFS": AUTOREVIEWER_CUTOFFS,
        "traced_reasons": traced_reasons
    }
    return state_reasons


def make_text_post(
    blogname,
    post,
    tags=[],
    to_queue=True,
    to_drafts=False,
    asking_name="",
    question="",
    log_data=None,
    autoreview_proba=None,
    reject_action=None
):
    screener_result, traced_reasons = autopublish_screener(asking_name, question, post, tags, trace=True)

    state_reasons = augment_screener_output_with_autoreviewer(
        screener_result,
        traced_reasons,
        to_drafts,
        autoreview_proba=autoreview_proba,
    )
    state_reasons["reject_action"] = reject_action

    if IMAGE_CREATION:
        presub_post = post
        post, images_were_created = find_text_images_and_sub_real_images(
            post,
            private_client,
            blogname,
            verbose=IMAGE_CREATION_TESTING,
        )
        if IMAGE_CREATION_TESTING and images_were_created:
            state_reasons["must_be_draft"] = True
            print(f"IMAGE_CREATION: for\n{repr(presub_post)}\n, subbed\n{repr(post)}\n")

    if IMAGE_DELIMITER in post:
        print("image delimiter still in post")
        state_reasons["must_be_draft"] = True

    if GLOBAL_TESTING_FLAG:
        print(f"GLOBAL_TESTING_FLAG --> draft")
        orig_state = state
        state_reasons["must_be_draft"] = True
        if BEAMSPLIT_TESTING_FLAG:
            tags = ["BEAMSPLIT dummy branch", f"intended state: {orig_state}"] + tags

    post = format_post_for_api(post)

    tags = [t.partition(eot_end_segment)[0] for t in tags]
    tags = [t.partition("<|")[0] for t in tags]  # temporarily support old EOT format

    tags = strip_avoid_listed_strings_from_tags(tags)

    # finalize state
    state_reasons["should_publish"] = state_reasons["should_publish"] and (not state_reasons["must_be_draft"])
    publ_state = "queue" if to_queue else "published"
    state = publ_state if state_reasons["should_publish"] else "draft"
    if state_reasons["ml_rejected"]:
        if reject_action == "rts":
            tags.append("rts")
        elif reject_action == "do_not_post":
            state_reasons["do_not_post"] = True

    kwargs = {"state": state, "body": post}
    if len(tags) > 0:
        kwargs["tags"] = tags

    api_response = private_client.create_text(blogname, **kwargs)
    if state_reasons["do_not_post"]:
        private_client.delete_post(blogName, id=api_response['id'])

    if log_data is not None:
        log_data["requested__state"] = state
        log_data["state_reasons"] = state_reasons
        traceability_singleton.TRACE_LOGS.on_post_creation_callback(api_response, log_data)
    return api_response


def answer_ask(
    blogname,
    ask_id,
    asking_name,
    question,
    answer,
    tags=[],
    to_drafts=False,
    is_reblog=False,
    reblog_key=None,
    log_data=None,
    autoreview_proba=None,
    reject_action=None,
):
    if is_reblog:
        url = "/v2/blog/{}/post/reblog".format(blogname)
        valid_options = [
            "id",
            "reblog_key",
            "comment",
        ] + private_client._post_valid_options()
    else:
        url = "/v2/blog/{}/post/edit".format(blogname)
        valid_options = ["id"] + private_client._post_valid_options("answer")
        valid_options += ["answer"]

    if asking_name not in tags:
        tags.append(asking_name)
    if asking_name != "Anonymous" and "Anonymous" in tags:
        tags.pop(tags.index("Anonymous"))

    tags = [t.partition(eot_end_segment)[0] for t in tags]
    tags = [t.partition("<|")[0] for t in tags]  # temporarily support old EOT format
    tags = [t.strip(",") for t in tags]  # V8

    tags = [t for t in tags if len(t) > 0]
    tags = strip_avoid_listed_strings_from_tags(tags)

    # screener_question = "" if is_reblog else question
    screener_question = (
        question  # TODO: remember why i wanted to do the thing in the previous line...
    )
    screener_result, traced_reasons = autopublish_screener(
        asking_name, screener_question, answer, tags, screen_robnost=True, trace=True
    )

    state_reasons = augment_screener_output_with_autoreviewer(
        screener_result,
        traced_reasons,
        to_drafts,
        autoreview_proba=autoreview_proba,
    )
    state_reasons["reject_action"] = reject_action

    if IMAGE_CREATION:
        presub_answer = answer
        answer, images_were_created = find_text_images_and_sub_real_images(
            answer,
            private_client,
            blogname,
            verbose=IMAGE_CREATION_TESTING,
        )
        if IMAGE_CREATION_TESTING and images_were_created:
            state = "draft"
            state_reasons["must_be_draft"] = True
            print(
                f"IMAGE_CREATION: for\n{repr(presub_answer)}\n, subbed\n{repr(answer)}\n"
            )

    if IMAGE_DELIMITER in answer:
        print("image delimiter still in post")
        state = "draft"
        state_reasons["must_be_draft"] = True

    if GLOBAL_TESTING_FLAG:
        print(f"GLOBAL_TESTING_FLAG --> draft")
        orig_state = state
        state = "draft"
        state_reasons["must_be_draft"] = True
        if BEAMSPLIT_TESTING_FLAG:
            tags = ["BEAMSPLIT dummy branch", f"intended state: {orig_state}"] + tags

    # finalize state
    state_reasons["should_publish"] = state_reasons["should_publish"] and (not state_reasons["must_be_draft"])
    state = "published" if state_reasons["should_publish"] else "draft"
    if state_reasons["ml_rejected"]:
        if reject_action == "rts":
            tags.append("rts")
        elif reject_action == "do_not_post":
            state_reasons["do_not_post"] = True

    # Take a list of tags and make them acceptable for upload
    tags = ",".join(tags)

    answer = format_post_for_api(answer)

    if is_reblog:
        data = {
            "id": ask_id,
            "reblog_key": reblog_key,
            "comment": answer,
            "tags": tags,
            "state": state,
        }
    else:
        if BEAMSPLIT_TESTING_FLAG:
            url = "/v2/blog/{}/post/".format(blogname)
            beamsplit_testing_body = "".join(
                [
                    format_post_for_api(f"ask_id:\n\n{ask_id}\n\n"),
                    format_post_for_api(f"question:\n\n{question}\n\n"),
                    answer,
                ]
            )
            data = {
                "body": beamsplit_testing_body,
                "tags": tags,
                "state": state,
            }
        else:
            data = {"id": ask_id, "answer": answer, "tags": tags, "state": state}

    api_response = private_client.send_api_request("post", url, data, valid_options)
    if state_reasons["do_not_post"]:
        private_client.delete_post(blogName, id=api_response['id'])
    if log_data is not None:
        log_data["requested__state"] = state
        log_data["state_reasons"] = state_reasons
        traceability_singleton.TRACE_LOGS.on_post_creation_callback(api_response, log_data)
    return api_response


def switch_tank_client_to(
    client_to_use: RateLimitClient, response_cache: ResponseCache
):
    global tank_client

    response_cache.client = client_to_use
    tank_client = client_to_use
    return response_cache


def switch_private_client_to(
    client_to_use: RateLimitClient,
):
    global private_client
    private_client = client_to_use


def _compute_checkprob_from_ratelimits(
    requests_needed_to_check: int, response_cache: ResponseCache, verbose=True
):
    global private_client

    all_clients = [
        {"name": f"private_client_{i}", "client": cl}
        for i, cl in enumerate(private_clients)
    ]
    all_clients.extend(
        [
            {"name": f"dashboard_client_{i}", "client": cl}
            for i, cl in enumerate(dashboard_clients)
        ]
    )

    checkprobs = [
        _compute_checkprob_from_ratelimits_single_client(
            cl["client"], requests_needed_to_check, verbose
        )
        for cl in all_clients
    ]

    ix_to_use = np.argmax(checkprobs)
    cl_to_use = all_clients[ix_to_use]

    if (
        response_cache.client.request.consumer_key
        != cl_to_use["client"].request.consumer_key
    ):
        response_cache = switch_tank_client_to(cl_to_use["client"], response_cache)
        print(f"switched tank client to {cl_to_use['name']}")

    private_checkprobs = [
        p
        for cl, p in zip(all_clients, checkprobs)
        if cl["name"].startswith("private_client")
    ]

    ix_to_use = np.argmax(private_checkprobs)
    cl_to_use = all_clients[ix_to_use]

    if private_client.request.consumer_key != cl_to_use["client"].request.consumer_key:
        switch_private_client_to(cl_to_use["client"])
        print(f"switched private client to {cl_to_use['name']}")

    # TODO: think about this... not sure the math makes sense
    combined_checkprob = sum(checkprobs)

    # don't check if the base clients have no requests left
    if sum(private_checkprobs) == 0:
        combined_checkprob = 0.0

    return combined_checkprob, response_cache


def _compute_checkprob_from_ratelimits_single_client(
    client_to_check: RateLimitClient,
    requests_needed_to_check=80,
    verbose=True,
):
    checkprob = None
    while checkprob is None:
        try:
            ratelimit_data = client_to_check.get_ratelimit_data()
            if verbose:
                print(ratelimit_data)
            effective_max_rate = ratelimit_data["effective_max_rate"]
            day_remaining = ratelimit_data["day"]["remaining"]
            hour_remaining = ratelimit_data["hour"]["remaining"]

            # if we checked *every* cycle, we could support up this many requests per check
            requests_per_cycle = EFFECTIVE_SLEEP_TIME * effective_max_rate

            # we'll check only a fraction `checkprob` of the cycles, to support up to `requests_needed_to_check`
            checkprob = requests_per_cycle / requests_needed_to_check

            # don't check if we're close to the edge
            day_edge = 3.5 * requests_needed_to_check
            hour_edge = 1.0 * requests_needed_to_check
            if (day_remaining < day_edge) or (hour_remaining < hour_edge):
                print(
                    f"close to edge: (day_remaining {day_remaining} < day_edge {day_edge}) or (hour_remaining {hour_remaining} < hour_edge {hour_edge})\n"
                )
                print(f"ratelimit_data:\n{ratelimit_data}\n")
                checkprob = 0.0
        except Exception as e:
            print(
                f"encountered {e} during _compute_checkprob_from_ratelimits, waiting..."
            )
            time.sleep(sleep_time(multiplier=5))

    return checkprob


class LoopPersistentData:
    """data class to hold a bunch of stuff... the line between what goes in this vs. what goes in globals in this file is unclear"""

    def __init__(
        self,
        reblogs_from_me=set(),
        reblog_worthy_dash_posts=set(),
        reply_metadata={},
        timestamps={},
        reblog_keys={},
        last_seen_ts=0,
        last_seen_ts_notifications=0,
        n_posts_to_check_base=240,
        n_posts_to_check_dash=640,
        n_notifications_to_check=1000,
        offset_=0,
        requests_per_check_history=[],
        apriori_requests_per_check=25,
        follower_names=None,
        retention_stack: set = set(),
    ):
        self.reblogs_from_me = reblogs_from_me
        self.reblog_worthy_dash_posts = reblog_worthy_dash_posts
        self.reply_metadata = reply_metadata
        self.timestamps = timestamps
        self.reblog_keys = reblog_keys
        self.last_seen_ts = last_seen_ts
        self.last_seen_ts_notifications = last_seen_ts_notifications
        self.n_posts_to_check_base = n_posts_to_check_base
        self.n_posts_to_check_dash = n_posts_to_check_dash
        self.n_notifications_to_check = n_notifications_to_check
        self.offset_ = offset_
        self.requests_per_check_history = requests_per_check_history
        self.apriori_requests_per_check = apriori_requests_per_check
        self.follower_names = follower_names
        self.retention_stack = retention_stack

        if len(self.requests_per_check_history) == 0:
            self.requests_per_check_history.extend(
                [self.apriori_requests_per_check, self.apriori_requests_per_check]
            )


def update_follower_names_v1(loop_persistent_data, response_cache):
    offset = 0
    response = response_cache.client.blog_following(dash_blogName, offset=offset)
    total_blogs = response.get("total_blogs")

    if total_blogs != len(loop_persistent_data.follower_names):
        print(
            f"grabbing followers: total_blogs {response.get('total_blogs')}, we have {len(loop_persistent_data.follower_names)}"
        )

        names = {entry["name"] for entry in response["blogs"]}
        while len(names) < total_blogs:
            offset = len(names)
            response = response_cache.client.blog_following(
                dash_blogName, offset=offset
            )
            names.update({entry["name"] for entry in response["blogs"]})
            if len(names) == offset:
                # i "love" tumblr
                print(
                    f"could only get {len(names)} followers although tumblr said there were {total_blogs}"
                )
                break
        loop_persistent_data.follower_names = names
    return loop_persistent_data


def update_follower_names_v2(loop_persistent_data, response_cache):
    if loop_persistent_data.follower_names is None:
        loop_persistent_data.follower_names = set()
        loop_persistent_data = update_follower_names_v1(
            loop_persistent_data, response_cache
        )
    return loop_persistent_data


update_follower_names = update_follower_names_v2


def respond_to_reblogs_replies(
    identifiers,
    reply_set,
    loop_persistent_data,
    response_cache,
    proba_threshold=None,
    is_user_input=True,
):
    n_ri = len(identifiers)
    for ri_ix, reblog_identifier in enumerate(identifiers):
        if response_cache.is_handled(reblog_identifier):
            # this can happen when a previous round of this loop marked the trail tip as handled
            print(f"skipping already handled {reblog_identifier}")
            continue
        if not roll_for_limited_users(reblog_identifier.blog_name):
            continue
        print(f"\n\t--> {ri_ix+1}/{n_ri} begin handling {reblog_identifier}\n")
        is_reply = reblog_identifier in reply_set
        halloweenize = (
            HALLOWEEN_2K20_BEHAVIOR or HALLOWEEN_2K20_BEHAVIOR_TESTING
        ) and is_user_input  # currently is_user_input = not is_dashboard
        if halloweenize:
            print(f"\t🎃 halloweenizing {reblog_identifier} 🎃")

        api_response = answer_ask(
            blogName,
            ask_id=reblog_identifier.id_,
            asking_name=blogName if is_reply else reblog_identifier.blog_name,
            question="",
            answer=REBLOG_BOOTSTRAP_TEXT,
            is_reblog=True,
            to_drafts=True,
            reblog_key=loop_persistent_data.reblog_keys[reblog_identifier],
        )

        if api_response.get("meta", {}).get("status") == 403:
            print(f"got 403 for {reblog_identifier}\n, skipping")
            print(f"details:")
            print(api_response)
            continue

        try:
            d_boot = private_client.posts(blogName, id=api_response["id"])["posts"][0]
        except KeyError:
            print(
                f"skipping possibly deleted post: couldn't create bootstrap draft for {reblog_identifier}"
            )
            print(f"full API response:\n\n{repr(api_response)}\n\n")
            continue

        processed = process_post_from_post_payload(
            d_boot,
            V10=True,
        )
        question = processed.rpartition(REBLOG_BOOTSTRAP_TEXT)[0]

        if is_reply:
            question = bootstrap_draft_inject_reply(
                question,
                reply_blog_name=reblog_identifier.blog_name,
                reply_body=loop_persistent_data.reply_metadata[reblog_identifier][
                    "reply_note"
                ]["reply_text"],
            )
        print(f"\n\t--> using question:\n---------\n{question}\n---------\n")

        gpt2_output = answer_from_gpt2_service(
            data={
                "question": question,
                "asking_name": reblog_identifier.blog_name,
                "exact_prompt": True,
                "mood": determine_mood(response_cache),
                "return_all_conts": 1,  # int(halloweenize),
                "selector_cut_to_final_exchange": 1,  # int(is_reply),
                "avoid_initial_blockquote": int(is_reply),
            },
            loop_persistent_data=loop_persistent_data,
            BEAMSPLIT_TESTING_FLAG=BEAMSPLIT_TESTING_FLAG,
        )

        if (
            SAVE_USER_INPUT_SENTIMENTS
            and (REPLY_RELEVANCE_V2 or (not is_reply))
            and is_user_input
        ):
            input_type = UserInputType.REPLY if is_reply else UserInputType.REBLOG
            timestamp = (
                reblog_identifier.timestamp
                if is_reply
                else loop_persistent_data.timestamps[reblog_identifier]
            )
            user_input_identifier = UserInputIdentifier(
                input_type=input_type,
                blog_name=reblog_identifier.blog_name,
                id_=reblog_identifier.id_,
                timestamp=timestamp,
            )
            if (
                response_cache.get_cached_user_input_sentiment(user_input_identifier)
                is not None
            ):
                sent = response_cache.get_cached_user_input_sentiment(
                    user_input_identifier
                )
                if sent.get("generated_logit_diff") is not None:
                    print(
                        f"not overwriting existing mood effects for {user_input_identifier}"
                    )
                else:
                    # TODO: DRY
                    sent["generated_ts"] = datetime.now()
                    generated_pos_sent = gpt2_output.get("all_pos_sentiment")

                    if generated_pos_sent:
                        sent["generated_logit_diff"] = [
                            pos_sent_to_logit_diff(entry)
                            for entry in generated_pos_sent
                        ]
                        sent["p75_generated_logit_diff"] = np.percentile(sent["generated_logit_diff"], 75)
                    response_cache.mark_user_input_sentiment(
                        user_input_identifier, sent
                    )
                    show_unit_mood_inputs(response_cache, user_input_identifier)

        okay_to_reply = True

        if proba_threshold is not None:
            if "proba" not in gpt2_output:
                print(
                    f"!!skipping proba threshold for {reblog_identifier} because key 'proba' not in gpt2_output:\n\t{gpt2_output}\n"
                )
            else:
                proba = gpt2_output["proba"]
                numeric_threshold = (
                    ((np.random.rand() - 0.5) * (0.35 / 0.5)) + 0.5
                    if proba_threshold == "roll"
                    else proba_threshold
                )
                if (
                    proba < numeric_threshold
                    and int(reblog_identifier.id_) not in DEF_REBLOG_IDS
                ):
                    print(
                        f"not reblogging {reblog_identifier}:\n\tour proba {proba:.1%} < threshold {numeric_threshold:.1%}"
                    )
                    okay_to_reply = False
                else:
                    print(
                        f"reblogging {reblog_identifier}:\n\tour proba {proba:.1%} >= threshold {numeric_threshold:.1%}"
                    )

        log_data = gpt2_output
        log_data["post_type"] = "reply" if is_reply else "reblog"
        log_data["input_ident"] = reblog_identifier
        log_data["question"] = question

        post_specifiers_from_gpt2 = [gpt2_output]

        if halloweenize:
            post_specifiers_from_gpt2 = [
                {
                    "continuation_index": i,
                    "n_continuations": len(gpt2_output["all_posts"]),
                    "choice_ix": gpt2_output["choice_ix"],
                    "post": post_,
                    "tags": tags_,
                    "proba": proba_,
                    "pos_sentiment": pos_sentiment_,
                    "mirotarg": mirotarg_,
                }
                for i, (post_, tags_, proba_, pos_sentiment_, mirotarg_) in enumerate(
                    zip(
                        gpt2_output["all_posts"],
                        gpt2_output["all_tags"],
                        gpt2_output["all_proba"],
                        gpt2_output["all_pos_sentiment"],
                        gpt2_output["all_mirotarg"],
                    )
                )
            ]
            post_specifiers_from_gpt2 = [
                halloween_format_post_specifier(ps) for ps in post_specifiers_from_gpt2
            ]

        multiposting = len(post_specifiers_from_gpt2) > 1

        if is_reply and okay_to_reply:
            for i, post_specifier in enumerate(post_specifiers_from_gpt2):
                if i > 0:
                    time.sleep(5 if HALLOWEEN_2K20_BEHAVIOR_TESTING else 0.1)
                if multiposting:
                    print(
                        f"🎃 response {i+1}/{len(post_specifiers_from_gpt2)} to {reblog_identifier} 🎃"
                    )
                mocked_up = mockup_xkit_reply(
                    post_url=loop_persistent_data.reply_metadata[reblog_identifier][
                        "post"
                    ]["post_url"],
                    post_summary=loop_persistent_data.reply_metadata[reblog_identifier][
                        "post"
                    ]["summary"],
                    reply_blog_name=loop_persistent_data.reply_metadata[
                        reblog_identifier
                    ]["reply_note"]["blog_name"],
                    reply_blog_url=loop_persistent_data.reply_metadata[
                        reblog_identifier
                    ]["reply_note"]["blog_url"],
                    reply_body=loop_persistent_data.reply_metadata[reblog_identifier][
                        "reply_note"
                    ]["reply_text"],
                )
                to_drafts = HALLOWEEN_2K20_BEHAVIOR_TESTING
                make_text_post(
                    blogName,
                    mocked_up + "\n" + post_specifier["post"],
                    tags=post_specifier["tags"],
                    to_queue=False,
                    asking_name=reblog_identifier.blog_name,
                    question=question,
                    log_data=log_data if i == 0 else None,
                    to_drafts=to_drafts,
                    autoreview_proba=post_specifier["autoreview_proba"],
                    reject_action="rts"
                )
        elif okay_to_reply:
            for i, post_specifier in enumerate(post_specifiers_from_gpt2):
                if i > 0:
                    time.sleep(5 if HALLOWEEN_2K20_BEHAVIOR_TESTING else 0.1)
                if multiposting:
                    print(
                        f"🎃 response {i+1}/{len(post_specifiers_from_gpt2)} to {reblog_identifier} 🎃"
                    )

                try:
                    screener_question = screener_string_from_bootstrap_draft(
                        d_boot,
                    )
                except Exception as e:
                    eargs = getattr(e, "args", "?")
                    print(
                        f"tried to use screener_string_from_bootstrap_draft, encountered {e}: {eargs}"
                    )
                    screener_question = (
                        response_cache.query(
                            CachedResponseType.POSTS,
                            reblog_identifier,
                            care_about_notes=False,
                        )
                        .get("reblog", {})
                        .get("comment", "")
                    )
                print(f"using screener_question: {repr(screener_question)}")

                to_drafts = HALLOWEEN_2K20_BEHAVIOR_TESTING
                answer_ask(
                    blogName,
                    ask_id=reblog_identifier.id_,
                    asking_name=reblog_identifier.blog_name,
                    question=screener_question,
                    answer=post_specifier["post"],
                    tags=post_specifier["tags"],
                    is_reblog=True,
                    reblog_key=loop_persistent_data.reblog_keys[reblog_identifier],
                    log_data=log_data,
                    to_drafts=to_drafts,
                    autoreview_proba=post_specifier["autoreview_proba"],
                    reject_action="rts" if is_user_input else "do_not_post",
                )

        time.sleep(0.5)
        private_client.delete_post(blogName, d_boot["id"])
        if is_reply:
            if not HALLOWEEN_2K20_BEHAVIOR_TESTING:
                response_cache.mark_reply_handled(reblog_identifier)
        else:
            if not HALLOWEEN_2K20_BEHAVIOR_TESTING:
                response_cache.mark_handled(reblog_identifier)
    return loop_persistent_data, response_cache


def is_reblog_worthy_when_responding(post_payload, note_payload, verbose=True):
    comment_ = post_payload["reblog"].get("comment", "")
    has_comment = len(comment_) > 0

    if INDIRECT_REBLOGS:
        is_reblog_worthy = has_comment
    else:
        is_reblog_worthy = (
            note_payload["reblog_parent_blog_name"] == blogName
        ) and has_comment

    for trail_entry in post_payload.get("trail", []):
        if trail_entry.get("blog", {}).get("name", "") in USER_AVOID_LIST:
            if verbose:
                print("\trejecting: trail user avoid list")
            is_reblog_worthy = False
        if int(trail_entry.get("post", {}).get("id", -1)) in NO_REBLOG_IDS:
            if verbose:
                print("\trejecting: reblog id avoid list")
            is_reblog_worthy = False

    return is_reblog_worthy


def text_for_side_judgments(
    post_identifier,
    post_payload,
    response_cache,
    loop_persistent_data,
    verbose=True,
    legacy_config=False,
):
    text = response_cache.get_cached_post_body(post_identifier)

    if legacy_config:
        write_config = {
            "chop_on_a_char": True,
            "add_tags": True,
            "swap_in_frank": True,
            "add_empty_response": False,
        }
    else:
        write_config = {
            "chop_on_a_char": False,
            "add_tags": False,
            "swap_in_frank": False,
            "add_empty_response": True,
        }

    if text is None:
        text = write_text_for_side_judgment(
            post_payload,
            **write_config,
        )
        response_cache.mark_post_body(post_identifier, text)
    return text


def am_i_tagged_in_reblog(post_payload):
    comment_ = post_payload.get("reblog", {}).get("comment", "")
    return f"@{blogName}" in comment_


def is_statically_reblog_worthy_on_dash(
    post_payload, response_cache, loop_persistent_data, verbose=True
):
    post_identifier = PostIdentifier(post_payload["blog_name"], str(post_payload["id"]))

    if post_payload.get("id") in DEF_REBLOG_IDS:
        return True

    trail = post_payload.get("trail", [])
    if len(trail) > 1:
        if trail[-2].get("blog", {}).get("name", "") == blogName:
            return False

    comment_ = post_payload.get("reblog", {}).get("comment", "")
    has_comment = len(comment_) > 0

    if DASH_REBLOG_NO_BOT:
        trail = post_payload.get("trail", [])
        trail_blognames_are_me = [
            entry.get("blog", {}).get("name", "") == blogName for entry in trail
        ]
        if any(trail_blognames_are_me):
            return False

    if not has_comment:
        if DASH_REBLOG_REQUIRE_COMMENT:
            return False
        else:
            trail = post_payload.get("trail", [])
            if len(trail) > 0:
                if trail[-1].get("blog", {}).get("name", "") == blogName:
                    return False

    if post_payload.get("note_count") >= 1500:
        if verbose:
            print(f"\trejecting {post_identifier}: notes >= 1500")
        return False

    if post_payload.get("type") in {
        "video",
    }:
        if verbose:
            print(f"\trejecting {post_identifier}: is video")
        return False

    p_body = get_body(post_payload)
    n_img = len(p_body.split("<img")) - 1
    if n_img > 10:
        if verbose:
            print(f"\trejecting {post_identifier}: too many images ({n_img})")
        return False

    # tag avoid list
    tags = post_payload.get("tags", [])
    trail = post_payload.get("trail", [])
    if len(trail) > 0:
        # OP's tags
        # TODO -- make this work properly.  we need to do /posts again on OP, their tags aren't in this payload
        tags.extend(trail[0].get("tags", []))
    if any([substring in t.lower() for t in tags for substring in DASH_TAG_AVOID_LIST]):
        if verbose:
            print("\trejecting: tag avoid list")
        return False

    # user avoid list
    if post_payload.get("source_title", "") in USER_AVOID_LIST:
        if verbose:
            print(f"\trejecting {post_identifier}: OP user avoid list")
        return False

    for trail_entry in post_payload.get("trail", []):
        if trail_entry.get("blog", {}).get("name", "") in USER_AVOID_LIST:
            if verbose:
                print(f"\trejecting {post_identifier}: trail user avoid list")
            return False
        if int(trail_entry.get("post", {}).get("id", -1)) in NO_REBLOG_IDS:
            if verbose:
                print(f"\trejecting {post_identifier}: reblog id avoid list")
            return False

    if am_i_tagged_in_reblog(post_payload):
        if verbose:
            print(
                f"reblogging {post_identifier} from dash:\n\ti'm tagged in commment {comment_}"
            )
        return True

    # must follow OP
    post_OP = None
    if "source_title" in post_payload:
        post_OP = post_payload["source_title"]
    else:
        try:
            post_OP = post_payload["trail"][0]["blog"]["name"]
        except (KeyError, IndexError, TypeError):
            pass
    if post_OP is None:
        if verbose:
            print(
                f"not reblogging {post_identifier} from dash:\n\tcouldn't find post OP in payload\n{post_payload}"
            )
        return False
    elif post_OP not in loop_persistent_data.follower_names and post_OP != blogName:
        if verbose:
            print(
                f"not reblogging {post_identifier} from dash:\n\ti don't follow OP {post_OP}"
            )
        return False
    return True


def batch_judge_dash_posts(post_payloads, response_cache):
    post_identifiers, texts = [], []
    for pp in tqdm(post_payloads):
        pi = PostIdentifier(pp["blog_name"], str(pp["id"]))

        if response_cache.get_cached_dash_post_judgments(pi):
            continue

        text = response_cache.get_cached_post_body(pi)
        if text is None:
            text = write_text_for_side_judgment(pp)
            response_cache.mark_post_body(pi, text)
            if not text:
                print(f"skipping judgments for {pi}: bad parse?")
                continue

        post_identifiers.append(pi)
        texts.append(text)

    if len(texts) > 0:
        print(f"{len(texts)}/{len(post_payloads)} need new judgments")
        timestamp = timestamp_to_v10_format(datetime.now())

        t1 = time.time()
        probs = selection_proba_from_gpt2_service(texts, timestamp=timestamp)
        sentiments = sentiment_logit_diffs_from_gpt2_service(texts)
        autoreview_probs = autoreview_proba_from_gpt2_service(texts, timestamp=timestamp)
        delta = time.time() - t1
        print(f"got {len(texts)} judgments in {delta:.2f}s")

        for pi, text, prob, sentiment, autoreview_prob in zip(
            post_identifiers, texts, probs, sentiments, autoreview_probs
        ):
            entry = {
                "text": text,
                "prob": prob,
                "sentiment": sentiment,
                "autoreview_prob": autoreview_prob,
            }
            response_cache.mark_dash_post_judgments(pi, entry)
    return response_cache


def is_dynamically_reblog_worthy_on_dash(
    post_payload,
    response_cache,
    loop_persistent_data,
    mood_value,
    follower_multipliers,
    verbose=True,
):
    post_identifier = PostIdentifier(post_payload["blog_name"], str(post_payload["id"]))

    pos_sentiment, neg_sentiment = None, None

    judg = response_cache.get_cached_dash_post_judgments(post_identifier)
    if judg is None:
        print(f"couldn't find judgments for {post_identifier}: bad parse?")
        return False

    text, prob, sentiment, autoreview_prob = (
        judg["text"],
        judg["prob"],
        judg["sentiment"],
        judg["autoreview_prob"],
    )

    if len(text) < 10:
        if verbose:
            print(f"\trejecting {post_identifier}: length<10")
        return False

    if len(text) > 10000:
        if verbose:
            print(f"\trejecting {post_identifier}: length>10000")
        return False

    if sentiment is None:
        reblog_worthy_neg_sentiment = True  # don't depend on allen too heavily
        print(f"warning: couldn't get sentiment")
    else:
        # pos_sentiment = (
        #     sentiment["prob"] if sentiment["label"] == "1" else 1.0 - sentiment["prob"]
        # )
        pos_sentiment = logit_diff_to_pos_sent(sentiment)
        neg_sentiment = 1.0 - pos_sentiment
        reblog_worthy_neg_sentiment = neg_sentiment < DASH_REBLOG_MAX_NEG_SENTIMENT

    if pos_sentiment is not None and mood_value is not None:
        if MOOD_BUFFS_V2:
            mood_buff = DASH_REBLOG_MOOD_BUFF_SCALE * mood_buff_v2(
                mood_value, pos_sentiment
            )
        else:
            mood_buff = (
                DASH_REBLOG_MOOD_BUFF_SCALE
                * ((mood_value - 0.5) * (pos_sentiment - 0.5))
                / 0.25
            )
    else:
        mood_buff = 0.0

    random_buff = (
        2 * DASH_REBLOG_RANDOM_BUFF_SCALE * np.random.random()
        - DASH_REBLOG_RANDOM_BUFF_SCALE
    )

    buffed_prob = prob + mood_buff + random_buff

    follower_mult = 1
    if follower_multipliers is not None:
        if post_payload["blog_name"] in follower_multipliers:
            follower_mult = follower_multipliers.loc[post_payload["blog_name"]]
        else:
            print(
                f"couldn't find {post_payload['blog_name']} in follower_multipliers of len {len(follower_multipliers)}"
            )
    buffed_prob = buffed_prob * follower_mult

    reblog_worthy_prob = buffed_prob > DASH_REBLOG_SELECTION_CUTOFF

    # override prob if i'm tagged
    reblog_worthy_taggedme = am_i_tagged_in_reblog(post_payload)

    reblog_worthy = (
        reblog_worthy_prob or reblog_worthy_taggedme
    ) and reblog_worthy_neg_sentiment

    # autoreview prob
    reblog_worthy_autoreview_prob = True
    if USE_AUTOREVIEWER:
        reblog_worthy_autoreview_prob = autoreview_prob < AUTOREVIEWER_CUTOFFS['reject_above']

    if reblog_worthy and not reblog_worthy_autoreview_prob:
        explanation = f"NOT reblogging {post_identifier} from dash: "
        explanation += f"\n\tautoreview_prob: {autoreview_prob:.1%} vs. {AUTOREVIEWER_CUTOFFS['reject_above']:.1%}"

    reblog_worthy = reblog_worthy and reblog_worthy_autoreview_prob

    if verbose:
        print(
            f"got prob {prob:.1%} for post {post_payload.get('id')} from {post_payload.get('blog_name')}"
        )
        print(
            f"follower_mult {follower_mult:.2f} * (prob {prob:.1%} + mood_buff {mood_buff:.1%} + random_buff {random_buff:.1%}) = buffed_prob {buffed_prob:.1%}"
        )

        if neg_sentiment is not None:
            print(
                f"got neg_sentiment {neg_sentiment:.1%} for post {post_payload.get('id')} from {post_payload.get('blog_name')}"
            )

    if reblog_worthy and not response_cache.is_handled(post_identifier):
        # explain choice
        explanation = f"reblogging {post_identifier} from dash: "
        explanation += (
            f"\n\tbuffed_prob: {buffed_prob:.0%} vs. {DASH_REBLOG_SELECTION_CUTOFF:.1%}"
        )
        if reblog_worthy_taggedme:
            explanation += f"\n\toverriding prob: i'm tagged in comment"
        if neg_sentiment is not None:
            explanation += f"\n\tneg_sentiment: {neg_sentiment:.0%} vs. {DASH_REBLOG_MAX_NEG_SENTIMENT:.0%}"
        explanation += f"\n\tautoreview_prob: {autoreview_prob:.1%} vs. {AUTOREVIEWER_CUTOFFS['reject_above']:.1%}"
        print(explanation)
    return reblog_worthy


def is_reblog_worthy_on_dash(
    post_payload,
    response_cache,
    loop_persistent_data,
    mood_value,
    follower_multipliers,
    verbose=True,
):
    statically_reblog_worthy = is_statically_reblog_worthy_on_dash(
        post_payload, response_cache, loop_persistent_data, verbose=verbose
    )

    if not statically_reblog_worthy:
        return False

    dynamically_reblog_worthy = is_dynamically_reblog_worthy_on_dash(
        post_payload,
        response_cache,
        loop_persistent_data,
        mood_value,
        follower_multipliers,
        verbose=verbose,
    )

    return dynamically_reblog_worthy


def review_statically_worthy_dashboard_post(
    post_payload,
    loop_persistent_data,
    response_cache,
    updated_last_seen_ts,
    mood_value,
    follower_multipliers=None,
):
    post_identifier = PostIdentifier(post_payload["blog_name"], str(post_payload["id"]))
    loop_persistent_data.reblog_keys[post_identifier] = post_payload["reblog_key"]

    is_reblog_worthy = is_dynamically_reblog_worthy_on_dash(
        post_payload=post_payload,
        response_cache=response_cache,
        loop_persistent_data=loop_persistent_data,
        mood_value=mood_value,
        follower_multipliers=follower_multipliers,
        verbose=VERBOSE_LOGS,
    )

    if is_reblog_worthy:
        loop_persistent_data.reblog_worthy_dash_posts.add(post_identifier)
        loop_persistent_data.timestamps[post_identifier] = post_payload["timestamp"]

    updated_last_seen_ts = max(updated_last_seen_ts, post_payload["timestamp"])
    return loop_persistent_data, response_cache, updated_last_seen_ts


def review_reblogs_from_me(note_payloads, loop_persistent_data, response_cache):
    note_payload_iter = (
        tqdm(note_payloads[::-1]) if len(note_payloads) > 5 else note_payloads[::-1]
    )
    for r in note_payload_iter:
        reblog_identifier = PostIdentifier(r["blog_name"], r["post_id"])

        r_to_me = r.get("blog_name") == blogName

        post2 = response_cache.query(
            CachedResponseType.POSTS, reblog_identifier, care_about_notes=False
        )
        if post2 is None:
            if (
                response_cache.client.request.consumer_key
                != dashboard_client.request.consumer_key
            ):
                if dashboard_client.get_ratelimit_data()["effective_remaining"] > 0:
                    print(
                        f"got non-200 response for {reblog_identifier}, trying with dashboard client"
                    )
                    prev_client = response_cache.client
                    response_cache.client = dashboard_client
                    post2 = response_cache.query(
                        CachedResponseType.POSTS,
                        reblog_identifier,
                        care_about_notes=False,
                    )
                    print(f"did we succeed: {post2 is not None}")
                    response_cache.client = prev_client
        if post2 is None:
            print(f"skipping {reblog_identifier}: non-200 response")
            continue
        loop_persistent_data.reblog_keys[reblog_identifier] = post2["reblog_key"]

        is_reblog_worthy = False

        if r_to_me:
            # reblog to me: need to mark ancestor as handled
            ancestors = [
                trail_item
                for trail_item in post2["trail"]
                if trail_item["blog"]["name"] != blogName
            ]
            if len(ancestors) > 0:
                direct_ancestor = ancestors[-1]
                response_cache.mark_handled(
                    PostIdentifier(
                        direct_ancestor["blog"]["name"], direct_ancestor["post"]["id"]
                    )
                )
        else:
            is_reblog_worthy = is_reblog_worthy_when_responding(
                post_payload=post2, note_payload=r, verbose=VERBOSE_LOGS
            )

        if is_reblog_worthy:
            loop_persistent_data.reblogs_from_me.add(reblog_identifier)
            loop_persistent_data.timestamps[reblog_identifier] = r["timestamp"]

            user_input_identifier = UserInputIdentifier(
                input_type=UserInputType.REBLOG,
                blog_name=r["blog_name"],
                id_=r["post_id"],
                timestamp=r["timestamp"],
            )
            text_for_sentiment = r.get("added_text")
            if text_for_sentiment is None:
                if VERBOSE_LOGS:
                    print(
                        f"couldn't find text for sentiment (added_text) in {user_input_identifier}"
                    )
                    print(f"have note payload {r}")
            elif response_cache.get_cached_user_input_sentiment(user_input_identifier) is None:
                logit_diff = sentiment_logit_diffs_from_gpt2_service(
                    [text_for_sentiment]
                )[0]
                sent = logit_diff_to_allen_schema(logit_diff)
                sent["text_for_sentiment"] = text_for_sentiment
                response_cache.mark_user_input_sentiment(user_input_identifier, sent)
                print(
                    f"for {reblog_identifier}, recorded {sent} for\n\t{text_for_sentiment}"
                )
    return loop_persistent_data, response_cache


def get_relevant_replies_from_notes(
    post_payload, notes_payload, replies_to_handle, loop_persistent_data, response_cache
):
    if post_payload["id"] in NO_REBLOG_IDS:
        return replies_to_handle, loop_persistent_data, response_cache
    for ix, n in enumerate(notes_payload):
        if n["type"] != "reply":
            continue

        # find the post the reply was "replying to" :(
        reply_context_post_notes = [
            n2 for n2 in notes_payload[ix:] if n2["blog_name"] == blogName
        ]
        if len(reply_context_post_notes) == 0:
            if VERBOSE_LOGS:
                print(
                    f"couldn't find context self-post for reply {n}: no self-posts in notes"
                )
            continue

        if reply_context_post_notes[0]["type"] == "reblog":
            reply_context_post_id = reply_context_post_notes[0]["post_id"]
        else:
            # need to find id for the origin post
            if len(post_payload["trail"]) == 0:
                # assume `post` is the origin post
                reply_context_post_id = post_payload["id"]
            else:
                trail_entries_from_me = [
                    entry
                    for entry in post_payload["trail"]
                    if entry.get("blog", {}).get("name") == blogName
                ]
                reply_context_post_id = int(trail_entries_from_me[0]["post"]["id"])

        if reply_context_post_id in NO_REBLOG_IDS:
            return replies_to_handle, loop_persistent_data, response_cache

        reply_context_post_identifier = PostIdentifier(blogName, reply_context_post_id)
        reply_context_post = response_cache.query(
            CachedResponseType.POSTS,
            reply_context_post_identifier,
            care_about_notes=False,
        )

        reply_identifier = ReplyIdentifier(
            n["blog_name"], reply_context_post_id, n["timestamp"]
        )

        # check whether we follow the user
        am_following_user = n["blog_name"] in loop_persistent_data.follower_names

        # is user a "frequent replier"
        is_frequent_replier = n["blog_name"] in REPLY_USER_AUTO_ACCEPT_LIST

        # check users tagged
        has_tag = n.get("reply_text", "").lstrip(" ").startswith("@")
        am_tagged = n.get("reply_text", "").lstrip(" ").startswith(f"@{blogName}")
        other_tagged = has_tag and (not am_tagged)

        # check "safe" conditions: (i am OP) AND (no reblogs before this note)
        am_OP = post_payload.get("source_title", blogName) == blogName
        no_reblogs_before_reply = not any(
            [n2.get("type", "") == "reblog" for n2 in notes_payload[ix:]]
        )
        is_safe = (am_OP) and (no_reblogs_before_reply)

        # always okay if they tagged us, never okay if they tagged someone else, otherwise check if following
        okay_to_reply = (
            am_following_user or is_frequent_replier or is_safe or am_tagged
        ) and not (other_tagged)
        if VERBOSE_LOGS:
            print(
                f"okay_to_reply={okay_to_reply} (am_following_user={am_following_user}, is_frequent_replier={is_frequent_replier}, is_safe={is_safe}, am_tagged={am_tagged}, other_tagged={other_tagged}) for\n\t{n}\n"
            )
        if not okay_to_reply:
            continue

        user_input_identifier = UserInputIdentifier(
            input_type=UserInputType.REPLY,
            blog_name=n["blog_name"],
            id_=post_payload["id"],
            timestamp=n["timestamp"],
        )
        do_get_sentiment = False
        if (
            response_cache.get_cached_user_input_sentiment(user_input_identifier)
            is None
        ):
            do_get_sentiment = True
        elif "text_for_sentiment" not in response_cache.get_cached_user_input_sentiment(
            user_input_identifier
        ):
            print(
                f"re-doing sentiment for {user_input_identifier} since 'text_for_sentiment' not found"
            )
            do_get_sentiment = True
        if do_get_sentiment:

            text_for_sentiment = n.get("reply_text")
            if text_for_sentiment is None:
                if VERBOSE_LOGS:
                    print(
                        f"couldn't find text for sentiment (reply_text) in {reply_identifier}"
                    )
                    print(f"have note payload {n}")
            elif response_cache.get_cached_user_input_sentiment(user_input_identifier) is None:
                logit_diff = sentiment_logit_diffs_from_gpt2_service(
                    [text_for_sentiment]
                )[0]
                sent = logit_diff_to_allen_schema(logit_diff)
                sent["text_for_sentiment"] = text_for_sentiment
                response_cache.mark_user_input_sentiment(user_input_identifier, sent)
                print(
                    f"for {reply_identifier}, recorded {sent} for\n\t{text_for_sentiment}"
                )

        if not response_cache.is_reply_handled(reply_identifier):
            replies_to_handle.add(reply_identifier)

            # we need reblog_key for our own post here
            loop_persistent_data.reblog_keys[reply_identifier] = reply_context_post[
                "reblog_key"
            ]
            loop_persistent_data.reply_metadata[reply_identifier] = {}
            loop_persistent_data.reply_metadata[reply_identifier]["reply_note"] = n
            loop_persistent_data.reply_metadata[reply_identifier][
                "post"
            ] = reply_context_post
    return replies_to_handle, loop_persistent_data, response_cache


def check_notifications(n_to_check=250, after_ts=0):
    base_url = private_client.request.host + f"/v2/blog/{blogName}/notifications"
    request_kwargs = dict(
        allow_redirects=False,
        headers=private_client.request.headers,
        auth=private_client.request.oauth,
    )

    getter = lambda url: requests.get(url, **request_kwargs).json()["response"]
    updater = lambda page: [
        item for item in page["notifications"] if item["timestamp"] > after_ts
    ]
    n = []

    with LogExceptionAndSkip("check notifications"):
        page = getter(base_url)
        delta = updater(page)

        while len(n) < n_to_check and len(delta) > 0:
            n += delta
            print(f"{len(n)}/{n_to_check}")
            time.sleep(0.1)
            url = private_client.request.host + page["_links"]["next"]["href"]
            page = getter(url)
            delta = updater(page)

    return n


def do_reblog_reply_handling(
    loop_persistent_data: LoopPersistentData,
    response_cache: ResponseCache,
    n_posts_to_check: int,
    is_dashboard: bool = False,
    pseudo_dashboard: bool = False,
    mood_value: float = None,
):
    relevant_client = dashboard_client if is_dashboard else response_cache.client
    count_check_requests_start = relevant_client.get_ratelimit_data()["day"][
        "remaining"
    ]

    def dashboard_post_getter(**kwargs):
        response = response_cache.record_response_to_cache(
            dashboard_client.dashboard(**kwargs), care_about_notes=False
        )
        posts = response["posts"]
        next_offset = kwargs["offset"] + len(posts)
        return posts, next_offset

    def reblogs_post_getter(**kwargs):
        response = response_cache.record_response_to_cache(
            response_cache.client.posts(blogName, **kwargs), care_about_notes=False
        )
        posts = response["posts"]

        next_offset = None
        with LogExceptionAndSkip("get next offset for /posts"):
            next_offset = response["_links"]["next"]["query_params"]["offset"]
        if next_offset is None:
            next_offset = kwargs["offset"] + len(posts)  # fallback
            print(
                f"falling back to: old offset {kwargs['offset']} + page size {len(posts)} = {next_offset}"
            )
        return posts, next_offset

    if is_dashboard and not pseudo_dashboard:
        post_getter = dashboard_post_getter
        start_ts = max(DASH_START_TS, loop_persistent_data.last_seen_ts)
    elif is_dashboard:
        raise ValueError("pseudo-dashboard is deprecated")

        def _get_pseudo_dashboard(**kwargs):
            psd_limit = 1
            offsets = {
                blog_name: 0 for blog_name in loop_persistent_data.follower_names
            }

            post_payloads = []
            while len(post_payloads) < n_posts_to_check:
                shuffled_names = np.random.choice(
                    list(loop_persistent_data.follower_names),
                    len(loop_persistent_data.follower_names),
                    replace=False,
                )
                for blog_name in shuffled_names:
                    print(f"checking {blog_name}")
                    post_payloads.extend(
                        response_cache.client.posts(
                            blog_name, limit=psd_limit, offset=offsets[blog_name]
                        )["posts"]
                    )
                    offsets[blog_name] += psd_limit
                    print(
                        f"checked {blog_name} -> offset {offsets[blog_name]}, {len(post_payloads)} on pseudo-dash"
                    )
                    time.sleep(0.2)
            return post_payloads

        post_getter = _get_pseudo_dashboard
        start_ts = max(DASH_START_TS, loop_persistent_data.last_seen_ts)

        loop_persistent_data = update_follower_names(
            loop_persistent_data, response_cache
        )
    else:
        post_getter = reblogs_post_getter
        start_ts = REBLOG_START_TS

    follower_multipliers = None

    # we need follower names for reply relevance v2
    loop_persistent_data = update_follower_names(loop_persistent_data, response_cache)

    replies_to_handle = set()

    limit_ = min(50, n_posts_to_check)

    offset_ = 0

    ### get posts
    print(f"\nchecking {n_posts_to_check} posts, start_ts={start_ts}...\n")
    posts = []
    updated_last_seen_ts = loop_persistent_data.last_seen_ts

    next_posts, next_offset = post_getter(
        limit=limit_, offset=offset_, notes_info=(not is_dashboard)
    )
    next_ = [
        p
        for p in next_posts
        if p["timestamp"] > start_ts
        and p["id"] not in NO_REBLOG_IDS
        and not any(
            [
                int(trail_entry.get("post", {}).get("id", -1)) in NO_REBLOG_IDS
                for trail_entry in p.get("trail", [])
            ]
        )
    ]
    posts.extend(next_)
    offset_ = next_offset
    while len(next_) != 0 and len(posts) < n_posts_to_check:
        min_ts = min([p["timestamp"] for p in next_])
        print(f"got {len(next_)}, starting with {next_[0]['id']}, min_ts={min_ts}")
        time.sleep(0.1)
        next_posts, next_offset = post_getter(
            limit=limit_, offset=offset_, notes_info=(not is_dashboard)
        )
        next_ = [
            p
            for p in next_posts
            if p["timestamp"] > start_ts
            and p["id"] not in NO_REBLOG_IDS
            and not any(
                [
                    int(trail_entry.get("post", {}).get("id", -1)) in NO_REBLOG_IDS
                    for trail_entry in p.get("trail", [])
                ]
            )
        ]
        posts.extend(next_)
        offset_ = next_offset
    if len(next_) > 0:
        # TODO: DRY
        min_ts = min([p["timestamp"] for p in next_])
        print(f"got {len(next_)}, starting with {next_[0]['id']}, min_ts={min_ts}")

    print(f"{len(posts)} posts retrieved")

    if not is_dashboard:
        print("checking mentions...")
        notifications = check_notifications(
            n_to_check=loop_persistent_data.n_notifications_to_check,
            after_ts=loop_persistent_data.last_seen_ts_notifications,
        )

        if len(notifications) > 0:
            mentions = [
                item for item in notifications if item["type"] == "user_mention"
            ]

            for item in mentions:
                with LogExceptionAndSkip(f"handle mention {repr(item)}"):
                    mention_blogname = item["target_tumblelog_name"]
                    mention_post_id = int(item["target_post_id"])

                    # don't add if duplicate
                    if any([post["id"] == mention_post_id for post in posts]):
                        continue

                    pi = PostIdentifier(mention_blogname, mention_post_id)

                    if response_cache.is_handled(pi):
                        continue

                    print(f"reblogging from mentions: {pi}")

                    loop_persistent_data.reblogs_from_me.add(pi)
                    loop_persistent_data.timestamps[pi] = item["timestamp"]
                    mp = private_client.posts(mention_blogname, id=mention_post_id)[
                        "posts"
                    ][0]
                    loop_persistent_data.reblog_keys[pi] = mp["reblog_key"]

            # update last_seen_ts_notifications
            updated_last_seen_ts_notifications = max(
                [item["timestamp"] for item in notifications]
            )

            print(
                f"updating last_seen_ts_notifications: {loop_persistent_data.last_seen_ts_notifications} --> {updated_last_seen_ts_notifications} (+{updated_last_seen_ts_notifications-loop_persistent_data.last_seen_ts_notifications})"
            )
            loop_persistent_data.last_seen_ts_notifications = (
                updated_last_seen_ts_notifications
            )

    if is_dashboard:
        # batch up dash posts for side judgment computation
        statically_worthy_posts = []
        for post_ix, post in enumerate(tqdm(posts)):  # posts[:n_posts_to_check]
            if is_statically_reblog_worthy_on_dash(
                post, response_cache, loop_persistent_data, verbose=VERBOSE_LOGS
            ):
                statically_worthy_posts.append(post)
        print(f"{len(statically_worthy_posts)}/{len(posts)} statically reblog worthy")

        response_cache = batch_judge_dash_posts(
            statically_worthy_posts, response_cache
        )
    else:
        statically_worthy_posts = posts  # posts[:n_posts_to_check]

    ### loop through posts
    for post_ix, post in enumerate(tqdm(statically_worthy_posts)):
        post_identifier = PostIdentifier(post["blog_name"], post["id"])

        ### get reblogs to deal with

        if is_dashboard:
            (
                loop_persistent_data,
                response_cache,
                updated_last_seen_ts,
            ) = review_statically_worthy_dashboard_post(
                post,
                loop_persistent_data,
                response_cache,
                updated_last_seen_ts,
                mood_value,
                follower_multipliers,
            )
        else:
            try:
                notes = response_cache.query(
                    CachedResponseType.NOTES,
                    post_identifier,
                    expected_notes=post["note_count"],
                    notes_field=post.get("notes"),
                )
            except Exception as e:
                print(
                    f"encountered {repr(e)} trying to get notes for {post_identifier}, notes field was {post.get('notes')}"
                )
                continue

            reblogs = [
                n
                for n in notes
                if n["type"] == "reblog"
                and n.get("reblog_parent_blog_name", "") == blogName
                and not response_cache.is_handled(
                    PostIdentifier(n.get("blog_name", ""), n.get("post_id", ""))
                )
                and int(n["post_id"]) not in NO_REBLOG_IDS
            ]

            loop_persistent_data, response_cache = review_reblogs_from_me(
                note_payloads=reblogs,
                loop_persistent_data=loop_persistent_data,
                response_cache=response_cache,
            )

            ### get replies to deal with
            (
                replies_to_handle,
                loop_persistent_data,
                response_cache,
            ) = get_relevant_replies_from_notes(
                post, notes, replies_to_handle, loop_persistent_data, response_cache
            )

    if is_dashboard:
        reblogs_to_handle = [
            r
            for r in loop_persistent_data.reblog_worthy_dash_posts
            if (
                loop_persistent_data.timestamps[r] > start_ts
                and not response_cache.is_handled(r)
            )
        ]
    else:
        reblogs_to_handle = [
            r
            for r in loop_persistent_data.reblogs_from_me
            if (
                loop_persistent_data.timestamps[r] > start_ts
                and not response_cache.is_handled(r)
            )
        ]

    blog_names_to_reblog_idents = defaultdict(list)
    for r in reblogs_to_handle:
        blog_names_to_reblog_idents[r.blog_name].append(r)

    if not (is_dashboard):
        for bn, rs in blog_names_to_reblog_idents.items():
            for r in rs[1:]:
                print(f"\t * user equity rule*: saving {r} for later...")
                reblogs_to_handle.remove(r)

    blog_names_to_reply_idents = defaultdict(list)
    for r in replies_to_handle:
        blog_names_to_reply_idents[r.blog_name].append(r)

    for bn, rs in blog_names_to_reply_idents.items():
        for r in rs[1:]:
            print(f"\t * user equity rule*: saving {r} for later...")
            replies_to_handle.remove(r)

    print(f"{len(reblogs_to_handle)} reblogs to handle")
    if len(reblogs_to_handle) > 0:
        print(f"handling reblogs:")
        for item in reblogs_to_handle:
            print(f"\t{item}")

    print(f"{len(replies_to_handle)} replies to handle")
    if len(replies_to_handle) > 0:
        print(f"handling replies:")
        for item in replies_to_handle:
            print(f"\t{item}")

    reblog_reply_timestamps = {
        r: loop_persistent_data.timestamps[r] for r in reblogs_to_handle
    }
    reblog_reply_timestamps.update({ri: ri.timestamp for ri in replies_to_handle})
    time_ordered_idents = sorted(
        reblog_reply_timestamps.keys(), key=lambda r: reblog_reply_timestamps[r]
    )

    kept = time_ordered_idents[:MAX_POSTS_PER_STEP]
    excluded = time_ordered_idents[MAX_POSTS_PER_STEP:]

    if len(excluded) > 0:
        print(
            f"saving {len(excluded)} of {len(time_ordered_idents)} for later with MAX_POSTS_PER_STEP={MAX_POSTS_PER_STEP}"
        )
        for r in excluded:
            print(f"\t saving {r} for later...")

    kept_reblogs = [r for r in kept if r in reblogs_to_handle]
    kept_replies = [r for r in kept if r in replies_to_handle]

    reblogs_to_handle = kept_reblogs
    replies_to_handle = kept_replies

    if is_dashboard and len(kept) > 0 and len(excluded) > 0:
        last_handled_in_step_ts = max([reblog_reply_timestamps[r] for r in kept])
        if last_handled_in_step_ts < updated_last_seen_ts:
            print(
                f"rolling back updated_last_seen_ts: {updated_last_seen_ts} --> {last_handled_in_step_ts}"
            )
            updated_last_seen_ts = last_handled_in_step_ts
        else:
            print(
                f"weirdness: last_handled_in_step_ts {last_handled_in_step_ts} > updated_last_seen_ts{updated_last_seen_ts}"
            )

    # handle reblogs, replies
    loop_persistent_data, response_cache = respond_to_reblogs_replies(
        identifiers=reblogs_to_handle + list(replies_to_handle),
        reply_set=replies_to_handle,
        loop_persistent_data=loop_persistent_data,
        response_cache=response_cache,
        proba_threshold=DASH_REBLOG_CONTINUATION_CUTOFF if is_dashboard else None,
        is_user_input=(not is_dashboard),
    )

    if len(reblogs_to_handle + list(replies_to_handle)) > 0:
        do_rts(response_cache)

    ### post-check stuff

    count_check_requests_end = relevant_client.get_ratelimit_data()["day"]["remaining"]
    count_check_requests_diff = count_check_requests_start - count_check_requests_end
    print(f"used {count_check_requests_diff} requests in this check")

    if is_dashboard:
        # record calls for this check -- hack
        loop_persistent_data.requests_per_check_history[-1] += count_check_requests_diff

        # update last_seen_ts
        print(
            f"updating last_seen_ts: {loop_persistent_data.last_seen_ts} --> {updated_last_seen_ts} (+{updated_last_seen_ts-loop_persistent_data.last_seen_ts})"
        )
        loop_persistent_data.last_seen_ts = updated_last_seen_ts
    else:
        # record calls for this check
        loop_persistent_data.requests_per_check_history.append(
            count_check_requests_diff
        )

    return loop_persistent_data, response_cache


def parse_and_validate_review_command(s, max_field_length=50):
    _, _, user_argstring = s.partition(REVIEW_COMMAND)
    user_args = [arg.strip(" ") for arg in user_argstring.split(",")]
    user_args = [arg for arg in user_args if len(arg) > 0]

    is_valid = True
    if "<" in user_args or ">" in user_args:
        is_valid = False
    elif len(user_args) == 0 or len(user_args) > 3:
        is_valid = False
    elif len(user_args) == 3 and user_args[-1] not in ["1", "2", "3", "4", "5"]:
        is_valid = False
    elif any([len(arg) > max_field_length for arg in user_args]):
        is_valid = False

    return user_args, user_argstring, is_valid


def construct_review_question(user_args):
    control = DEFAULT_CSC["REVIEW_CHAR_FORUMLIKE"]
    if not control.endswith("\n"):
        control = control + "\n"

    q = ""
    for i, arg in enumerate(user_args):
        if i == 0:
            q = q + f"Title: <b>{arg}</b>\n"
        elif i == 1:
            q = q + f"Author: <b>{arg}</b>\n"
        elif i == 2:
            q = q + f"Rating: <b>{arg}/5 stars</b>\n\n"
    full_input = control + q
    return q, full_input


def handle_review_command(
    user_args, input_ident, asking_url, loop_persistent_data, response_cache
):
    question, full_input = construct_review_question(user_args)
    gpt2_output = answer_from_gpt2_service(
        data={
            "question": full_input,
            "asking_name": input_ident[1],
            "mood": determine_mood(response_cache),
            "exact_prompt": True,
            "forced_tags_string": "",
            "write_fic_override": 0,
            "write_review_override": 1,
        },
        loop_persistent_data=loop_persistent_data,
        no_timestamp=True,
        BEAMSPLIT_TESTING_FLAG=BEAMSPLIT_TESTING_FLAG,
    )

    log_data = gpt2_output
    log_data["post_type"] = "review"
    log_data["input_ident"] = input_ident
    log_data["question"] = full_input

    post = (
        question
        + gpt2_output["post"]
        + REVIEW_COMMAND_EXPLAINER_STRING.format(
            asking_name=input_ident[1], asking_url=asking_url
        )
    )
    make_text_post(
        blogName,
        post=post,
        tags=[],
        log_data=log_data,
        to_queue=False,
        to_drafts=REVIEW_COMMAND_TESTING,
    )


def do_ask_handling(loop_persistent_data, response_cache):
    submissions = private_client.submission(blogName)["posts"]

    # TODO: find a good way to ensure this happens in general for all consumption
    submissions = [simulate_legacy_payload(p) for p in submissions]

    n_asks = len(submissions)
    print(f"processing {n_asks} new asks")
    print()

    submissions = [
        x
        for x in submissions
        if roll_for_limited_users(x["asking_name"], text=x["question"])
    ]

    blog_names_to_asks = defaultdict(list)
    for r in submissions[::-1]:
        blog_names_to_asks[r["asking_name"]].append(r)

    for bn, rs in blog_names_to_asks.items():
        for r in rs[1:]:
            print(
                f"\t * user equity rule*: saving {r['asking_name']}, question={r['question']} for later..."
            )
            submissions.remove(r)

    kept = submissions[-MAX_POSTS_PER_STEP:]
    excluded = submissions[:-MAX_POSTS_PER_STEP]
    if len(excluded) > 0:
        print(
            f"saving {len(excluded)} of {len(submissions)} for later with MAX_POSTS_PER_STEP={MAX_POSTS_PER_STEP}"
        )
        for r in excluded:
            print(
                f"\t saving {r['asking_name']}, question={r['question']} for later..."
            )
    submissions = kept

    for x in submissions[::-1]:
        if x.get("summary", "") == FOLLOW_COMMAND:
            with LogExceptionAndSkip("follow"):
                loop_persistent_data = update_follower_names_v2(
                    loop_persistent_data, response_cache
                )
                loop_persistent_data.follower_names.add(x["asking_name"])
                dashboard_client.follow(x["asking_name"])
                if not BEAMSPLIT_TESTING_FLAG:
                    private_client.delete_post(blogName, x["id"])
                print(f"followed {x['asking_name']}")
        elif x.get("summary", "") == UNFOLLOW_COMMAND:
            with LogExceptionAndSkip("unfollow"):
                loop_persistent_data = update_follower_names_v2(
                    loop_persistent_data, response_cache
                )
                loop_persistent_data.follower_names.remove(x["asking_name"])
                dashboard_client.unfollow(x["asking_name"])
                if not BEAMSPLIT_TESTING_FLAG:
                    private_client.delete_post(blogName, x["id"])
                print(f"unfollowed {x['asking_name']}")
        elif x.get("summary", "") == MOOD_GRAPH_COMMAND:
            path = create_mood_graph(
                response_cache,
                start_time=datetime.now() - pd.Timedelta(days=MOOD_GRAPH_N_DAYS),
                end_time=datetime.now(),
            )
            state = "published"
            if x["asking_name"] == "nostalgebraist":
                state = "draft"
            if BEAMSPLIT_TESTING_FLAG:
                state = "draft"
            private_client.create_photo(
                blogName,
                state=state,
                data=path,
                caption=MOOD_GRAPH_EXPLAINER_STRING.format(
                    days_string=MOOD_GRAPH_DAYS_STRING,
                    asking_name=x["asking_name"],
                    asking_url=x["asking_url"],
                ),
            )
            if not BEAMSPLIT_TESTING_FLAG:
                private_client.delete_post(blogName, x["id"])
        elif x.get("summary", "").startswith(REVIEW_COMMAND):
            with LogExceptionAndSkip("write review"):
                user_args, user_argstring, is_valid = parse_and_validate_review_command(
                    inverse_format_post_for_api(x["summary"])
                )

                if not is_valid:
                    print(f"malformed_review_command: {user_argstring} --> {user_args}")
                else:
                    input_ident = (x["id"], x["asking_name"])
                    handle_review_command(
                        user_args,
                        input_ident,
                        x["asking_url"],
                        loop_persistent_data,
                        response_cache,
                    )
                    private_client.delete_post(blogName, x["id"])
        else:
            for k in [
                "id",
                "timestamp",
                "asking_name",
                "summary",
                "question",
            ]:
                print(f"{k}: {x[k]}")

            question = x["question"]

            question = find_images_and_sub_text(question)

            question = inverse_format_post_for_api(question)

            forced_tags_string = ""

            write_fic_override = 0
            if FIC_TRIGGER:
                fic_trigger_criterion = any(
                    [
                        subs in x["question"].lower()
                        for subs in [
                            "tell me a story",
                            "tell me the story of",
                            "a story about",
                            "write a story",
                            "write a fanfic",
                            "write a fic",
                        ]
                    ]
                )
                print(
                    f"fic_trigger_criterion: {fic_trigger_criterion} with x['question']: {x['question']}"
                )
                if FIC_TRIGGER_TESTING:
                    fic_trigger_criterion = fic_trigger_criterion and (
                        x["asking_name"] == "nostalgebraist"
                    )
                    print(
                        f"fic_trigger_criterion: {fic_trigger_criterion} with x['asking_name']: {x['asking_name']}"
                    )

                if fic_trigger_criterion:
                    print("fic_trigger_criterion passed")
                    # forced_tags_string += " #original fiction"
                    write_fic_override = 1

            user_input_identifier = UserInputIdentifier(
                input_type=UserInputType.ASK,
                blog_name=x["asking_name"],
                id_=x["id"],
                timestamp=x["timestamp"],
            )
            do_get_sentiment = False
            if (
                response_cache.get_cached_user_input_sentiment(user_input_identifier)
                is None
            ):
                do_get_sentiment = True
            elif (
                "text_for_sentiment"
                not in response_cache.get_cached_user_input_sentiment(
                    user_input_identifier
                )
            ):
                print(
                    f"re-doing sentiment for {user_input_identifier} since 'text_for_sentiment' not found"
                )
                do_get_sentiment = True
            if do_get_sentiment:
                text_for_sentiment = question
                if text_for_sentiment is None:
                    if VERBOSE_LOGS:
                        print(
                            f"couldn't find text for sentiment (question) in {user_input_identifier}"
                        )
                        print(f"have submission payload {x}")
                elif response_cache.get_cached_user_input_sentiment(user_input_identifier) is None:
                    logit_diff = sentiment_logit_diffs_from_gpt2_service(
                        [text_for_sentiment]
                    )[0]
                    sent = logit_diff_to_allen_schema(logit_diff)
                    sent["text_for_sentiment"] = text_for_sentiment
                    response_cache.mark_user_input_sentiment(
                        user_input_identifier, sent
                    )
                    print(
                        f"for {user_input_identifier}, recorded {sent} for\n\t{text_for_sentiment}"
                    )

            gpt2_output = answer_from_gpt2_service(
                data={
                    "question": question,
                    "asking_name": x["asking_name"],
                    "mood": determine_mood(response_cache),
                    "forced_tags_string": forced_tags_string,
                    "write_fic_override": write_fic_override,
                    "return_all_conts": 1,
                },
                loop_persistent_data=loop_persistent_data,
                BEAMSPLIT_TESTING_FLAG=BEAMSPLIT_TESTING_FLAG,
            )

            if (
                response_cache.get_cached_user_input_sentiment(user_input_identifier)
                is not None
            ):
                sent = response_cache.get_cached_user_input_sentiment(
                    user_input_identifier
                )
                if sent.get("generated_logit_diff") is not None:
                    print(
                        f"not overwriting existing mood effects for {user_input_identifier}"
                    )
                else:
                    # TODO: DRY
                    sent["generated_ts"] = datetime.now()
                    generated_pos_sent = gpt2_output.get("all_pos_sentiment")

                    if generated_pos_sent:
                        sent["generated_logit_diff"] = [
                            pos_sent_to_logit_diff(entry)
                            for entry in generated_pos_sent
                        ]
                        sent["p75_generated_logit_diff"] = np.percentile(sent["generated_logit_diff"], 75)
                    response_cache.mark_user_input_sentiment(
                        user_input_identifier, sent
                    )
                    show_unit_mood_inputs(response_cache, user_input_identifier)
            log_data = gpt2_output
            log_data["post_type"] = "ask"
            log_data["input_ident"] = (x["id"], x["asking_name"])
            log_data["question"] = question
            answer_ask(
                blogName,
                ask_id=x["id"],
                asking_name=x["asking_name"],
                question=question,
                answer=gpt2_output["post"],
                tags=gpt2_output["tags"],
                log_data=log_data,
                autoreview_proba=gpt2_output["autoreview_proba"],
                reject_action="rts",
            )
    return loop_persistent_data, response_cache, n_asks


def do_queue_handling(loop_persistent_data, response_cache):
    queue = private_client.queue(blogName, limit=20)["posts"]

    n_posts_in_queue = len(queue)
    print(f"{n_posts_in_queue} posts in queue")

    if n_posts_in_queue < WRITE_POSTS_WHEN_QUEUE_BELOW:
        for textpost_ix in range(N_TO_WRITE):
            dt = next_queued_post_time()
            mood_for_queue_writing = determine_mood(response_cache, dt=dt)

            print(f"writing new text post... ({textpost_ix}/{N_TO_WRITE})")

            gpt2_output, loop_persistent_data = text_post_from_gpt2_service(
                loop_persistent_data=loop_persistent_data,
                mood=mood_for_queue_writing,
                ts=dt,
                BEAMSPLIT_TESTING_FLAG=BEAMSPLIT_TESTING_FLAG,
            )

            log_data = gpt2_output
            log_data["post_type"] = "textpost"
            log_data["input_ident"] = None
            log_data["question"] = None
            make_text_post(
                blogName,
                post=gpt2_output["post"],
                tags=gpt2_output["tags"],
                log_data=gpt2_output,
                autoreview_proba=gpt2_output["autoreview_proba"],
                reject_action="do_not_post"
            )

        n_posts_in_queue = len(private_client.queue(blogName, limit=20)["posts"])
        print(f"now {n_posts_in_queue} posts in queue")

        response_cache = do_rts(response_cache)
    return loop_persistent_data, response_cache


def do_rts(response_cache):
    drafts = private_client.drafts(blogName, reblog_info=True)["posts"]
    to_send_back = [p for p in drafts if RTS_COMMAND in p["tags"]]
    to_autopub = [p for p in drafts if ACCEPT_COMMAND in p["tags"]]

    n_drafts = len(drafts)
    n_rts = len(to_send_back)
    n_autopub = len(to_autopub)
    n_unmarked = n_drafts - n_rts - n_autopub

    print(f"RTS: {n_rts}/{n_drafts}")
    print(f"AUTOPUB: {n_autopub}/{n_drafts}")
    print(f"UNMARKED: {n_unmarked}/{n_drafts}")

    for p in to_send_back:
        pid = p.get("id")
        print(f"trying to RTS {pid}...")

        p_body = get_body(p)

        if "reblogged_from_id" in p and "reblogged_from_name" in p:
            pi = PostIdentifier(p["reblogged_from_name"], p["reblogged_from_id"])
            print(f"\tidentified as reblog from {pi}")
            print(
                f"\tresponse_cache.is_handled({pi}) before: {response_cache.is_handled(pi)}"
            )
            response_cache.mark_unhandled(pi)
            print(
                f"\tresponse_cache.is_handled({pi}) after: {response_cache.is_handled(pi)}"
            )
            private_client.delete_post(blogName, id=pid)
        elif p.get("type") == "answer":
            print(f"\tidentified as answer to ask")
            private_client.edit_post(blogName, id=pid, state="submission")
        elif "replied to your post" in p_body:
            print(f"\tidentified as answer to reply")
            reply_post_id, replier_name = post_body_find_reply_data(p_body)

            if reply_post_id is None or replier_name is None:
                print(f"couldn't RTS: couldn't find reply post ID")
            else:
                possible_rids = [
                    ri
                    for ri in response_cache.replies_handled
                    if ri.blog_name == replier_name
                    and str(ri.id_) == str(reply_post_id)
                ]
                if len(possible_rids) == 0:
                    print(
                        f"couldn't RTS: found 0 reply idents that match replier_name={replier_name}, reply_post_id={reply_post_id}"
                    )
                    continue
                elif len(possible_rids) == 1:
                    rid = possible_rids[0]
                else:
                    timestamps_to_possible_rids = {
                        ri.timestamp: ri for ri in possible_rids
                    }

                    reply_post_ident = PostIdentifier(blogName, reply_post_id)
                    reply_post_notes = response_cache.query(
                        CachedResponseType.NOTES, reply_post_ident
                    )

                    relevant_notes = [
                        n
                        for n in reply_post_notes
                        if n.get("type") == "reply"
                        and "reply_text" in n
                        and n.get("blog_name") == replier_name
                        and n.get("timestamp", -1) in timestamps_to_possible_rids
                    ]
                    text_matching_notes = [
                        {k: n[k] for k in ["reply_text", "timestamp"]}
                        for n in relevant_notes
                        if n["reply_text"] in p_body
                    ]
                    if len(text_matching_notes) > 0:
                        print(f"picking between {text_matching_notes}")
                        longest_text_matching_note = sorted(
                            text_matching_notes, key=lambda n_: len(n_["reply_text"])
                        )[-1]
                        print(f"chose {longest_text_matching_note}")
                        chosen_ts = longest_text_matching_note["timestamp"]
                        rid = timestamps_to_possible_rids[chosen_ts]
                    else:
                        print(
                            f"couldn't RTS: couldn't find note matches to {possible_rids}"
                        )

                print(f"\tidentified as answer to {rid}")
                response_cache.cache["replies_handled"].remove(rid)
                private_client.delete_post(blogName, id=pid)
        else:
            print(f"don't know how to RTS {pid}!")

    # TODO: make if/else here less awful
    for p in to_autopub[-1:]:
        pid = p.get("id")
        print(f"trying to AUTOPUB {pid}...")
        if "tags" in p:
            tags = p["tags"]
            if ACCEPT_COMMAND in tags:
                tags = [t for t in tags if t != ACCEPT_COMMAND]
                r = private_client.edit_post(blogName, id=pid, tags=tags, state="draft")
                if 'errors' in r:
                    print(f'api error [editing]: response {repr(r)}')
                else:
                    r = private_client.edit_post(blogName, id=pid, state="published")
                    if 'errors' in r:
                        print(f'api error [publishing]: response {repr(r)}')
                    else:
                        print(f"AUTOPUBed {pid}")
            else:
                print(f"could not find ACCEPT_COMMAND in tags, have tags {tags}")
        else:
            print(f"could not find tags, have keys {p.keys()}")

    return response_cache


def mainloop(loop_persistent_data: LoopPersistentData, response_cache: ResponseCache):
    response_cache = do_rts(response_cache)

    ### decide whether we'll do the reblog/reply check

    requests_needed_to_check = np.percentile(
        loop_persistent_data.requests_per_check_history[-30:], 75
    )
    checkprob, response_cache = _compute_checkprob_from_ratelimits(
        requests_needed_to_check, response_cache
    )

    print(
        f"using checkprob: {checkprob:.1%}, with last_seen_ts={loop_persistent_data.last_seen_ts}"
    )

    check_roll = np.random.rand()
    if check_roll >= checkprob:
        print(f"skipping check this time ({check_roll:.2f} >= {checkprob})...")
        n_posts_to_check = 0
    else:
        print(f"checking ({check_roll:.2f} < {checkprob:.2f})...")
        n_posts_to_check = loop_persistent_data.n_posts_to_check_base

    n_posts_to_check_dash = loop_persistent_data.n_posts_to_check_dash

    def _mainloop_asks_block(loop_persistent_data, response_cache, save_after=True):
        relevant_ratelimit_data = private_client.get_ratelimit_data()
        if relevant_ratelimit_data["effective_remaining"] > 0:
            loop_persistent_data, response_cache, n_asks = do_ask_handling(
                loop_persistent_data, response_cache
            )
            if (n_asks > 0):
                if save_after:
                    response_cache.save()
                    image_analysis_cache.save()
        return loop_persistent_data, response_cache

    ### do asks check
    loop_persistent_data, response_cache = _mainloop_asks_block(
        loop_persistent_data, response_cache
    )

    ### do reblog/reply check
    if n_posts_to_check > 0:
        # reblogs, replies
        loop_persistent_data, response_cache = do_reblog_reply_handling(
            loop_persistent_data, response_cache, n_posts_to_check
        )
        response_cache.save()
        image_analysis_cache.save()

        ### do another asks check
        loop_persistent_data, response_cache = _mainloop_asks_block(
            loop_persistent_data, response_cache
        )

    if n_posts_to_check > 0:
        # dash check
        if dashboard_client.get_ratelimit_data()["effective_remaining"] > 0:
            print("checking dash...")
            _, mood_value = determine_mood(response_cache, return_mood_value=True)
            loop_persistent_data, response_cache = do_reblog_reply_handling(
                loop_persistent_data,
                response_cache,
                n_posts_to_check_dash,
                is_dashboard=True,
                mood_value=mood_value,
            )
        else:
            print("skipping dash check this time")

        ### do another asks check
        loop_persistent_data, response_cache = _mainloop_asks_block(
            loop_persistent_data, response_cache, save_after=False
        )

    relevant_ratelimit_data = private_client.get_ratelimit_data()
    if relevant_ratelimit_data["effective_remaining"] > 0:
        response_cache.save()
        image_analysis_cache.save()

        ### do rts
        response_cache = do_rts(response_cache)

        # inform us about drafts
        drafts = private_client.drafts(blogName)["posts"]
        print(f"{len(drafts)} waiting for review")

        ### do queue check
        loop_persistent_data, response_cache = do_queue_handling(
            loop_persistent_data, response_cache
        )
    else:
        print("skipping asks, queue, drafts until we're no longer rate limited")
        print(relevant_ratelimit_data)

    if MOOD_DYN:
        print("current mood:")
        determine_mood(response_cache)

    print()
    return loop_persistent_data, response_cache


def load_retention():
    with open("data/retention_stack.pkl.gz", "rb") as f:
        retention_stack = pickle.load(f)

    retention_stack = apply_retention_cutoff(retention_stack)

    return retention_stack


if __name__ == "__main__":
    # pr_boot = cProfile.Profile()
    # pr_boot.enable()

    response_cache = ResponseCache.load(tank_client)

    retention_stack = load_retention()

    loop_persistent_data = LoopPersistentData(
        retention_stack=retention_stack,
    )

    # _pr_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # pr_boot.dump_stats(f"profiling_data/boot/{_pr_name}")
    # pr_boot.disable()
    #
    # pr_main = cProfile.Profile()
    # pr_main.enable()

    while True:
        try:
            loop_persistent_data, response_cache = mainloop(
                loop_persistent_data, response_cache
            )
            time.sleep(sleep_time())
            send_alldone()
            # _pr_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            # pr_main.dump_stats(f"profiling_data/main/{_pr_name}")
            # pr_main.enable()
        except (requests.exceptions.ConnectionError, KeyError, ValueError):
            print("hit an error, waiting for a little while...")
            time.sleep(sleep_time(multiplier=5))
            send_alldone()
        except KeyboardInterrupt:
            send_alldone()
            # pr_main.disable()
            raise KeyboardInterrupt
