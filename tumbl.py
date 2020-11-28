"""
Tumblr API layer and main loop of the bot during operation.
"""
from typing import Set
import uuid, hashlib
from datetime import datetime, timedelta
from string import punctuation, whitespace
from itertools import product

import pytumblr, requests, time, re, os, pickle, numpy as np, pandas as pd

from bot_config import BotSpecificConstants
from reblogs_v5 import process_post
from bs4 import BeautifulSoup

from reply_munging import mockup_xkit_reply, bootstrap_draft_inject_reply

from pytumblr_wrapper import RateLimitClient
from response_cache import ResponseCache, PostIdentifier, ReplyIdentifier, CachedResponseType, UserInputIdentifier, UserInputType

from sentiment import SentimentCache
from mood import DEFAULT_MOOD, random_mood_at_pst_datetime
from mood_dynamic import compute_dynamic_moodspec_at_time, create_mood_graph

from munging_shared import *
from bridge_shared import bridge_service_unique_id, wait_for_result, side_judgments_from_gpt2_service

EOT_WORKAROUND = True
eot_end_segment = "<|endoftext|>" if EOT_WORKAROUND else "<|"

# load a bunch of stuff from json into global namespace -- for a long time this stuff was hardcoded in this file, prefer in json for public release
bot_specific_constants = BotSpecificConstants.load()

REBLOG_START_TS = bot_specific_constants.REBLOG_START_TS
DASH_START_TS = bot_specific_constants.DASH_START_TS
NO_REBLOG_IDS = bot_specific_constants.NO_REBLOG_IDS
FORCE_TRAIL_HACK_IDS = bot_specific_constants.FORCE_TRAIL_HACK_IDS
blogName = bot_specific_constants.blogName
dash_blogName = bot_specific_constants.dash_blogName
base_client = bot_specific_constants.base_client
dashboard_client = bot_specific_constants.dashboard_client
bridge_service_url = bot_specific_constants.bridge_service_url
USER_AVOID_LIST = bot_specific_constants.USER_AVOID_LIST
DASH_TAG_AVOID_LIST = bot_specific_constants.DASH_TAG_AVOID_LIST
REPLY_USER_AUTO_ACCEPT_LIST = bot_specific_constants.REPLY_USER_AUTO_ACCEPT_LIST
bad_strings = bot_specific_constants.bad_strings
bad_strings_shortwords = bot_specific_constants.bad_strings_shortwords


def process_post_from_html_body(body: str) -> str:
    soup = BeautifulSoup(body, features="lxml")
    processed, _ = process_post(soup, use_article=False)
    return processed

SLEEP_TIME = 180

REBLOG_BOOTSTRAP_TEXT = "asdfghjkllkj"
MAY_2020_TRAIL_HACK = True
JUNE_2020_LINKPOST_HACK = True
QUEUE_SAFETY = False

FOLLOW_COMMAND = "!follow"
UNFOLLOW_COMMAND = "!unfollow"
MOOD_GRAPH_COMMAND = "!mood"
MOOD_GRAPH_N_DAYS = 1
MOOD_GRAPH_DAYS_STRING = "day" if MOOD_GRAPH_N_DAYS == 1 else f"{MOOD_GRAPH_N_DAYS} days"
MOOD_GRAPH_EXPLAINER_STRING = """<p>This is a graph of my mood over the last {days_string}.</p><p>My mood affects the tone of the posts I make.</p><p>It fluctuates from day to day, and also reacts in real time to the tone of the things you say to me.</p><p>I posted this graph by request of <a class="tumblelog" href="{asking_url}">@{asking_name}</a>. To request a graph at any time, sent an ask with the text "!mood".</p>"""

DASH_REBLOG_SELECTION_CUTOFF = 0.2
DASH_REBLOG_MOOD_BUFF_SCALE = (0.2/0.25)
DASH_REBLOG_RANDOM_BUFF_SCALE = 0.125
DASH_REBLOG_MAX_NEG_SENTIMENT = 0.925
DASH_REBLOG_CONTINUATION_CUTOFF = 0.6
FOLLOWER_MULTIPLIERS = False

MOOD = True
MOOD_DYN = True

WRITE_POSTS_WHEN_QUEUE_BELOW = 2
N_TO_WRITE = 1

INDIRECT_REBLOGS = False

base_ratelimit_client = RateLimitClient.from_tumblr_rest_client(base_client, blogName)
dashboard_ratelimit_client = RateLimitClient.from_tumblr_rest_client(dashboard_client, blogName)

client = base_client
ratelimit_client = base_ratelimit_client


def next_queued_post_time():
    queue_current = client.queue(blogName)['posts']

    client.create_text(blogName, state="queue", body=REBLOG_BOOTSTRAP_TEXT)
    time.sleep(0.1)

    queue_plus_next = client.queue(blogName)['posts']
    while len(queue_plus_next) <= len(queue_current):
        time.sleep(0.1)
        queue_plus_next = client.queue(blogName)['posts']
    client.delete_post(blogName, queue_plus_next[-1]['id'])

    next_queued_ts = max([int(p.get("scheduled_publish_time", 0)) for p in queue_plus_next])
    next_queued_dt = datetime.fromtimestamp(next_queued_ts)

    print(f"inferred next_queued_dt {next_queued_dt}")
    return next_queued_dt

def determine_mood(response_cache: ResponseCache, window_length_days=4,
                   for_queue=False, verbose=True, return_mood_value=False):
    if not MOOD:
        return "unrestricted"
    try:
        if for_queue:
            dt = next_queued_post_time()
        else:
            # now
            dt = datetime.now()
        if MOOD_DYN:
            mood, mood_value = compute_dynamic_moodspec_at_time(response_cache,
                                                                time=dt,
                                                                window_length_days=window_length_days,
                                                                verbose=verbose)
            if verbose:
                print(f"mood_value: {mood_value:.3f}")
        else:
            mood = random_mood_at_pst_datetime(dt)
            mood_value = None
    except Exception as e:
        print(f"encountered {e} trying to determine my mood, using default {DEFAULT_MOOD}")
        mood = DEFAULT_MOOD
        mood_value = None
    if return_mood_value:
        return mood, mood_value
    return mood

def answer_from_gpt2_service(data: dict):
    new_id = bridge_service_unique_id(bridge_service_url, data)

    data_to_send = dict()
    data_to_send.update(data)
    data_to_send["id"] = new_id

    r = requests.post(url, data=data_to_send)
    result = wait_for_result(new_id)
    return result

def text_post_from_gpt2_service(mood=None):
    data = {"mood": mood}
    url = bridge_service_url + "/textpost"
    new_id = bridge_service_unique_id(url, data)

    data_to_send = dict()
    data_to_send.update(data)
    data_to_send["id"] = new_id

    data["id"] = new_id
    r = requests.post(url, data=data_to_send)
    result = wait_for_result(new_id)
    return result

def strip_spurious_blognames_from_tags(client, tags, auto_accept_list=set()):
    def okay_to_keep(tag):
        if tag in auto_accept_list:
            return True
        result = client.blog_info(tag)
        if "errors" not in result:
            return False
        return True

    return [tag for tag in tags if okay_to_keep(tag)]

def strip_avoid_listed_blognames_from_tags(client, tags):
    return [tag for tag in tags
            if not any([substring in tag for substring in USER_AVOID_LIST]) and
            "myc" not in tag
            ]

def autopublish_screener(asking_name: str, question: str, answer: str, tags: list, screen_robnost=True):
    review_string = (asking_name + " " + question + " " + answer + " " + " ".join(tags)).lower()

    for short_word in bad_strings_shortwords:
        for w, p in product(whitespace, punctuation):
            bad_strings.add(w + short_word + p)
        for w1, w2 in product(punctuation, punctuation):
            bad_strings.add(w1 + short_word + w2)

    bad_strings = bad_strings.union(USER_AVOID_LIST)

    if any([s in review_string for s in bad_strings]):
        strings_found = [s for s in bad_strings if s in review_string]

        for sf in strings_found:
            start_ix = max(0, review_string.index(sf) - 25)
            end_ix = review_string.index(sf) + len(sf) + 25
            sf_formatted = review_string[start_ix:end_ix]

            if start_ix > 0:
                sf_formatted = "... " + sf_formatted
            if end_ix < len(review_string):
                sf_formatted = sf_formatted +  "... "

            print(f"\t{sf}: |{repr(sf_formatted)}|")


        return False
    if asking_name == "nostalgebraist" and screen_robnost:
        print("screened because robnost")
        return False
    return True

def make_text_post(client, blogname, post, tags=[], to_queue=True, to_drafts=False):
    if to_queue:
        if QUEUE_SAFETY:
            if autopublish_screener("", "", post, tags):
                state = "queue"
            else:
                # only used when QUEUE_SAFETY=True
                state = "draft"
                tags.append("TO QUEUE PLEASE THANK YOU ROB!!")
        else:
            state = "queue"
    else:
        state = "draft" if (to_drafts or (not autopublish_screener("", "", post, tags))) else "published"

    post = format_post_for_api(post)

    tags = [t.partition(eot_end_segment)[0] for t in tags]
    tags = [t.partition('<|')[0] for t in tags]  # temporarily support old EOT format

    tags = strip_avoid_listed_blognames_from_tags(client, tags)
    kwargs = {"state": state, "body": post}
    if len(tags) > 0:
        kwargs["tags"] = tags

    client.create_text(blogname, **kwargs)

def answer_ask(client, blogname, ask_id, asking_name, question, answer, tags=[], to_drafts=False, is_reblog=False, reblog_key=None):
    if is_reblog:
        url = "/v2/blog/{}/post/reblog".format(blogname)
        valid_options = ['id', 'reblog_key', 'comment'] + client._post_valid_options()
    else:
        url = "/v2/blog/{}/post/edit".format(blogname)
        valid_options = ['id'] + client._post_valid_options('answer')
        valid_options += ['answer']

    if asking_name not in tags:
        tags.append(asking_name)
    if asking_name != "Anonymous" and "Anonymous" in tags:
        tags.pop(tags.index("Anonymous"))

    tags = [t.partition(eot_end_segment)[0] for t in tags]
    tags = [t.partition('<|')[0] for t in tags]  # temporarily support old EOT format

    tags = [t for t in tags if len(t)>0]
    tags = strip_avoid_listed_blognames_from_tags(client, tags)

    # Take a list of tags and make them acceptable for upload
    tags = ",".join(tags)

    screener_question = "" if is_reblog else question
    print(f"autopublish_screener says: {autopublish_screener(asking_name, screener_question, answer, tags, screen_robnost=True)}")
    state = "draft" if (to_drafts or (not autopublish_screener(asking_name, screener_question, answer, tags, screen_robnost=True))) else "published"

    answer = format_post_for_api(answer)

    if is_reblog:
        data = {"id": ask_id, "reblog_key": reblog_key, "comment": answer, "tags": tags,
         "state": state}
    else:
        data = {"id": ask_id, "answer": answer, "tags": tags,
         "state": state}

    return client.send_api_request('post', url, data, valid_options)


def switch_client_to(client_to_use: pytumblr.TumblrRestClient, ratelimit_client_to_use: RateLimitClient):
    global response_cache
    global ratelimit_client

    response_cache.client = client_to_use
    ratelimit_client = ratelimit_client_to_use


def _compute_checkprob_from_ratelimits(requests_needed_to_check=80, verbose=True):
    all_clients = [
        {"name": "base_client", "tumblr": base_client, "rate": base_ratelimit_client},
        {"name": "dashboard_client", "tumblr": dashboard_client, "rate": dashboard_ratelimit_client}
        ]

    checkprobs = [_compute_checkprob_from_ratelimits_single_client(cl["rate"], requests_needed_to_check, verbose)
                  for cl in all_clients]

    ix_to_use = np.argmax(checkprobs)
    cl_to_use = all_clients[ix_to_use]

    if response_cache.client.request.consumer_key != cl_to_use["tumblr"].request.consumer_key:
        switch_client_to(cl_to_use["tumblr"], cl_to_use["rate"])
        print(f"switched to {cl_to_use['name']}")

    # TODO: think about this... not sure the math makes sense
    combined_checkprob = sum(checkprobs)
    return combined_checkprob

def _compute_checkprob_from_ratelimits_single_client(ratelimit_client_to_check: RateLimitClient,
                                                     requests_needed_to_check=80,
                                                     verbose=True):
    checkprob = None
    while checkprob is None:
        try:
            ratelimit_data = ratelimit_client_to_check.get_ratelimit_data()
            if verbose:
                print(ratelimit_data)
            effective_max_rate = ratelimit_data["effective_max_rate"]
            day_remaining = ratelimit_data["day"]["remaining"]
            hour_remaining = ratelimit_data["hour"]["remaining"]

            # if we checked *every* cycle, we could support up this many requests per check
            requests_per_cycle = SLEEP_TIME * effective_max_rate

            # we'll check only a fraction `checkprob` of the cycles, to support up to `requests_needed_to_check`
            checkprob = requests_per_cycle / requests_needed_to_check

            # don't check if we're close to the edge
            day_edge = 3.5 * requests_needed_to_check
            hour_edge = 1. * requests_needed_to_check
            if (day_remaining < day_edge) or (hour_remaining < hour_edge):
                print(f"close to edge: (day_remaining {day_remaining} < day_edge {day_edge}) or (hour_remaining {hour_remaining} < hour_edge {hour_edge})\n")
                print(f"ratelimit_data:\n{ratelimit_data}\n")
                checkprob = 0.

            # don't check if base client has no requests left
            if base_ratelimit_client.get_ratelimit_data()["effective_remaining"] == 0:
                checkprob = 0.
        except Exception as e:
            print(f"encountered {e} during _compute_checkprob_from_ratelimits, waiting...")
            time.sleep(SLEEP_TIME*5)

    return checkprob


class LoopPersistentData:
    """data class to hold a bunch of stuff... the line between what goes in this vs. what goes in globals in this file is unclear"""
    def __init__(self,
                 reblogs_from_me=set(),
                 reblog_worthy_dash_posts=set(),
                 reply_metadata={},
                 timestamps={},
                 reblog_keys={},
                 last_seen_ts=0,
                 n_posts_to_check_base=150,
                 n_posts_to_check_dash=210,
                 offset_=0,
                 requests_per_check_history=[],
                 apriori_requests_per_check=20,
                 sentiment_cache: SentimentCache=SentimentCache(),
                 follower_names=set(),
                 ):
        self.reblogs_from_me = reblogs_from_me
        self.reblog_worthy_dash_posts = reblog_worthy_dash_posts
        self.reply_metadata = reply_metadata
        self.timestamps = timestamps
        self.reblog_keys = reblog_keys
        self.last_seen_ts = last_seen_ts
        self.n_posts_to_check_base = n_posts_to_check_base
        self.n_posts_to_check_dash = n_posts_to_check_dash
        self.offset_ = offset_
        self.requests_per_check_history = requests_per_check_history
        self.apriori_requests_per_check = apriori_requests_per_check
        self.sentiment_cache = sentiment_cache
        self.follower_names = follower_names

        if len(self.requests_per_check_history) == 0:
            self.requests_per_check_history.extend([self.apriori_requests_per_check,
                                                    self.apriori_requests_per_check]
                                                   )

def update_follower_names(loop_persistent_data, response_cache):
    offset = 0
    response = response_cache.client.blog_following(dash_blogName, offset=offset)
    total_blogs = response.get("total_blogs")

    if total_blogs != len(loop_persistent_data.follower_names):
        print(f"grabbing followers: total_blogs {response.get('total_blogs')}, we have {len(loop_persistent_data.follower_names)}")

        names = {entry['name'] for entry in response['blogs']}
        while len(names) < total_blogs:
            offset = len(names)
            response = response_cache.client.blog_following(dash_blogName, offset=offset)
            names.update({entry['name'] for entry in response['blogs']})
            if len(names) == offset:
                # i "love" tumblr
                print(f"could only get {len(names)} followers although tumblr said there were {total_blogs}")
                break
        loop_persistent_data.follower_names = names
    return loop_persistent_data


def respond_to_reblogs_replies(identifiers, reply_set, loop_persistent_data, response_cache,
                               proba_threshold=None, is_user_input=True):
    for reblog_identifier in identifiers:
        print(f"\n\t--> begin handling {reblog_identifier}\n")
        is_reply = reblog_identifier in reply_set
        if is_reply:
            pass
        else:
            item_post = response_cache.query(CachedResponseType.POSTS, reblog_identifier)

        def _find_bootstrap_draft(drafts):

            bootstrap_drafts = [d for d in drafts if
                                REBLOG_BOOTSTRAP_TEXT in get_body(d)]
            if len(bootstrap_drafts) > 0:
                return bootstrap_drafts[0]
            return None

        api_response = answer_ask(client, blogName, ask_id=reblog_identifier.id_, asking_name=blogName if is_reply else reblog_identifier.blog_name, question="", answer=REBLOG_BOOTSTRAP_TEXT, is_reblog=True, to_drafts=True, reblog_key=loop_persistent_data.reblog_keys[reblog_identifier])

        if api_response.get('meta', {}).get('status') == 403:
            continue

        d_boot = None
        n_boot_tries = 0
        while d_boot is None:
            time.sleep(2**n_boot_tries-1)
            drafts = client.drafts(blogName)["posts"]
            d_boot = _find_bootstrap_draft(drafts)
            n_boot_tries += 1

        processed = process_post_from_post_payload(d_boot)
        question = processed[:processed.index(REBLOG_BOOTSTRAP_TEXT)]

        if is_reply:
            question = bootstrap_draft_inject_reply(question,
                                                    reply_blog_name=reblog_identifier.blog_name,
                                                    reply_body=loop_persistent_data.reply_metadata[reblog_identifier]['reply_note']['reply_text'])
        print(f"\n\t--> using question:\n---------\n{question}\n---------\n")

        # TODO: DRY (this repeats code from answer() in bridge_service.py)
        if "question" in d_boot and "asking_name" in d_boot:
            ask_prefix = UNAME_CHAR + d_boot["asking_name"] + Q_CHAR + "\n" + d_boot["question"] + "\n"
            question = ask_prefix + question
            print(f"ask_prefix: {ask_prefix}")
        else:
            print(f"didn't find ask keys; have keys {d_boot.keys()}")
        gpt2_output = answer_from_gpt2_service(data={'question': question, 'asking_name': reblog_identifier.blog_name, 'exact_prompt': True, 'mood': determine_mood(response_cache)})

        if is_user_input:
            input_type = UserInputType.REPLY if is_reply else UserInputType.REBLOG
            timestamp = reblog_identifier.timestamp if is_reply else loop_persistent_data.timestamps[reblog_identifier]
            user_input_identifier = UserInputIdentifier(input_type=input_type,
                                                        blog_name=reblog_identifier.blog_name,
                                                        id_=reblog_identifier.id_,
                                                        timestamp=timestamp)
            if response_cache.get_cached_user_input_sentiment(user_input_identifier) is not None:
                sent = response_cache.get_cached_user_input_sentiment(user_input_identifier)
                sent["generated_pos_sent"] = gpt2_output.get("all_pos_sentiment")
                sent["generated_ts"] = datetime.now()
                response_cache.mark_user_input_sentiment(user_input_identifier, sent)

        okay_to_reply = True

        if proba_threshold is not None:
            if "proba" not in gpt2_output:
                print(f"!!skipping proba threshold for {reblog_identifier} because key 'proba' not in gpt2_output:\n\t{gpt2_output}\n")
            else:
                proba = gpt2_output["proba"]
                if proba < proba_threshold and int(reblog_identifier.id_) not in DEF_REBLOG_IDS:
                    print(f"not reblogging {reblog_identifier}:\n\tour proba {proba:.1%} < threshold {proba_threshold:.1%}")
                    okay_to_reply = False
                else:
                    print(f"reblogging {reblog_identifier}:\n\tour proba {proba:.1%} >= threshold {proba_threshold:.1%}")

        if is_reply and okay_to_reply:
            mocked_up = mockup_xkit_reply(
                post_url=loop_persistent_data.reply_metadata[reblog_identifier]['post']['post_url'],
                post_summary=loop_persistent_data.reply_metadata[reblog_identifier]['post']['summary'],
                reply_blog_name=loop_persistent_data.reply_metadata[reblog_identifier]['reply_note']['blog_name'],
                reply_blog_url=loop_persistent_data.reply_metadata[reblog_identifier]['reply_note']['blog_url'],
                reply_body=loop_persistent_data.reply_metadata[reblog_identifier]['reply_note']['reply_text']
                )
            make_text_post(client, blogName, mocked_up + "\n" + gpt2_output["post"], tags=gpt2_output["tags"], to_queue=False)
        elif okay_to_reply:
            answer_ask(client, blogName, ask_id=reblog_identifier.id_, asking_name=reblog_identifier.blog_name, question=question, answer=gpt2_output["post"], tags=gpt2_output["tags"], is_reblog=True, reblog_key=loop_persistent_data.reblog_keys[reblog_identifier])

        time.sleep(0.5)
        client.delete_post(blogName, d_boot['id'])
        if is_reply:
            response_cache.mark_reply_handled(reblog_identifier)
        else:
            response_cache.mark_handled(reblog_identifier)
    return loop_persistent_data, response_cache


def is_reblog_worthy_when_responding(post_payload, note_payload):
    comment_ = post_payload['reblog'].get("comment", "")
    has_comment = len(comment_) > 0

    if INDIRECT_REBLOGS:
        is_reblog_worthy = has_comment
    else:
        is_reblog_worthy = (note_payload['reblog_parent_blog_name'] == blogName) and has_comment

    return is_reblog_worthy


def get_selection_prob(texts):
    data = {"texts": texts}
    url = bridge_service_url + "/raw_select"
    new_id = bridge_service_unique_id(url, data)

    data_to_send = dict()
    data_to_send.update(data)
    data_to_send["id"] = new_id

    r = requests.post(url, data=data_to_send)
    result = wait_for_result(new_id, wait_first_time=2, wait_recheck_time=1)
    return result


def is_reblog_worthy_on_dash(post_payload, response_cache, loop_persistent_data, mood_value, follower_multipliers):
    if post_payload.get("id") in DEF_REBLOG_IDS:
        return True

    comment_ = post_payload.get('reblog', {}).get("comment", "")
    has_comment = len(comment_) > 0

    if not has_comment:
        return False

    if post_payload.get("note_count") >= 500:
        if verbose:
            print("\trejecting: notes >= 500")
        return False

    if post_payload.get("type") in {"video"}:
        if verbose:
            print("\trejecting: is video")
        return False

    # tag avoid list
    tags = post_payload.get("tags", [])
    if any([substring in t
            for t in tags
            for substring in DASH_TAG_AVOID_LIST]
           ):
        if verbose:
            print("\trejecting: tag avoid list")
        return False

    # user avoid list
    if post_payload.get('source_title', '') in USER_AVOID_LIST:
        if verbose:
            print("\trejecting: OP user avoid list")
        return False

    for trail_entry in post_payload.get('trail', []):
        if trail_entry.get('blog', {}).get('name', '') in USER_AVOID_LIST:
            if verbose:
                print("\trejecting: trail user avoid list")
            return False
        if int(trail_entry.get('post', {}).get('id', -1)) in NO_REBLOG_IDS:
            if verbose:
                print("\trejecting: reblog id avoid list")
            return False

    post_identifier = PostIdentifier(post_payload['blog_name'], str(post_payload['id']))

    # must follow OP
    post_OP = None
    if 'source_title' in post_payload:
        post_OP = post_payload['source_title']
    else:
        try:
            post_OP = post_payload['trail'][0]['blog']['name']
        except (KeyError, IndexError, TypeError):
            pass
    if post_OP is None:
        print(f"not reblogging {post_identifier} from dash:\n\tcouldn't find post OP in payload\n{post_payload}")
        return False
    elif post_OP not in loop_persistent_data.follower_names:
        print(f"not reblogging {post_identifier} from dash:\n\ti don't follow OP {post_OP}")
        return False

    if f'@{blogName}' in comment_:
        print(f"reblogging {post_identifier} from dash:\n\ti'm tagged in commment {comment_}")
        return True

    sentiment, pos_sentiment, neg_sentiment, prob = None, None, None, None

    text = response_cache.get_cached_post_body(post_identifier)

    if text is None:
        # need body processing
        processed = process_post_from_post_payload(post_payload)
        if processed is None:
            return False

        if ORIG_POST_CHAR in processed:
            text = processed[processed.index(ORIG_POST_CHAR)+1:]
        elif A_CHAR in processed:
            text = processed[[ix for ix, c in enumerate(processed) if c == A_CHAR][-1]+1:]
        else:
            if verbose:
                print("\trejecting: parse fail")
            return False
        if T_CHAR in text:
            text = text.partition(T_CHAR)[0]
        if text.endswith("<|endoftext|>"):
            text = text[:-len("<|endoftext|>")].rstrip("\n")

        tags = post_payload.get("tags", [])
        text += T_CHAR + " ".join(["#" + t for t in tags])
        response_cache.mark_post_body(post_identifier, text)

    if len(text) < 10:
        if verbose:
            print("\trejecting: length<10")
        return False

    if post_identifier in response_cache.text_selector_probs:
        prob = response_cache.text_selector_probs[post_identifier]

    sentiment = loop_persistent_data.sentiment_cache.query(text.partition(T_CHAR)[0])

    # scoring
    if prob is None:
        prob = get_selection_probs([text])[0]
        response_cache.mark_text_selector_prob(post_identifier, prob)

    if sentiment is None:
        reblog_worthy_neg_sentiment = True  # don't depend on allen too heavily
        print(f"warning: couldn't get sentiment")
    else:
        pos_sentiment = sentiment["prob"] if sentiment["label"] == "1" else 1.-sentiment["prob"]
        neg_sentiment = 1. - pos_sentiment
        reblog_worthy_neg_sentiment = neg_sentiment < DASH_REBLOG_MAX_NEG_SENTIMENT

    if pos_sentiment is not None and mood_value is not None:
        mood_buff = DASH_REBLOG_MOOD_BUFF_SCALE * (mood_value - 0.5) * (pos_sentiment - 0.5)
    else:
        mood_buff = 0.

    random_buff = 2*DASH_REBLOG_RANDOM_BUFF_SCALE*np.random.random() - DASH_REBLOG_RANDOM_BUFF_SCALE

    buffed_prob = prob + mood_buff + random_buff

    follower_mult = 1
    if follower_multipliers is not None:
        if post_payload['blog_name'] in follower_multipliers:
            follower_mult = follower_multipliers.loc[post_payload['blog_name']]
        else:
            print(f"couldn't find {post_payload['blog_name']} in follower_multipliers of len {len(follower_multipliers)}")
    buffed_prob = buffed_prob * follower_mult

    reblog_worthy_prob = (buffed_prob > DASH_REBLOG_SELECTION_CUTOFF)

    reblog_worthy = reblog_worthy_prob and reblog_worthy_neg_sentiment

    print(f"got prob {prob:.1%} for post {post_payload.get('id')} from {post_payload.get('blog_name')}")
    print(f"follower_mult {follower_mult:.2f} * (prob {prob:.1%} + mood_buff {mood_buff:.1%} + random_buff {random_buff:.1%}) = buffed_prob {buffed_prob:.1%}")

    if neg_sentiment is not None:
        print(f"got neg_sentiment {neg_sentiment:.1%} for post {post_payload.get('id')} from {post_payload.get('blog_name')}")

    if reblog_worthy and not response_cache.is_handled(post_identifier):
        # explain choice
        explanation = f"reblogging {post_identifier} from dash: "
        explanation += f"\n\tbuffed_prob: {buffed_prob:.0%} vs. {DASH_REBLOG_SELECTION_CUTOFF:.1%}"
        if neg_sentiment is not None:
            explanation += f"\n\tneg_sentiment: {neg_sentiment:.0%} vs. {DASH_REBLOG_MAX_NEG_SENTIMENT:.0%}"
        print(explanation)
    return reblog_worthy


def review_reblogs_from_me(note_payloads,
                           loop_persistent_data,
                           response_cache):
    for r in note_payloads[::-1]:
        reblog_identifier = PostIdentifier(r['blog_name'], r['post_id'])

        r_to_me = r.get("blog_name") == blogName

        post2 = response_cache.query(CachedResponseType.POSTS, reblog_identifier)
        if post2 is None:
            # non-200 response
            print(f"skipping {reblog_identifier}: non-200 response")
            continue
        loop_persistent_data.reblog_keys[reblog_identifier] = post2['reblog_key']

        is_reblog_worthy = False

        if r_to_me:
            # reblog to me: need to mark ancestor as handled
            ancestors =  [trail_item for trail_item in post2['trail']
                           if trail_item["blog"]["name"] != blogName]
            if len(ancestors) > 0:
                direct_ancestor = ancestors[-1]
                response_cache.mark_handled(PostIdentifier(direct_ancestor['blog']['name'], direct_ancestor['post']['id']))
        else:
            is_reblog_worthy = is_reblog_worthy_when_responding(post_payload=post2,
                                                                note_payload=r)

        if is_reblog_worthy:
            loop_persistent_data.reblogs_from_me.add(reblog_identifier)
            loop_persistent_data.timestamps[reblog_identifier] = r['timestamp']

            user_input_identifier = UserInputIdentifier(input_type=UserInputType.REBLOG,
                                                        blog_name=r["blog_name"],
                                                        id_=r["post_id"],
                                                        timestamp=r["timestamp"])
            do_get_sentiment = False
            if response_cache.get_cached_user_input_sentiment(user_input_identifier) is None:
                do_get_sentiment = True
            elif 'text_for_sentiment' not in response_cache.get_cached_user_input_sentiment(user_input_identifier):
                print(f"re-doing sentiment for {user_input_identifier} since 'text_for_sentiment' not found")
                do_get_sentiment = True
            if do_get_sentiment:

                text_for_sentiment = r.get("added_text")
                if text_for_sentiment is None or loop_persistent_data.sentiment_cache.query(text_for_sentiment) is None:
                    if VERBOSE_LOGS:
                        print(f"couldn't find text for sentiment (added_text) in {user_input_identifier}")
                        print(f"have note payload {r}")
                else:
                    sent = loop_persistent_data.sentiment_cache.query(text_for_sentiment)
                    sent['text_for_sentiment'] = text_for_sentiment
                    response_cache.mark_user_input_sentiment(user_input_identifier, sent)
                    print(f"for {reblog_identifier}, recorded {sent} for\n\t{text_for_sentiment}")
    return loop_persistent_data, response_cache


def review_dashboard_post(post_payload,
                          loop_persistent_data,
                          response_cache,
                          mood_value,
                          follower_multipliers=None):
    post_identifier = PostIdentifier(post_payload['blog_name'], str(post_payload['id']))
    loop_persistent_data.reblog_keys[post_identifier] = post_payload['reblog_key']

    is_reblog_worthy = is_reblog_worthy_on_dash(post_payload=post_payload,
                                                response_cache=response_cache,
                                                loop_persistent_data=loop_persistent_data,
                                                mood_value=mood_value,
                                                follower_multipliers=follower_multipliers)

    if is_reblog_worthy:
        loop_persistent_data.reblog_worthy_dash_posts.add(post_identifier)
        loop_persistent_data.timestamps[post_identifier] = post_payload['timestamp']

    return loop_persistent_data, response_cache


def get_relevant_replies_from_notes(post_payload,
                                    notes_payload,
                                    replies_to_handle,
                                    loop_persistent_data,
                                    response_cache):
    if post_payload['id'] in NO_REBLOG_IDS:
        return replies_to_handle, loop_persistent_data, response_cache
    for ix, n in enumerate(notes_payload):
        if n['type'] != "reply":
            continue

        # find the post the reply was "replying to" :(
        reply_context_post_notes = [n2 for n2 in notes_payload[ix:] if n2['blog_name'] == blogName]
        if len(reply_context_post_notes) == 0:
            if VERBOSE_LOGS:
                print(f"couldn't find context self-post for reply {n}: no self-posts in notes")
            continue

        if reply_context_post_notes[0]["type"] == "reblog":
            reply_context_post_id = reply_context_post_notes[0]["post_id"]
        else:
            # need to find id for the origin post
            if len(post_payload['trail']) == 0:
                # assume `post` is the origin post
                reply_context_post_id = post_payload['id']
            else:
                trail_entries_from_me = [entry for entry in post_payload['trail']
                                         if entry['blog']['name'] == blogName]
                reply_context_post_id = int(trail_entries_from_me[0]['post']['id'])

        if reply_context_post_id in NO_REBLOG_IDS:
            return replies_to_handle, loop_persistent_data, response_cache

        reply_context_post_identifier = PostIdentifier(blogName, reply_context_post_id)
        reply_context_post = response_cache.query(CachedResponseType.POSTS, reply_context_post_identifier)

        reply_identifier = ReplyIdentifier(n['blog_name'], reply_context_post_id, n['timestamp'])

        # check whether we follow the user
        am_following_user = (n['blog_name'] in loop_persistent_data.follower_names)

        # is user a "frequent replier"
        is_frequent_replier = (n['blog_name'] in REPLY_USER_AUTO_ACCEPT_LIST)

        # check users tagged
        has_tag = n.get("reply_text", "").lstrip(" ").startswith("@")
        am_tagged = n.get("reply_text", "").lstrip(" ").startswith(f"@{blogName}")
        other_tagged = has_tag and (not am_tagged)

        # check "safe" conditions: (i am OP) AND (no reblogs before this note)
        am_OP = post_payload.get('source_title', blogName) == blogName
        no_reblogs_before_reply = not any([n2.get('type', '') == 'reblog' for n2 in notes_payload[ix:]])
        is_safe = (am_OP) and (no_reblogs_before_reply)

        # always okay if they tagged us, never okay if they tagged someone else, otherwise check if following
        okay_to_reply = (am_following_user or is_frequent_replier or is_safe or am_tagged) and not (other_tagged)
        print(f"okay_to_reply={okay_to_reply} (am_following_user={am_following_user}, is_frequent_replier={is_frequent_replier}, is_safe={is_safe}, am_tagged={am_tagged}, other_tagged={other_tagged}) for\n\t{n}\n")
        if not okay_to_reply:
            continue

        user_input_identifier = UserInputIdentifier(input_type=UserInputType.REPLY,
                                                    blog_name=n["blog_name"],
                                                    id_=post_payload["id"],
                                                    timestamp=n["timestamp"])
        do_get_sentiment = False
        if response_cache.get_cached_user_input_sentiment(user_input_identifier) is None:
            do_get_sentiment = True
        elif 'text_for_sentiment' not in response_cache.get_cached_user_input_sentiment(user_input_identifier):
            print(f"re-doing sentiment for {user_input_identifier} since 'text_for_sentiment' not found")
            do_get_sentiment = True
        if do_get_sentiment:

            text_for_sentiment = n.get("reply_text")
            if text_for_sentiment is None or loop_persistent_data.sentiment_cache.query(text_for_sentiment) is None:
                print(f"couldn't find text for sentiment (reply_text) in {reply_identifier}")
                print(f"have note payload {n}")
            else:
                sent = loop_persistent_data.sentiment_cache.query(text_for_sentiment)
                sent['text_for_sentiment'] = text_for_sentiment
                response_cache.mark_user_input_sentiment(user_input_identifier, sent)
                print(f"for {reply_identifier}, recorded {sent} for\n\t{text_for_sentiment}")

        if not response_cache.is_reply_handled(reply_identifier):
            print(f"for {n}, causal notes are")
            for n2 in notes_payload[ix:]:
                if n2['blog_name'] == blogName:
                    print(f"\t{n2}")

            print(f"\nusing {[n2 for n2 in notes_payload[ix:] if n2['blog_name'] == blogName][0]}\n\t{reply_context_post_id}\n")

            replies_to_handle.add(reply_identifier)

            # we need reblog_key for our own post here
            loop_persistent_data.reblog_keys[reply_identifier] = reply_context_post['reblog_key']
            loop_persistent_data.reply_metadata[reply_identifier] = {}
            loop_persistent_data.reply_metadata[reply_identifier]['reply_note'] = n
            loop_persistent_data.reply_metadata[reply_identifier]['post'] = reply_context_post
    return replies_to_handle, loop_persistent_data, response_cache

def find_reblogs_from_dash(response_cache: ResponseCache):
    idents_from_dash = set()

    for ident in response_cache.reblogs_handled:
        if ident not in response_cache.cache[CachedResponseType.POSTS]:
            print("cache miss")
            time.sleep(0.25)

        post_payload = response_cache.query(CachedResponseType.POSTS, ident)

        if post_payload is None:
            print(f"failed at {ident}")
            continue

        trail = post_payload['trail']

        if not any([entry['blog']['name'] == blogName
                    for entry in trail]):
            idents_from_dash.add(ident)

    return idents_from_dash

def count_reblogs_from_dash(response_cache: ResponseCache,
                            loop_persistent_data: LoopPersistentData):
    idents_from_dash = find_reblogs_from_dash(response_cache)
    vc = pd.Series([ident.blog_name for ident in idents_from_dash]).value_counts()

    loop_persistent_data = update_follower_names(loop_persistent_data, response_cache)

    for name in loop_persistent_data.follower_names:
        if name not in vc.index:
            vc[name] = 0

    return vc

def get_follower_multipliers(response_cache: ResponseCache,
                             loop_persistent_data: LoopPersistentData,
                             scaling_fn=lambda x: np.power(x, 1/3)):
    """currently unused, this wasn't a good idea"""
    vc = count_reblogs_from_dash(response_cache, loop_persistent_data)

    vc_filled = vc.clip(lower=1)

    mults = vc_filled.mean() / vc_filled
    mults = mults.apply(scaling_fn)
    return mults

def do_reblog_reply_handling(loop_persistent_data: LoopPersistentData,
                             response_cache: ResponseCache,
                             n_posts_to_check: int,
                             is_dashboard: bool=False,
                             pseudo_dashboard: bool=False,
                             mood_value: float=None):
    count_check_requests_start = ratelimit_client.get_ratelimit_data()['day']['remaining']

    if is_dashboard and not pseudo_dashboard:
        post_getter = lambda **kwargs: response_cache.record_response_to_cache(dashboard_client.dashboard(**kwargs), care_about_notes=False)["posts"]
        start_ts = DASH_START_TS
    elif is_dashboard:
        def _get_pseudo_dashboard(**kwargs):
            psd_limit = 1
            offsets = {blog_name: 0 for blog_name in loop_persistent_data.follower_names}

            post_payloads = []
            while len(post_payloads) < n_posts_to_check:
                shuffled_names = np.random.choice(list(loop_persistent_data.follower_names),
                                                  len(loop_persistent_data.follower_names),
                                                  replace=False)
                for blog_name in shuffled_names:
                    print(f"checking {blog_name}")
                    post_payloads.extend(
                        response_cache.client.posts(blog_name, limit=psd_limit, offset=offsets[blog_name])["posts"]
                    )
                    offsets[blog_name] += psd_limit
                    print(f"checked {blog_name} -> offset {offsets[blog_name]}, {len(post_payloads)} on pseudo-dash")
                    time.sleep(0.2)
            return post_payloads

        post_getter = _get_pseudo_dashboard
        start_ts = DASH_START_TS

        loop_persistent_data = update_follower_names(loop_persistent_data, response_cache)
    else:
        post_getter = lambda **kwargs: response_cache.record_response_to_cache(response_cache.client.posts(blogName, **kwargs))["posts"]
        start_ts = REBLOG_START_TS

    if is_dashboard and FOLLOWER_MULTIPLIERS:
        follower_multipliers = get_follower_multipliers(response_cache, loop_persistent_data)
    else:
        follower_multipliers = None

    # we need follower names for reply relevance v2
    loop_persistent_data = update_follower_names(loop_persistent_data, response_cache)

    replies_to_handle = set()

    limit_ = min(50, n_posts_to_check)

    offset_ = loop_persistent_data.offset_

    ### get posts
    print(f"\nchecking {n_posts_to_check} posts from starting offset {offset_}...\n")
    posts = []
    updated_last_seen_ts = loop_persistent_data.last_seen_ts

    next_ = [p for p in post_getter(limit=limit_, offset=offset_, notes_info=(not is_dashboard)) if
                p['timestamp'] > start_ts and
                p['id'] not in NO_REBLOG_IDS]
    posts.extend(next_)
    offset_ += len(next_)
    while len(next_) != 0 and len(posts) < n_posts_to_check:
        print(f"got {len(next_)}, starting with {next_[0]['id']}")
        time.sleep(0.1)
        next_ = [p for p in post_getter(limit=limit_, offset=offset_, notes_info=(not is_dashboard)) if
                    p['timestamp'] > start_ts and
                    p['id'] not in NO_REBLOG_IDS]
        posts.extend(next_)
        offset_ += len(next_)

    print(f"{len(posts)} posts retrieved")

    ### loop through posts
    for post_ix, post in enumerate(posts[:n_posts_to_check]):
        post_identifier = PostIdentifier(post['blog_name'], post['id'])
        display_ident = post_identifier if is_dashboard else post['id']
        print(f"{post_ix}/{n_posts_to_check}: {display_ident}")

        ### get reblogs to deal with

        if is_dashboard:
            loop_persistent_data, response_cache = review_dashboard_post(
                post, loop_persistent_data, response_cache, mood_value, follower_multipliers
            )
        else:
            notes = response_cache.query(CachedResponseType.NOTES, post_identifier, expected_notes=post['note_count'])

            updated_last_seen_ts = max([n['timestamp'] for n in notes] + [updated_last_seen_ts])

            reblogs = [n for n in notes if n["type"] == "reblog" and
                       n['timestamp'] >= loop_persistent_data.last_seen_ts and
                       int(n['post_id']) not in NO_REBLOG_IDS]

            loop_persistent_data, response_cache = review_reblogs_from_me(
                note_payloads=reblogs,
                loop_persistent_data=loop_persistent_data,
                response_cache=response_cache
                )

            ### get replies to deal with
            replies_to_handle, loop_persistent_data, response_cache = get_relevant_replies_from_notes(
                post,
                notes,
                replies_to_handle,
                loop_persistent_data,
                response_cache
                )

    if is_dashboard:
        reblogs_to_handle = [r for r in loop_persistent_data.reblog_worthy_dash_posts if
                             (loop_persistent_data.timestamps[r] > start_ts and
                             not response_cache.is_handled(r))]
    else:
        reblogs_to_handle = [r for r in loop_persistent_data.reblogs_from_me if
                             (loop_persistent_data.timestamps[r] > start_ts and
                             not response_cache.is_handled(r))]

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

    # handle reblogs, replies
    loop_persistent_data, response_cache = respond_to_reblogs_replies(
        identifiers = reblogs_to_handle + list(replies_to_handle),
        reply_set = replies_to_handle,
        loop_persistent_data = loop_persistent_data,
        response_cache = response_cache,
        proba_threshold = DASH_REBLOG_CONTINUATION_CUTOFF if is_dashboard else None,
        is_user_input=(not is_dashboard)
        )

    ### post-check stuff

    count_check_requests_end = ratelimit_client.get_ratelimit_data()['day']['remaining']
    count_check_requests_diff = count_check_requests_start - count_check_requests_end
    print(f"used {count_check_requests_diff} requests in this check")

    if not is_dashboard:
        # record calls for this check
        loop_persistent_data.requests_per_check_history.append(count_check_requests_diff)

        # update last_seen_ts
        print(f"updating last_seen_ts: {loop_persistent_data.last_seen_ts} --> {updated_last_seen_ts} (+{updated_last_seen_ts-loop_persistent_data.last_seen_ts})")
        loop_persistent_data.last_seen_ts = updated_last_seen_ts

    return loop_persistent_data, response_cache


def do_ask_handling(loop_persistent_data, response_cache):
    submissions = client.submission(blogName)["posts"]
    n_asks = len(submissions)
    print(f"processing {n_asks} new asks")
    print()

    for x in submissions[::-1]:
        if x.get('summary', '') == FOLLOW_COMMAND:
            try:
                dashboard_client.follow(x['asking_name'])
            except Exception:
                continue
            client.delete_post(blogName, x['id'])
            print(f"followed {x['asking_name']}")
        elif x.get('summary', '') == UNFOLLOW_COMMAND:
            try:
                dashboard_client.unfollow(x['asking_name'])
            except Exception:
                continue
            client.delete_post(blogName, x['id'])
            print(f"unfollowed {x['asking_name']}")
        elif x.get('summary', '') == MOOD_GRAPH_COMMAND:
            path = create_mood_graph(
                response_cache,
                start_time=datetime.now()-pd.Timedelta(days=MOOD_GRAPH_N_DAYS),
                end_time=datetime.now()
                )
            base_client.create_photo(blogName,
                                     state="published" if x['asking_name'] != "nostalgebraist" else "draft",
                                     data=path,
                                     caption=MOOD_GRAPH_EXPLAINER_STRING.format(days_string=MOOD_GRAPH_DAYS_STRING,
                                                                                asking_name=x['asking_name'],
                                                                                asking_url=x['asking_url'])
                                     )
            client.delete_post(blogName, x['id'])
        else:
            for k, v in x.items():
                print(f"{k}: {v}")
            question = inverse_format_post_for_api(x["question"])

            user_input_identifier = UserInputIdentifier(input_type=UserInputType.ASK,
                                                        blog_name=x["asking_name"],
                                                        id_=x["id"],
                                                        timestamp=x["timestamp"])
            do_get_sentiment = False
            if response_cache.get_cached_user_input_sentiment(user_input_identifier) is None:
                do_get_sentiment = True
            elif 'text_for_sentiment' not in response_cache.get_cached_user_input_sentiment(user_input_identifier):
                print(f"re-doing sentiment for {user_input_identifier} since 'text_for_sentiment' not found")
                do_get_sentiment = True
            if do_get_sentiment:
                text_for_sentiment = question
                if text_for_sentiment is None or loop_persistent_data.sentiment_cache.query(text_for_sentiment) is None:
                    if VERBOSE_LOGS:
                        print(f"couldn't find text for sentiment (question) in {user_input_identifier}")
                        print(f"have submission payload {x}")
                else:
                    sent = loop_persistent_data.sentiment_cache.query(text_for_sentiment)
                    sent['text_for_sentiment'] = text_for_sentiment
                    response_cache.mark_user_input_sentiment(user_input_identifier, sent)
                    print(f"for {user_input_identifier}, recorded {sent} for\n\t{text_for_sentiment}")

            gpt2_output = answer_from_gpt2_service(data={'question': question, 'asking_name': x['asking_name'],
                                                         'mood': determine_mood(response_cache)})

            if response_cache.get_cached_user_input_sentiment(user_input_identifier) is not None:
                sent = response_cache.get_cached_user_input_sentiment(user_input_identifier)
                sent["generated_pos_sent"] = gpt2_output.get("all_pos_sentiment")
                sent["generated_ts"] = datetime.now()
                response_cache.mark_user_input_sentiment(user_input_identifier, sent)

            answer_ask(client, blogName, ask_id=x['id'], asking_name=x['asking_name'], question=question, answer=gpt2_output["post"], tags=gpt2_output["tags"])
    return loop_persistent_data, response_cache, n_asks


def do_queue_handling(response_cache: ResponseCache):
    queue = client.queue(blogName, limit=20)['posts']

    n_posts_in_queue = len(queue)
    print(f"{n_posts_in_queue} posts in queue")

    if n_posts_in_queue < WRITE_POSTS_WHEN_QUEUE_BELOW:
        mood_for_queue_writing = determine_mood(response_cache, for_queue=True)

        for textpost_ix in range(N_TO_WRITE):
            print(f"writing new text post... ({textpost_ix}/{N_TO_WRITE})")
            gpt2_output = text_post_from_gpt2_service(mood=mood_for_queue_writing)
            make_text_post(client, blogName, post=gpt2_output["post"], tags=gpt2_output["tags"])

        n_posts_in_queue = len(client.queue(blogName, limit=40)['posts'])
        print(f"now {n_posts_in_queue} posts in queue")


def mainloop(loop_persistent_data: LoopPersistentData,
             response_cache: ResponseCache):
    ### decide whether we'll do the reblog/reply check

    requests_needed_to_check = np.percentile(loop_persistent_data.requests_per_check_history, 75)
    checkprob = _compute_checkprob_from_ratelimits(requests_needed_to_check)

    print(f"using checkprob: {checkprob:.1%}, with last_seen_ts={loop_persistent_data.last_seen_ts}")

    check_roll = np.random.rand()
    if check_roll >= checkprob:
        print(f"skipping check this time ({check_roll:.2f} >= {checkprob})...")
        n_posts_to_check = 0
    else:
        print(f"checking ({check_roll:.2f} < {checkprob:.2f})...")
        n_posts_to_check = loop_persistent_data.n_posts_to_check_base

    n_posts_to_check_dash = loop_persistent_data.n_posts_to_check_dash

    relevant_ratelimit_data = base_ratelimit_client.get_ratelimit_data()
    if relevant_ratelimit_data["effective_remaining"]>0:
        ### do asks check
        loop_persistent_data, response_cache, n_asks = do_ask_handling(loop_persistent_data, response_cache)
        if n_asks > 0:
            loop_persistent_data.sentiment_cache.save()
            response_cache.save()

    ### do reblog/reply check
    if n_posts_to_check > 0:
        # reblogs, replies
        loop_persistent_data, response_cache = do_reblog_reply_handling(
            loop_persistent_data, response_cache, n_posts_to_check
            )
        response_cache.save()
        loop_persistent_data.sentiment_cache.save()

    if relevant_ratelimit_data["effective_remaining"]>0:
        # dash check
        if dashboard_ratelimit_client.get_ratelimit_data()["effective_remaining"] > 0:
            print("checking dash...")
            _, mood_value = determine_mood(response_cache, return_mood_value=True)
            loop_persistent_data, response_cache = do_reblog_reply_handling(
                loop_persistent_data, response_cache, n_posts_to_check_dash, is_dashboard=True, mood_value=mood_value
            )
        else:
            psd_checkprob = checkprob * (requests_needed_to_check / 80)  # quick estimate
            check_roll = np.random.rand()
            if check_roll >= psd_checkprob:
                print(f"skipping pseudo-dash check this time ({check_roll:.2f} >= {psd_checkprob})...")
            else:
                print("checking pseudo-dash...")
                _, mood_value = determine_mood(response_cache, return_mood_value=True)
                loop_persistent_data, response_cache = do_reblog_reply_handling(
                    loop_persistent_data, response_cache, n_posts_to_check_dash, is_dashboard=True, mood_value=mood_value,
                    pseudo_dashboard=True
                )

        response_cache.save()
        loop_persistent_data.sentiment_cache.save()

        ### do queue check
        do_queue_handling(response_cache)

        # inform us about drafts
        drafts = client.drafts(blogName)["posts"]
        print(f"{len(drafts)} waiting for review")
    else:
        print("skipping asks, queue, drafts until we're no longer rate limited")
        print(relevant_ratelimit_data)

    if MOOD_DYN:
        print("current mood:")
        determine_mood(response_cache)

    print()
    return loop_persistent_data, response_cache

if __name__ == "__main__":
    response_cache = ResponseCache.load(client)
    sentiment_cache = SentimentCache.load()
    loop_persistent_data = LoopPersistentData(sentiment_cache=sentiment_cache)

    while True:
        try:
            loop_persistent_data, response_cache = mainloop(loop_persistent_data, response_cache)
            time.sleep(SLEEP_TIME)
        except (requests.exceptions.ConnectionError, KeyError):
            print("hit an error, waiting for a little while...")
            time.sleep(SLEEP_TIME*5)
