"""
Originally for caching tumblr API responses to help w/ ratelimiting.

Scope creep has caused this to be more of a general cache for lots of stuff, so it now
holds a lot of stuff needed for persistent-over-time elements of bot operation, like
the mood feature.
"""
import subprocess
from collections import namedtuple, defaultdict
from enum import Enum
from datetime import datetime, timedelta

import pytumblr
import time
import os
import pickle

from smart_open import open

from util.times import now_pst, fromtimestamp_pst

import config.bot_config_singleton
bot_specific_constants = config.bot_config_singleton.bot_specific_constants

NO_REBLOG_IDS = bot_specific_constants.NO_REBLOG_IDS
blogName = bot_specific_constants.blogName

PostIdentifier = namedtuple("PostIdentifier", "blog_name id_")
ReplyIdentifier = namedtuple("ReplyIdentifier", "blog_name id_ timestamp")
CachedResponseType = Enum("CachedResponseType", "POSTS NOTES")

UserInputType = Enum("UserInputType", "ASK REBLOG REPLY")
UserInputIdentifier = namedtuple(
    "UserInputIdentifier", "input_type blog_name id_ timestamp"
)


class ResponseCache:
    def __init__(
        self, client: pytumblr.TumblrRestClient, path: str, cache: dict = None
    ):
        self.client = client
        self.path = path
        self.cache = cache

        if self.cache is None:
            self.cache = {rtype: {} for rtype in CachedResponseType}
            self.cache["reblogs_handled"] = set()

        if "reblogs_handled" not in self.cache:
            self.cache["reblogs_handled"] = set()

        if "replies_handled" not in self.cache:
            self.cache["replies_handled"] = set()

        if "user_input_sentiments" not in self.cache:
            self.cache["user_input_sentiments"] = {}

        if "blocked_by_users" not in self.cache:
            self.cache["blocked_by_users"] = set()

        if "last_accessed_time" not in self.cache:
            self.cache["last_accessed_time"] = {}

        if "dash_post_judgments" not in self.cache:
            self.cache["dash_post_judgments"] = {}

        if "last_seen_ts" not in self.cache:
            self.cache["last_seen_ts"] = defaultdict(int)

        if "following_names" not in self.cache:
            self.cache["following_names"] = set()

    @staticmethod
    def load(client=None, path="gs://nost-trc/nbar_data/response_cache.pkl.gz", verbose=True):
        cache = None
        with open(path, "rb") as f:
            cache = pickle.load(f)
        if verbose:
            lengths = {k: len(cache[k]) for k in cache.keys()}
            print(f"loaded response cache with lengths {lengths}")
        loaded = ResponseCache(client, path, cache)
        loaded.remove_oldest()
        return loaded

    def save(self, verbose=True, do_backup=False):
        self.remove_oldest()

        t1 = time.time()

        with open(self.path, "wb") as f:
            pickle.dump(self.cache, f)

        _tsave = time.time()
        print(f"response_cache save 1: {_tsave-t1:.3f}s sec")

        if do_backup:
            # TODO: better path handling
            backup_path = self.path[: -len(".pkl.gz")] + "_backup.pkl.gz"
            subprocess.check_output(f"cp {self.path} {backup_path}", shell=True)

            _tsave2 = time.time()
            print(f"response_cache save 2: {_tsave2-_tsave:.3f}s sec")
        if verbose:
            lengths = {k: len(self.cache[k]) for k in CachedResponseType}
            print(f"saved response cache with lengths {lengths}")

    def decomposed_save(self, directory: str):
        # for tracking sizes of individual parts
        os.makedirs(directory, exist_ok=True)
        for k, v in self.cache.items():
            path = os.path.join(directory, f"{repr(k)}.pkl.gz")
            with open(path, "wb") as f:
                pickle.dump({k: self.cache[k]}, f)
                print(f"wrote {repr(k)} to {path}")

    def remove_oldest(self, max_hours=18, dryrun=False):
        lat = self.cache["last_accessed_time"]
        existing_p = self.cache[CachedResponseType.POSTS]
        existing_n = self.cache[CachedResponseType.NOTES]
        existing_dpj = self.cache["dash_post_judgments"]

        last_allowed_time = now_pst() - timedelta(hours=max_hours)

        allowed_p = {pi for pi, t in lat.items() if t >= last_allowed_time}

        new_p = {pi: existing_p[pi] for pi in existing_p if pi in allowed_p}
        new_n = {pi: existing_n[pi] for pi in existing_n if pi in allowed_p}
        new_lat = {pi: lat[pi] for pi in lat if pi in allowed_p}
        new_dpj = {pi: existing_dpj[pi] for pi in existing_dpj if pi in allowed_p}

        before_len_p = len(existing_p)
        before_len_n = len(existing_n)
        before_len_lat = len(lat)
        before_len_dpj = len(existing_dpj)
        delta_len_p = before_len_p - len(new_p)
        delta_len_n = before_len_n - len(new_n)
        delta_len_lat = before_len_lat - len(new_lat)
        delta_len_dpj = before_len_dpj - len(new_dpj)

        if dryrun:
            print(f"remove_oldest: would drop {delta_len_p} of {before_len_p} POSTS")
            print(f"remove_oldest: would drop {delta_len_n} of {before_len_n} NOTES")
            print(f"remove_oldest: would drop {delta_len_lat} of {before_len_lat} last_accessed_time")
            print(f"remove_oldest: would drop {delta_len_dpj} of {before_len_dpj} dash_post_judgments")
        else:
            print(f"remove_oldest: dropping {delta_len_p} of {before_len_p} POSTS")
            print(f"remove_oldest: dropping {delta_len_n} of {before_len_n} NOTES")
            print(f"remove_oldest: dropping {delta_len_lat} of {before_len_lat} last_accessed_time")
            print(f"remove_oldest: dropping {delta_len_dpj} of {before_len_dpj} dash_post_judgments")
            self.cache[CachedResponseType.POSTS] = new_p
            self.cache[CachedResponseType.NOTES] = new_n
            self.cache["last_accessed_time"] = new_lat
            self.cache["dash_post_judgments"] = new_dpj

    def record_response_to_cache(
        self, response: dict, care_about_notes=True, care_about_likes=False
    ):
        if response.get("response") == "You do not have permission to view this blog":
            # TODO: make this work properly
            user = response.get("blog", {}).get("name", None)
            if user is not None:
                self.mark_blocked_by_user(user)
            return response

        if "posts" not in response:
            print(f"weirdness: {response}")
            return response

        for response_core in response["posts"]:
            identifier = PostIdentifier(response_core["blog_name"], response_core["id"])
            post_payload = {k: v for k, v in response_core.items() if k != "notes"}

            notes = self.normalized_lookup(CachedResponseType.NOTES, identifier)
            if notes is None:
                notes = []
            timestamps = {n["timestamp"] for n in notes}

            payload_notes = response_core.get("notes", [])
            new_notes = [n for n in payload_notes if n["timestamp"] not in timestamps]
            if care_about_notes and len(payload_notes) > 0:
                compare_on_conversational = (
                    len(self.conversational_notes(payload_notes)) > 0
                )
                if compare_on_conversational:
                    latest_obtained_ts = self.latest_stored_conversational_note_ts(
                        identifier
                    )
                else:
                    latest_obtained_ts = self.latest_stored_note_ts(identifier)
                reference_note_ts = self.earliest_conversational_note_ts(payload_notes)
            notes.extend(new_notes)

            if care_about_notes and len(payload_notes) > 0:
                expected_notes = response_core["note_count"] + 1
                cache_up_to_date = (
                    False
                    if len(timestamps) == 0
                    else (reference_note_ts < latest_obtained_ts)
                )
                if not cache_up_to_date and response_core["id"] not in NO_REBLOG_IDS:
                    done_calling_notes_endpt = False
                    # need this to get the links
                    time.sleep(0.33)
                    print(f"\thave {len(notes)} notes of {expected_notes}")
                    note_response = self.client.notes(
                        identifier.blog_name, id=identifier.id_, mode="conversation"
                    )
                    payload_notes = note_response["notes"]
                    new_notes = [
                        n for n in payload_notes if n["timestamp"] not in timestamps
                    ]
                    notes.extend(new_notes)

                    done_calling_notes_endpt = (
                        False
                        if len(timestamps) == 0
                        else (
                            self.earliest_conversational_note_ts(payload_notes)
                            < latest_obtained_ts
                        )
                    )

                    while (not done_calling_notes_endpt) and (
                        "_links" in note_response
                    ):
                        time.sleep(0.33)
                        note_response = self.client.notes(
                            identifier.blog_name,
                            id=identifier.id_,
                            mode="conversation",
                            before_timestamp=note_response["_links"]["next"][
                                "query_params"
                            ]["before_timestamp"],
                        )
                        payload_notes = note_response["notes"]
                        new_notes = [
                            n for n in payload_notes if n["timestamp"] not in timestamps
                        ]
                        notes.extend(new_notes)
                        print(f"\thave {len(notes)} notes of {expected_notes}")
                        done_calling_notes_endpt = (
                            False
                            if len(timestamps) == 0
                            else (
                                self.earliest_conversational_note_ts(payload_notes)
                                < latest_obtained_ts
                            )
                        )
            self.cache[CachedResponseType.NOTES][identifier] = sorted(
                notes, key=lambda n: n["timestamp"], reverse=True
            )

            self.cache[CachedResponseType.POSTS][identifier] = post_payload

        return response

    def _cached_note_count(self, rtype, identifier, use_overrides=True):
        if ((identifier, "note_count_override") in self.cache[rtype]) and use_overrides:
            return self.cache[rtype][(identifier, "note_count_override")]
        return max(0, len(self.cache[rtype][identifier]))

    def _note_cache_uptodate(
        self,
        identifier: PostIdentifier,
        expected_notes: int,
        reference_note_ts: dict,
        compare_on_conversational=True,
    ):
        if expected_notes is None and reference_note_ts is None:
            print(f"matchers not provided, pulling fresh notes for {identifier}")
            return False
        normalized_ident = self.get_normalized_ident(
            CachedResponseType.NOTES, identifier
        )
        if normalized_ident is None:
            print(f"note cache unavailable for {identifier}")
            return False

        if reference_note_ts is not None:
            if compare_on_conversational:
                latest_stored_ts = self.latest_stored_conversational_note_ts(
                    normalized_ident
                )
            else:
                latest_stored_ts = self.latest_stored_note_ts(normalized_ident)
            if latest_stored_ts < reference_note_ts:
                print(f"_note_cache_uptodate: NOT up to date")
                print(
                    f"_note_cache_uptodate: got latest_stored_ts={latest_stored_ts} vs reference_note_ts={reference_note_ts}"
                )
                print(f"_note_cache_uptodate: {latest_stored_ts >= reference_note_ts}")

            return latest_stored_ts >= reference_note_ts

        cached_notes = self._cached_note_count(
            CachedResponseType.NOTES, normalized_ident
        )
        cache_uptodate = expected_notes <= cached_notes
        if not cache_uptodate:
            print(
                f"note cache stale for {normalized_ident}: expected {expected_notes} notes but have {cached_notes} in cache"
            )
        return cache_uptodate

    def _api_call_for_rtype(
        self,
        rtype: CachedResponseType,
        identifier: PostIdentifier,
        care_about_notes=True,
        care_about_likes=False,
        notes_field=None,  # TODO: use this properly
    ):
        time.sleep(0.33)
        response = self.client.posts(
            identifier.blog_name, id=identifier.id_, notes_info=True
        )
        self.record_response_to_cache(
            response,
            care_about_notes=care_about_notes,
            care_about_likes=care_about_likes,
        )

    def _can_use_cached(
        self,
        rtype: CachedResponseType,
        identifier: PostIdentifier,
        expected_notes: int = None,
        notes_field: list = None,
    ):
        is_in_cache = self.get_normalized_ident(rtype, identifier) is not None
        cache_uptodate = True

        if rtype == CachedResponseType.NOTES and is_in_cache:
            reference_note_ts = self.earliest_conversational_note_ts(notes_field)
            if reference_note_ts is None:
                cache_uptodate = True
            else:
                compare_on_conversational = (
                    True
                    if notes_field is None
                    else len(self.conversational_notes(notes_field)) > 0
                )
                cache_uptodate = self._note_cache_uptodate(
                    identifier,
                    expected_notes,
                    reference_note_ts,
                    compare_on_conversational=compare_on_conversational,
                )

        return is_in_cache and cache_uptodate

    def _record_unexpected_note_counts(self, rtype, identifier, expected_notes):
        cached_notes = self._cached_note_count(rtype, identifier, use_overrides=False)
        if cached_notes != expected_notes:
            print(
                f"cache note count {cached_notes} still != expected {expected_notes} for {identifier}, marking {expected_notes} as override"
            )
            self.cache[rtype][(identifier, "note_count_override")] = expected_notes
        if (cached_notes == expected_notes) and (
            (identifier, "note_count_override") in self.cache[rtype]
        ):
            print(
                f"cache note count {cached_notes} = expected {expected_notes} for {identifier}, unsetting override"
            )
            del self.cache[rtype][(identifier, "note_count_override")]

    def get_normalized_ident(self, rtype, identifier):
        identifier_int = PostIdentifier(identifier.blog_name, int(identifier.id_))
        identifier_str = PostIdentifier(identifier.blog_name, str(identifier.id_))
        if identifier_int in self.cache[rtype]:
            self.cache["last_accessed_time"][identifier_int] = now_pst()
            return identifier_int
        if identifier_str in self.cache[rtype]:
            self.cache["last_accessed_time"][identifier_str] = now_pst()
            return identifier_str
        return None

    def normalized_lookup(self, rtype, identifier, expect_in_cache=False):
        normalized_ident = self.get_normalized_ident(rtype, identifier)
        if normalized_ident is None:
            if expect_in_cache:
                print(f"{identifier} should be in {rtype} cache but isn't")
            return None
        return self.cache[rtype][normalized_ident]

    def query(
        self,
        rtype: CachedResponseType,
        identifier: PostIdentifier,
        expected_notes: int = None,
        notes_field: list = None,
        care_about_notes=True,
        care_about_likes=False,
    ):
        if care_about_likes:
            notes_field = None
        if not self._can_use_cached(rtype, identifier, expected_notes, notes_field):
            self._api_call_for_rtype(
                rtype,
                identifier,
                care_about_notes=care_about_notes,
                care_about_likes=care_about_likes,
                notes_field=notes_field,
            )
        return self.normalized_lookup(rtype, identifier, expect_in_cache=True)

    def mark_handled(self, identifier: PostIdentifier):
        identifier_normalized = PostIdentifier(
            blog_name=identifier.blog_name, id_=str(identifier.id_)
        )

        tip = self.cached_trail_tip(identifier_normalized)
        if tip is not None:
            if tip != identifier and tip.blog_name != blogName:
                print(
                    f"mark_handled: for {identifier}, also marking tip {tip} as handled"
                )
                self.cache["reblogs_handled"].add(tip)
        else:
            print(f"mark_handled: for {identifier}, found no tip {tip} to mark")

        self.cache["reblogs_handled"].add(identifier_normalized)

    def mark_unhandled(self, identifier: PostIdentifier):
        tip = self.cached_trail_tip(identifier)
        if tip is not None and tip in self.cache["reblogs_handled"]:
            if tip != identifier:
                print(
                    f"mark_unhandled: for {identifier}, also marking tip {tip} as unhandled"
                )
            self.cache["reblogs_handled"].remove(tip)
        if identifier in self.cache["reblogs_handled"]:
            self.cache["reblogs_handled"].remove(identifier)

    @staticmethod
    def trail_tip(trail: list):
        if trail is None:
            return None
        ordered_trail = sorted(trail, key=lambda x: x.get("post", {}).get("id", "-1"))
        if len(ordered_trail) > 0:
            tip = ordered_trail[-1]
            tip_ident = PostIdentifier(
                tip.get("blog", {}).get("name", ""),
                str(tip.get("post", {}).get("id", "-1")),
            )
            return tip_ident

    def cached_trail_tip(self, identifier: PostIdentifier):
        cached_post = self.normalized_lookup(CachedResponseType.POSTS, identifier)
        if cached_post is not None:
            tip = ResponseCache.trail_tip(cached_post.get("trail"))
            return tip

    def is_handled(self, identifier: PostIdentifier, check_tip=True):
        identifier_normalized = PostIdentifier(
            blog_name=identifier.blog_name, id_=str(identifier.id_)
        )

        handled_at_ident = identifier_normalized in self.cache["reblogs_handled"]
        handled_at_tip = None

        tip = self.cached_trail_tip(identifier_normalized) if check_tip else None
        if tip is not None:
            handled_at_tip = self.is_handled(tip, check_tip=False)

        if handled_at_tip is not None:
            # print(f"identifier: handled_at_ident={handled_at_ident}")
            # print(f"identifier: handled_at_tip={handled_at_tip}")
            handled = handled_at_tip or handled_at_ident
        else:
            handled = handled_at_ident

        return handled

    def mark_reply_handled(self, identifier: ReplyIdentifier):
        identifier_normalized = ReplyIdentifier(
            blog_name=identifier.blog_name,
            id_=str(identifier.id_),
            timestamp=identifier.timestamp,
        )
        self.cache["replies_handled"].add(identifier_normalized)

    def is_reply_handled(self, identifier: ReplyIdentifier):
        identifier_normalized = ReplyIdentifier(
            blog_name=identifier.blog_name,
            id_=str(identifier.id_),
            timestamp=identifier.timestamp,
        )
        return identifier_normalized in self.cache["replies_handled"]

    def mark_user_input_sentiment(
        self, identifier: UserInputIdentifier, sentiment: dict
    ):
        identifier_normalized = UserInputIdentifier(
            input_type=identifier.input_type,
            blog_name=identifier.blog_name,
            id_=str(identifier.id_) if identifier.id_ is not None else None,
            timestamp=identifier.timestamp,
        )
        self.cache["user_input_sentiments"][identifier_normalized] = sentiment

    def get_cached_user_input_sentiment(self, identifier: UserInputIdentifier):
        identifier_normalized = UserInputIdentifier(
            input_type=identifier.input_type,
            blog_name=identifier.blog_name,
            id_=str(identifier.id_) if identifier.id_ is not None else None,
            timestamp=identifier.timestamp,
        )
        return self.cache["user_input_sentiments"].get(identifier_normalized)

    def mark_dash_post_judgments(
        self, identifier: PostIdentifier, judgments: dict
    ):
        identifier_normalized = PostIdentifier(
            blog_name=identifier.blog_name, id_=str(identifier.id_)
        )
        self.cache["dash_post_judgments"][identifier_normalized] = judgments

    def get_cached_dash_post_judgments(
        self, identifier: PostIdentifier
    ):
        return self.normalized_lookup('dash_post_judgments', identifier)

    def mark_blocked_by_user(self, blog_name: str):
        self.cache["blocked_by_users"].add(blog_name)

    @staticmethod
    def conversational_notes(notes_field: list):
        # return [n for n in notes_field if n.get('type') != "like"]
        added_text_fields = ["added_text", "reply_text"]
        return [
            n
            for n in notes_field
            if any([field in n for field in added_text_fields])
            or n.get("type") == "posted"
        ]

    @staticmethod
    def conversational_notes_with_fallback(notes_field: list, direction="earliest"):
        conv_notes = ResponseCache.conversational_notes(notes_field)
        if len(conv_notes) > 0:
            return conv_notes
        # fallback
        if direction == "earliest":
            return sorted(notes_field, key=lambda n: n.get("timestamp", -1))[:1]
        else:
            return sorted(notes_field, key=lambda n: n.get("timestamp", -1))[-1:]

    @staticmethod
    def conversational_notes_ts_with_fallback(
        notes_field: list, direction="earliest", debug=False
    ):
        conv_notes = ResponseCache.conversational_notes_with_fallback(
            notes_field, direction=direction
        )
        notes_ts = [n.get("timestamp") for n in conv_notes if "timestamp" in n]
        if debug:
            print(
                f"\nnotes_ts={notes_ts}\n\tgot notes_field={notes_field}\n\tgot conv_notes={conv_notes}\n"
            )
        return notes_ts

    @staticmethod
    def earliest_conversational_note_ts(notes_field: list):
        if notes_field is None:
            return None
        return min(
            ResponseCache.conversational_notes_ts_with_fallback(
                notes_field, direction="earliest"
            )
        )

    @staticmethod
    def latest_conversational_note_ts(notes_field: list):
        if notes_field is None:
            return None
        return max(
            ResponseCache.conversational_notes_ts_with_fallback(
                notes_field, direction="latest"
            )
        )

    def latest_stored_conversational_note_ts(self, identifier: PostIdentifier):
        notes = self.normalized_lookup(CachedResponseType.NOTES, identifier)
        if notes is not None:
            return self.latest_conversational_note_ts(notes)
        return None

    def latest_stored_note_ts(self, identifier: PostIdentifier):
        notes = self.normalized_lookup(CachedResponseType.NOTES, identifier)
        if notes is not None:
            notes_ts = [n.get("timestamp") for n in notes if "timestamp" in n]
            return -1 if len(notes_ts) == 0 else max(notes_ts)
        return -1

    def get_last_seen_ts(self, key):
        return self.cache['last_seen_ts'][key]

    def update_last_seen_ts(self, key, ts):
        prev = self.get_last_seen_ts(key)
        print(
            f"updating {key}: {prev} --> {ts} (+{ts-prev})"
        )
        self.cache['last_seen_ts'][key] = ts

    def set_following_names(self, following_names):
        self.cache["following_names"] = following_names

    def follow(self, name, dashboard_client):
        self.cache["following_names"].add(name)
        dashboard_client.follow(name)

    def unfollow(self, name, dashboard_client):
        self.cache["following_names"].remove(name)
        dashboard_client.unfollow(name)

    @property
    def reblogs_handled(self):
        return self.cache["reblogs_handled"]

    @property
    def replies_handled(self):
        return self.cache["replies_handled"]

    @property
    def text_selector_probs(self):
        return self.cache["text_selector_probs"]

    @property
    def text_sentiments(self):
        return self.cache["text_sentiments"]

    @property
    def user_input_sentiments(self):
        return self.cache["user_input_sentiments"]

    @property
    def blocked_by_users(self):
        return self.cache["blocked_by_users"]

    @property
    def following_names(self):
        return self.cache["following_names"]
