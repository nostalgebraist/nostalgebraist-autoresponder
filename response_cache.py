"""
Originally for caching tumblr API responses to help w/ ratelimiting.

Scope creep has caused this to be more of a general cache for lots of stuff, so it now
holds a lot of stuff needed for persistent-over-time elements of bot operation, like
the mood feature.
"""
from collections import namedtuple
from enum import Enum

import pytumblr, time, os, pickle

PostIdentifier = namedtuple("PostIdentifier", "blog_name id_")
ReplyIdentifier = namedtuple("ReplyIdentifier", "blog_name id_ timestamp")
CachedResponseType = Enum("CachedResponseType", "POSTS NOTES")

UserInputType = Enum("UserInputType", "ASK REBLOG REPLY")
UserInputIdentifier = namedtuple("UserInputIdentifier", "input_type blog_name id_ timestamp")


class ResponseCache:
    def __init__(self, client: pytumblr.TumblrRestClient, path: str, cache: dict=None):
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

        if "text_selector_probs" not in self.cache:
            self.cache["text_selector_probs"] = {}

        # TODO: deprecate (now in SentimentCache)
        if "text_sentiments" not in self.cache:
            self.cache["text_sentiments"] = {}

        if "post_bodies" not in self.cache:
            self.cache["post_bodies"] = {}

        if "user_input_sentiments" not in self.cache:
            self.cache["user_input_sentiments"] = {}

        if "blocked_by_users" not in self.cache:
            self.cache["blocked_by_users"] = set()

    @staticmethod
    def load(client, path="response_cache.pkl.gz", verbose=True):
        cache = None
        if os.path.exists(path):
            with open(path, "rb") as f:
                cache = pickle.load(f)
            if verbose:
                lengths = {k: len(cache[k]) for k in cache.keys()}
                print(f"loaded response cache with lengths {lengths}")
        else:
            print(f"initialized response cache")
        return ResponseCache(client, path, cache)

    def save(self, verbose=True, do_backup=True):
        with open(self.path, "wb") as f:
            pickle.dump(self.cache, f)
        if do_backup:
            # TODO: better path handling
            with open(self.path[:-len(".pkl.gz")] + "_backup.pkl.gz", "wb") as f:
                pickle.dump(self.cache, f)
        if verbose:
            lengths = {k: len(self.cache[k]) for k in CachedResponseType}
            print(f"saved response cache with lengths {lengths}")

    def record_response_to_cache(self, response: dict, care_about_notes=True):
        if response.get('response') == "You do not have permission to view this blog":
            self.mark_blocked_by_user(reblog_identifier.blog_name)
            return response

        for response_core in response['posts']:
            identifier = PostIdentifier(response_core["blog_name"], response_core["id"])
            post_payload = {k: v for k, v in response_core.items() if k != "notes"}
            notes = response_core.get('notes', [])

            if care_about_notes:
                expected_notes = response_core['note_count']+1
                cache_up_to_date = self._note_cache_uptodate(identifier, expected_notes)
                if not cache_up_to_date:
                    if expected_notes >= 50:
                        # need this to get the links
                        time.sleep(0.33)
                        print(f"\thave {len(notes)} notes of {expected_notes}")
                        note_response = self.client.notes(identifier.blog_name, id=identifier.id_)
                        while (expected_notes > (len(notes))) and ("_links" in note_response):
                            time.sleep(0.33)
                            note_response = self.client.notes(identifier.blog_name, id=identifier.id_, before_timestamp=note_response["_links"]["next"]["query_params"]["before_timestamp"])
                            timestamps = {n['timestamp'] for n in notes}
                            new_notes = [n for n in note_response["notes"] if n['timestamp'] not in timestamps]
                            notes.extend(new_notes)
                            print(f"\thave {len(notes)} notes of {expected_notes}")
                            if len(new_notes) == 0:
                                break
                    self.cache[CachedResponseType.NOTES][identifier] = notes
                    self._record_unexpected_note_counts(CachedResponseType.NOTES, identifier,
                                                        expected_notes=expected_notes
                                                        )

            self.cache[CachedResponseType.POSTS][identifier] = post_payload

        return response

    def _cached_note_count(self, rtype, identifier, use_overrides=True):
        if ((identifier, "note_count_override") in self.cache[rtype]) and use_overrides:
            return self.cache[rtype][(identifier, "note_count_override")]
        return max(0, len(self.cache[rtype][identifier]))

    def _note_cache_uptodate(self, identifier: PostIdentifier, expected_notes: int):
        if expected_notes is None:
            print(f"expected_notes not provided, pulling fresh notes for {identifier}")
            return False
        normalized_ident = self.get_normalized_ident(CachedResponseType.NOTES, identifier)
        if normalized_ident is None:
            print(f"note cache unavailable for {identifier}")
            return False
        cached_notes = self._cached_note_count(CachedResponseType.NOTES, normalized_ident)
        cache_uptodate = expected_notes <= cached_notes
        if not cache_uptodate:
            print(f"note cache stale for {normalized_ident}: expected {expected_notes} notes but have {cached_notes} in cache")
        return cache_uptodate

    def _api_call_for_rtype(self, rtype: CachedResponseType, identifier: PostIdentifier):
        time.sleep(0.33)
        response = self.client.posts(identifier.blog_name, id=identifier.id_, notes_info=True)
        self.record_response_to_cache(response)

    def _can_use_cached(self, rtype: CachedResponseType, identifier: PostIdentifier, expected_notes: int=None):
        is_in_cache = self.get_normalized_ident(rtype, identifier) is not None
        cache_uptodate = True

        if rtype == CachedResponseType.NOTES and is_in_cache:
            cache_uptodate = self._note_cache_uptodate(identifier, expected_notes)

        return is_in_cache and cache_uptodate

    def _record_unexpected_note_counts(self, rtype, identifier, expected_notes):
        cached_notes = self._cached_note_count(rtype, identifier, use_overrides=False)
        if cached_notes != expected_notes:
            print(f"cache note count {cached_notes} still != expected {expected_notes} for {identifier}, marking {expected_notes} as override")
            self.cache[rtype][(identifier, "note_count_override")] = expected_notes
        if (cached_notes == expected_notes) and ((identifier, "note_count_override") in self.cache[rtype]):
            print(f"cache note count {cached_notes} = expected {expected_notes} for {identifier}, unsetting override")
            del self.cache[rtype][(identifier, "note_count_override")]

    def get_normalized_ident(self, rtype, identifier):
        identifier_int = PostIdentifier(identifier.blog_name, int(identifier.id_))
        identifier_str = PostIdentifier(identifier.blog_name, str(identifier.id_))
        if identifier_int in self.cache[rtype]:
            return identifier_int
        if identifier_str in self.cache[rtype]:
            return identifier_str
        return None

    def normalized_lookup(self, rtype, identifier):
        normalized_ident = self.get_normalized_ident(rtype, identifier)
        if normalized_ident is None:
            raise ValueError(f"{identifier} should be in {rtype} cache but isn't")
        return self.cache[rtype][normalized_ident]

    def query(self, rtype: CachedResponseType, identifier: PostIdentifier, expected_notes: int=None):
        if not self._can_use_cached(rtype, identifier, expected_notes):
            self._api_call_for_rtype(rtype, identifier)
        return self.normalized_lookup(rtype, identifier)

    def mark_handled(self, identifier: PostIdentifier):
        identifier_normalized = PostIdentifier(blog_name=identifier.blog_name, id_=str(identifier.id_))
        self.cache["reblogs_handled"].add(identifier_normalized)

    def is_handled(self, identifier: PostIdentifier):
        identifier_normalized = PostIdentifier(blog_name=identifier.blog_name, id_=str(identifier.id_))
        return identifier_normalized in self.cache["reblogs_handled"]

    def mark_reply_handled(self, identifier: ReplyIdentifier):
        identifier_normalized = ReplyIdentifier(blog_name=identifier.blog_name,
                                                id_=str(identifier.id_),
                                                timestamp=identifier.timestamp)
        self.cache["replies_handled"].add(identifier_normalized)

    def is_reply_handled(self, identifier: ReplyIdentifier):
        identifier_normalized = ReplyIdentifier(blog_name=identifier.blog_name,
                                                id_=str(identifier.id_),
                                                timestamp=identifier.timestamp)
        return identifier_normalized in self.cache["replies_handled"]

    def mark_text_selector_prob(self, identifier: PostIdentifier, prob: float):
        self.cache["text_selector_probs"][identifier] = prob

    def mark_text_sentiment(self, identifier: PostIdentifier, sentiment: dict):
        self.cache["text_sentiments"][identifier] = sentiment

    def mark_post_body(self, identifier: PostIdentifier, body: str):
        identifier_normalized = PostIdentifier(blog_name=identifier.blog_name, id_=str(identifier.id_))
        self.cache["post_bodies"][identifier_normalized] = body

    def get_cached_post_body(self, identifier: PostIdentifier):
        identifier_normalized = PostIdentifier(blog_name=identifier.blog_name, id_=str(identifier.id_))
        return self.cache["post_bodies"].get(identifier_normalized)

    def mark_user_input_sentiment(self, identifier: UserInputIdentifier, sentiment: dict):
        identifier_normalized = UserInputIdentifier(input_type=identifier.input_type,
                                                    blog_name=identifier.blog_name,
                                                    id_=str(identifier.id_) if identifier.id_ is not None else None,
                                                    timestamp=identifier.timestamp)
        self.cache["user_input_sentiments"][identifier_normalized] = sentiment

    def get_cached_user_input_sentiment(self, identifier: UserInputIdentifier):
        identifier_normalized = UserInputIdentifier(input_type=identifier.input_type,
                                                    blog_name=identifier.blog_name,
                                                    id_=str(identifier.id_) if identifier.id_ is not None else None,
                                                    timestamp=identifier.timestamp)
        return self.cache["user_input_sentiments"].get(identifier_normalized)

    def mark_blocked_by_user(self, blog_name: str):
        self.cache['blocked_by_users'].add(blog_name)

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
