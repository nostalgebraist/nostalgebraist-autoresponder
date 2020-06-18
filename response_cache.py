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

    def _api_call_for_rtype(self, rtype: CachedResponseType, identifier: PostIdentifier):
        time.sleep(0.33)
        if rtype == CachedResponseType.POSTS:
            response = self.client.posts(identifier.blog_name, id=identifier.id_)
            if 'posts' not in response:
                if response.get('response') == "You do not have permission to view this blog":
                    self.mark_blocked_by_user(reblog_identifier.blog_name)
                return None
            return response['posts'][0]
        elif rtype == CachedResponseType.NOTES:
            response = self.client.notes(identifier.blog_name, id=identifier.id_)
            notes = response['notes']
            while (response['total_notes'] > (len(notes)-1)) and ("_links" in response):
                time.sleep(0.33)
                response = self.client.notes(identifier.blog_name, id=identifier.id_, before_timestamp=response["_links"]["next"]["query_params"]["before_timestamp"])
                notes.extend(response["notes"])
            return notes
        else:
            raise ValueError(f"rtype {rtype} not understood")

    def _cached_note_count(self, rtype, identifier, use_overrides=True):
        if ((identifier, "note_count_override") in self.cache[rtype]) and use_overrides:
            return self.cache[rtype][(identifier, "note_count_override")]
        return len(self.cache[rtype][identifier]) - 1 # minus one b/c posting is a 'note' in notes payload

    def _can_use_cached(self, rtype: CachedResponseType, identifier: PostIdentifier, expected_notes: int=None):
        is_in_cache = identifier in self.cache[rtype]
        cache_uptodate = True

        if rtype == CachedResponseType.NOTES and is_in_cache:
            cached_notes = self._cached_note_count(rtype, identifier)
            cache_uptodate = expected_notes == cached_notes
            if not cache_uptodate:
                print(f"note cache stale for {identifier}: expected {expected_notes} notes but have {cached_notes} in cache")

        return is_in_cache and cache_uptodate

    def _record_unexpected_note_counts(self, rtype, identifier, expected_notes):
        cached_notes = self._cached_note_count(rtype, identifier, use_overrides=False)
        if cached_notes != expected_notes:
            print(f"cache note count {cached_notes} still != expected {expected_notes} for {identifier}, marking {expected_notes} as override")
            self.cache[rtype][(identifier, "note_count_override")] = expected_notes
        if (cached_notes == expected_notes) and ((identifier, "note_count_override") in self.cache[rtype]):
            print(f"cache note count {cached_notes} = expected {expected_notes} for {identifier}, unsetting override")
            del self.cache[rtype][(identifier, "note_count_override")]

    def query(self, rtype: CachedResponseType, identifier: PostIdentifier, expected_notes: int=None):
        if self._can_use_cached(rtype, identifier, expected_notes):
            return self.cache[rtype][identifier]
        else:
            response = self._api_call_for_rtype(rtype, identifier)
            self.cache[rtype][identifier] = response
            if rtype == CachedResponseType.NOTES:
                self._record_unexpected_note_counts(rtype, identifier, expected_notes)
            return response

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
