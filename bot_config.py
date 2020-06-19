from typing import List, Set
import json

import pytumblr

class BotSpecificConstants:
    """Values specific to my development environment and/or the social context of my bot, e.g. specific posts IDs where I need apply some override, or specific users I need to treat specially, etc"""
    def __init__(self,
                 blogName: str,
                 dash_blogName: str,
                 REBLOG_START_TS: int,
                 DASH_START_TS: int,
                 base_client_api_keys: List[str],
                 dashboard_client_api_keys: List[str],
                 bridge_service_url: str,
                 NO_REBLOG_IDS: Set[int]=set(),
                 FORCE_TRAIL_HACK_IDS: Set[int]=set(),
                 USER_AVOID_LIST: Set[str]=set(),
                 DASH_TAG_AVOID_LIST: Set[str]=set(),
                 REPLY_USER_AUTO_ACCEPT_LIST: Set[str]=set(),
                 bad_strings: Set[str]=set(),
                 bad_strings_shortwords: Set[str]=set(),
                 ):
        self.blogName = blogName
        self.dash_blogName = dash_blogName

        # when reblog feature started
        self.REBLOG_START_TS = REBLOG_START_TS

        # when reblog-from-dash feature started
        self.DASH_START_TS = DASH_START_TS

        # don't reblog these post IDs -- generally used when I want to write about the bot and then reblog to the bot
        # i don't want a separate bot reblog "responding" to me
        self.NO_REBLOG_IDS = NO_REBLOG_IDS

        # overrides for tumblr blockquote weirdness
        self.FORCE_TRAIL_HACK_IDS = FORCE_TRAIL_HACK_IDS

        # tumblr api keys (4 strings per key)
        self.base_client_api_keys = base_client_api_keys
        self.dashboard_client_api_keys = dashboard_client_api_keys

        # should be localhost port 5000 if you run bridge service w/o modification
        self.bridge_service_url = bridge_service_url

        # don't interact or mention these users
        self.USER_AVOID_LIST = USER_AVOID_LIST

        # don't reblog from dash if tags contain these (substring matches)
        self.DASH_TAG_AVOID_LIST = DASH_TAG_AVOID_LIST

        # for frequent repliers who don't otherwise trigger "OK to respond to this reply" logic
        self.REPLY_USER_AUTO_ACCEPT_LIST = REPLY_USER_AUTO_ACCEPT_LIST

        # draft instead of auto-publish when contains these substrings
        self.bad_strings = bad_strings

        # form elements of bad_strings from these surrounded by various whitespace/punctuation
        self.bad_strings_shortwords = bad_strings_shortwords

    @staticmethod
    def load(path: str="config.json") -> 'BotSpecificConstants':
        with open(path, "r", encoding="utf-8") as f:
            constants = json.load(f)
        for list_to_set_key in {"NO_REBLOG_IDS", "FORCE_TRAIL_HACK_IDS",
                                "USER_AVOID_LIST", "DASH_TAG_AVOID_LIST",
                                "REPLY_USER_AUTO_ACCEPT_LIST", "bad_strings",
                                "bad_strings_shortwords"}:
            constants[list_to_set_key] = set(constants[list_to_set_key])
        return BotSpecificConstants(**constants)

    @property
    def base_client(self) -> pytumblr.TumblrRestClient:
        return pytumblr.TumblrRestClient(*self.base_client_api_keys)

    @property
    def dashboard_client(self) -> pytumblr.TumblrRestClient:
        return pytumblr.TumblrRestClient(*reblogs_v5.pyself.dashboard_client_api_keys)
