from typing import List, Set, Dict
import json

import pytumblr
from pytumblr_wrapper import RateLimitClient

API_KEYS_TYPE = List[str]


class BotSpecificConstants:
    """Values specific to my development environment and/or the social context of my bot, e.g. specific posts IDs where I need apply some override, or specific users I need to treat specially, etc"""

    def __init__(
        self,
        blogName: str,
        dash_blogName: str,
        REBLOG_START_TS: int,
        DASH_START_TS: int,
        private_clients_api_keys: List[API_KEYS_TYPE],
        dashboard_clients_api_keys: List[API_KEYS_TYPE],
        bridge_service_host: str,
        bridge_service_port: int,
        BRIDGE_SERVICE_REMOTE_HOST: str,
        BUCKET_NAME: str,
        NO_REBLOG_IDS: Set[int] = set(),
        DEF_REBLOG_IDS: Set[int] = set(),
        FORCE_TRAIL_HACK_IDS: Set[int] = set(),
        USER_AVOID_LIST: Set[str] = set(),
        TAG_AVOID_LIST: Set[str] = set(),
        DASH_TAG_AVOID_LIST: Set[str] = set(),
        REPLY_USER_AUTO_ACCEPT_LIST: Set[str] = set(),
        bad_strings: Set[str] = set(),
        bad_strings_shortwords: Set[str] = set(),
        okay_superstrings: Set[str] = set(),
        likely_obscured_strings: Set[str] = set(),
        profane_strings: Set[str] = set(),
        LIMITED_USERS: Dict[str, float] = dict(),
        LIMITED_SUBSTRINGS: Dict[str, float] = dict(),
        SCREENED_USERS: Set[str] = set(),
    ):
        # TODO: standardize case in names
        self.blogName = blogName
        self.dash_blogName = dash_blogName

        # when reblog feature started
        self.REBLOG_START_TS = REBLOG_START_TS

        # when reblog-from-dash feature started
        self.DASH_START_TS = DASH_START_TS

        # don't reblog these post IDs -- generally used when I want to write about the bot and then reblog to the bot
        # i don't want a separate bot reblog "responding" to me
        self.NO_REBLOG_IDS = NO_REBLOG_IDS

        self.DEF_REBLOG_IDS = DEF_REBLOG_IDS

        # overrides for tumblr blockquote weirdness
        self.FORCE_TRAIL_HACK_IDS = FORCE_TRAIL_HACK_IDS

        # tumblr api keys (4 strings per key)
        self.private_clients_api_keys = private_clients_api_keys
        self.dashboard_clients_api_keys = dashboard_clients_api_keys

        # host name of the bridge service used in clients we expect to be running on the same machine
        # (i.e. should be localhost under normal circumstances)
        self.bridge_service_host = bridge_service_host

        # port of the bridge service
        self.bridge_service_port = bridge_service_port

        # name of Google Cloud Storage bucket used to store models and data
        self.BUCKET_NAME = BUCKET_NAME

        # host name of the bridge service used in ML code
        # if the ML code is running remotely, this will differ from `bridge_service_host`
        self.BRIDGE_SERVICE_REMOTE_HOST = BRIDGE_SERVICE_REMOTE_HOST

        # don't interact or mention these users
        self.USER_AVOID_LIST = USER_AVOID_LIST

        # bot-written post tags are removed if they contain any of these (substring matches, case-insensitive)
        self.TAG_AVOID_LIST = TAG_AVOID_LIST

        # don't reblog from dash if tags contain these (substring matches)
        self.DASH_TAG_AVOID_LIST = DASH_TAG_AVOID_LIST

        # for frequent repliers who don't otherwise trigger "OK to respond to this reply" logic
        self.REPLY_USER_AUTO_ACCEPT_LIST = REPLY_USER_AUTO_ACCEPT_LIST

        # write draft instead of auto-publish when post/tags contain these substrings
        self.bad_strings = bad_strings

        # form elements of bad_strings from these surrounded by various whitespace/punctuation
        self.bad_strings_shortwords = bad_strings_shortwords

        # ignore items from `bad_strings` when they appear inside of these longer strings
        # e.g. if we wanted to filter "sex" without filtering "anne sexton"
        self.okay_superstrings = okay_superstrings

        # like bad_strings, but we attempt to detect these even if the user is trying to obscure them
        # with e.g. zero-width unicode or l33tsp34k
        self.likely_obscured_strings = likely_obscured_strings

        # like bad_strings, but only used in contexts where we're trying to keep the language rated PG
        self.profane_strings = profane_strings

        # `LIMITED_USERS` allows limiting the rate at which we interact with certain users, e.g. bots who post extremely often or people who send huge numbers of asks
        #
        # `LIMITED_USERS` should be a dict with usernames as keys.  the values are floats.  a value of X means approximately "respond to this user at most once per X hours."
        self.LIMITED_USERS = LIMITED_USERS

        # like `LIMITED_USERS`, but triggers the limiting on the presence of a substring in the input, rather than the name of the user
        self.LIMITED_SUBSTRINGS = LIMITED_SUBSTRINGS

        # write draft instead of auto-publish when responding to these users
        self.SCREENED_USERS = SCREENED_USERS

    @staticmethod
    def load(path: str = "config.json") -> "BotSpecificConstants":
        with open(path, "r", encoding="utf-8") as f:
            constants = json.load(f)

        list_to_set_keys = {
            "NO_REBLOG_IDS",
            "FORCE_TRAIL_HACK_IDS",
            "USER_AVOID_LIST",
            "TAG_AVOID_LIST",
            "DASH_TAG_AVOID_LIST",
            "REPLY_USER_AUTO_ACCEPT_LIST",
            "bad_strings",
            "bad_strings_shortwords",
            "okay_superstrings",
            "likely_obscured_strings",
            "profane_strings",
            "SCREENED_USERS",
        }

        for list_to_set_key in list_to_set_keys:
            constants[list_to_set_key] = set(constants[list_to_set_key])
        return BotSpecificConstants(**constants)

    @property
    def private_clients(self) -> List[RateLimitClient]:
        return [
            RateLimitClient.from_tumblr_rest_client(
                pytumblr.TumblrRestClient(*keys), self.blogName
            )
            for keys in self.private_clients_api_keys
        ]

    @property
    def dashboard_clients(self) -> List[RateLimitClient]:
        return [
            RateLimitClient.from_tumblr_rest_client(
                pytumblr.TumblrRestClient(*keys), self.dash_blogName
            )
            for keys in self.dashboard_clients_api_keys
        ]

    @property
    def bridge_service_url(self):
        return self.bridge_service_host + ":" + str(self.bridge_service_port)

    def LIMITED_USERS_PROBS(self, EFFECTIVE_SLEEP_TIME) -> dict:
        LIMITED_USERS_MINUTES_LOWER_BOUNDS = {
            name: hours * 60 for name, hours in self.LIMITED_USERS.items()
        }
        LIMITED_USERS_PROBS = {
            name: EFFECTIVE_SLEEP_TIME / (60 * lb)
            for name, lb in LIMITED_USERS_MINUTES_LOWER_BOUNDS.items()
        }
        return LIMITED_USERS_PROBS
