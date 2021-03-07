"""define mood concept, compute pseudo-random (reproducible) daily mood offset"""
import pickle
from datetime import datetime

import numpy as np
import pandas as pd


DEFAULT_MOOD = "unrestricted"


def logit_diff_to_pos_sent(x):
    return 1 / (1 + np.exp(-x))


def pos_sent_to_logit_diff(x, eps=1e-4):
    return -np.log(1 / min(max(x, eps), 1 - eps) - 1)


def load_logit_diff_sample():
    with open("data/logit_diff_sample.pkl.gz", "rb") as f:
        logit_diff_sample = pickle.load(f)
    return pd.Series(logit_diff_sample)


def estimate_expected_rejections(
    min_logit_diff: float, max_logit_diff: float, logit_diff_sample_series: pd.Series
):
    return (
        (logit_diff_sample_series < min_logit_diff)
        | (logit_diff_sample_series > max_logit_diff)
    ).mean()


def get_mood_by_name(mood_name: str):
    bound_names = {"no_lower_bound": 0.0, "no_upper_bound": 1.0}

    moods_original_flavor = {
        "only_sad": {
            "min_allowed_score": bound_names["no_lower_bound"],
            "max_allowed_score": 0.1,
            "score_fn": "pos_sentiment",
        },
        "only_non_happy": {
            "min_allowed_score": bound_names["no_lower_bound"],
            "max_allowed_score": 0.2,
            "score_fn": "pos_sentiment",
        },
        "meh": {
            "min_allowed_score": 0.02,
            "max_allowed_score": 0.6,
            "score_fn": "pos_sentiment",
        },
        "only_non_sad": {
            "min_allowed_score": 0.15,
            "max_allowed_score": bound_names["no_upper_bound"],
            "score_fn": "pos_sentiment",
        },
        "only_happy": {
            "min_allowed_score": 0.3,  # 0.4,
            "max_allowed_score": bound_names["no_upper_bound"],
            "score_fn": "pos_sentiment",
        },
        "unrestricted": {
            "min_allowed_score": bound_names["no_lower_bound"],
            "max_allowed_score": bound_names["no_upper_bound"],
            "score_fn": "pos_sentiment",
        },
    }

    moods_logit_diff_version = {}
    for k, v in moods_original_flavor.items():
        moods_logit_diff_version[k] = v

        moods_logit_diff_version[k]["min_allowed_score"] = pos_sent_to_logit_diff(
            moods_logit_diff_version[k]["min_allowed_score"]
        )
        moods_logit_diff_version[k]["max_allowed_score"] = pos_sent_to_logit_diff(
            moods_logit_diff_version[k]["max_allowed_score"]
        )
        moods_logit_diff_version[k]["score_fn"] = "logit_diff"
    moods = moods_logit_diff_version

    moods_ = {}
    for k, v in moods.items():
        moods_[k] = v
        moods_[k]["name"] = k
    moods = moods_

    if mood_name.startswith("interp_"):
        segments = mood_name[len("interp_") :].split("__")

        lower_mood = moods[segments[0]]
        upper_mood = moods[segments[1]]
        lower_frac = float(segments[2])
        upper_frac = float(segments[3])

        interp_min_allowed_score = (lower_frac * lower_mood["min_allowed_score"]) + (
            upper_frac * upper_mood["min_allowed_score"]
        )

        interp_max_allowed_score = (lower_frac * lower_mood["max_allowed_score"]) + (
            upper_frac * upper_mood["max_allowed_score"]
        )

        interp_mood = {
            "name": mood_name,
            "score_fn": "logit_diff",
            "min_allowed_score": interp_min_allowed_score,
            "max_allowed_score": interp_max_allowed_score,
        }
        return interp_mood

    if mood_name not in moods:
        print(f"couldn't find {mood_name}, using default {DEFAULT_MOOD}")
        return DEFAULT_MOOD
    return moods[mood_name]


def random_mood_at_pst_datetime(dt: datetime, verbose=True):
    if dt > datetime(2020, 5, 27, 9, 24, 43):
        mood_options = [
            "only_non_happy",
            "meh",
            "only_non_sad",
        ]
    else:
        # legacy (but post-moon)
        mood_options = [
            "only_non_happy",
            "meh",
            "unrestricted",
            "only_non_sad",
            "only_happy",
        ]

    # for continuity with moon
    if dt.strftime("%Y-%m-%d") == "2020-05-22":
        return "only_non_sad"

    seed = int(str(dt.year) + str(dt.month) + str(dt.day))
    if verbose:
        print(f"using seed: {seed}")

    mood_chosen = np.random.RandomState(seed=seed).choice(mood_options)

    if verbose:
        print(f"on {dt.strftime('%Y-%m-%d')}, my mood is {mood_chosen}")

    return mood_chosen
