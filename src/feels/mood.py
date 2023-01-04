"""define mood concept, compute pseudo-random (reproducible) daily mood offset"""
import pickle
from datetime import datetime

import numpy as np
import pandas as pd

from smart_open import open
from google.auth import exceptions

from util.cloudsave import CLOUDSAVE_BUCKET

DEFAULT_MOOD = "unrestricted"

INTERPOLATE_IN_SENT_SPACE_DEFAULT = True  # ! testing, 7/4/22


def logit_diff_to_pos_sent(x):
    return 1 / (1 + np.exp(-x))


def pos_sent_to_logit_diff(x, eps=None):
    if not eps:
        eps = np.finfo(np.float64).eps
    return -np.log(1 / min(max(x, eps), 1 - eps) - 1)


def logit_diff_to_allen_schema(logit_diff: float):
    label = "1" if logit_diff > 0 else "0"

    pos_logit = logit_diff / 2
    neg_logit = -1 * logit_diff / 2
    logits = [pos_logit, neg_logit]

    prob_pos = logit_diff_to_pos_sent(logit_diff)
    prob = prob_pos if logit_diff > 0 else (1.0 - prob_pos)

    entry = {"label": label, "prob": prob, "logits": logits}
    return entry


def load_logit_diff_sample():
    try:
        with open(f"gs://{CLOUDSAVE_BUCKET}/nbar_data/logit_diff_sample.pkl.gz", "rb") as f:
            logit_diff_sample = pickle.load(f)
    except (FileNotFoundError, exceptions.GoogleAuthError):
        logit_diff_sample = [
            -4.055, -0.483, -5.576, -2.096, -5.473, -2.132, -1.503, -5.712, 2.182,
            -1.419, -2.548, -2.114, 0.071, -3.568, -1.11, 0.379, -1.584, -0.045,
            -9.21, -2.885, -9.21, 9.807
        ]
    return pd.Series(logit_diff_sample)


def estimate_expected_rejections(
    min_logit_diff: float, max_logit_diff: float, logit_diff_sample_series: pd.Series
):
    return (
        (logit_diff_sample_series < min_logit_diff)
        | (logit_diff_sample_series > max_logit_diff)
    ).mean()


def get_mood_by_name(mood_name: str, interpolate_in_sent_space=INTERPOLATE_IN_SENT_SPACE_DEFAULT):
    bound_names = {"no_lower_bound": 0.0, "no_upper_bound": 1.0}

    moods_original_flavor = {
        "only_sad": {
            "min_allowed_score": bound_names["no_lower_bound"],
            "max_allowed_score": 0.2593,
            "score_fn": "pos_sentiment",
        },
        "only_non_happy": {
            "min_allowed_score": bound_names["no_lower_bound"],
            "max_allowed_score": 0.462,
            "score_fn": "pos_sentiment",
        },
        "meh": {
            "min_allowed_score": 0.0128,
            "max_allowed_score": 0.7031,
            "score_fn": "pos_sentiment",
        },
        "only_non_sad": {
            "min_allowed_score": 0.0483,
            "max_allowed_score": bound_names["no_upper_bound"],
            "score_fn": "pos_sentiment",
        },
        "only_happy": {
            "min_allowed_score": 0.1192,
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
    for k in moods_original_flavor.keys():
        moods_logit_diff_version[k] = {}

        moods_logit_diff_version[k]["min_allowed_score"] = pos_sent_to_logit_diff(
            moods_original_flavor[k]["min_allowed_score"]
        )
        moods_logit_diff_version[k]["max_allowed_score"] = pos_sent_to_logit_diff(
            moods_original_flavor[k]["max_allowed_score"]
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

        if interpolate_in_sent_space:
            lower_mood = moods_original_flavor[segments[0]]
            upper_mood = moods_original_flavor[segments[1]]
        else:
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

        if interpolate_in_sent_space:
            interp_min_allowed_score = pos_sent_to_logit_diff(interp_min_allowed_score)
            interp_max_allowed_score = pos_sent_to_logit_diff(interp_max_allowed_score)

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
    if dt >= datetime(2022, 7, 5, 0, 0, 0):
        mood_options = [
            "interp_only_non_happy__meh__0.50__0.50",
            "meh",
            "interp_meh__only_non_sad__0.50__0.50",
        ]
    elif dt > datetime(2020, 5, 27, 9, 24, 43):
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
