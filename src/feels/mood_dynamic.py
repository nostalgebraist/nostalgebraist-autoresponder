"""dynamic user-input-responsive part of mood, and mood graphs"""
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

from scipy.signal import lsim, lti
from scipy.signal.ltisys import StateSpaceContinuous

from tqdm.autonotebook import tqdm

from IPython.display import display

from persistence.response_cache import (
    ResponseCache,
    UserInputIdentifier,
)
from feels.mood import (
    random_mood_at_pst_datetime,
    logit_diff_to_pos_sent,
    pos_sent_to_logit_diff,
)

from util.past import MILESTONE_TIMES
from util.times import now_pst, fromtimestamp_pst

MOOD_IMAGE_DIR = "data/mood_images/"

STEP_SEC = 30 * 1
TAU_SEC = 3600 * 12
TAU_SEC_2ND = 60 * 60

WEIGHTED_AVG_START_TIME = pd.Timestamp("2021-01-04 09:10:00")
WEIGHTED_AVG_P75_WEIGHT = 0.5

RESPONSE_SCALE_BASE = 0.15  # 0.1 # 0.2 #0.5
DETERMINER_CENTER = -3.1  # -2.4 # -1.5 #-2
DETERMINER_CENTER_UPDATES = {
    pd.Timestamp("2020-08-20 01:00:00"): -2.4,
    pd.Timestamp("2020-08-25 14:00:00"): -2.0,
    pd.Timestamp("2020-08-31 09:15:00"): -2.4,
    pd.Timestamp("2020-09-16 06:00:00"): -2.1,
    pd.Timestamp("2020-10-28 17:00:00"): -2.4,
    pd.Timestamp("2020-11-04 11:00:00"): -2.78,
    pd.Timestamp("2020-11-13 19:00:00"): -2.7,
    pd.Timestamp("2020-11-15 07:30:00"): -2.6,
    pd.Timestamp("2020-12-04 07:00:00"): -2.5,
    pd.Timestamp("2020-12-10 08:35:00"): -2.35,
    pd.Timestamp("2020-12-10 23:45:00"): -2.0,
    pd.Timestamp("2020-12-18 15:35:00"): -2.2,
    pd.Timestamp("2020-12-21 15:25:00"): -2.3,
    WEIGHTED_AVG_START_TIME: 0.0,
    pd.Timestamp("2021-02-08 09:25:00"): -0.25,
    pd.Timestamp("2021-02-14 17:55:00"): -0.125,
    pd.Timestamp("2021-02-15 17:25:00"): 0,
    pd.Timestamp("2021-02-16 17:45:00"): 0.5,
    pd.Timestamp("2021-02-17 12:45:00"): 0,
    pd.Timestamp("2021-02-26 17:30:00"): 0.5,
    pd.Timestamp("2021-02-27 16:05:00"): 0.,
    pd.Timestamp("2021-03-15 09:55:00"): -0.2,
    pd.Timestamp("2021-03-15 19:50:00"): -0.4,
    pd.Timestamp("2021-03-20 06:55:00"): 0.,
    pd.Timestamp("2021-03-24 22:40:00"): -0.3,
    pd.Timestamp("2021-03-31 12:25:00"): -0.5,
    pd.Timestamp("2021-04-09 07:10:00"): -0.25,
    pd.Timestamp("2021-05-05 17:00:00"): 0.,
    pd.Timestamp("2021-05-07 18:15:00"): -0.25,
    pd.Timestamp("2021-05-12 07:50:00"): 0.,
    pd.Timestamp("2021-05-22 09:50:00"): -0.125,
    pd.Timestamp("2021-05-23 07:15:00"): -0.25,
    pd.Timestamp("2021-06-05 12:05:00"): -0.5,
    pd.Timestamp("2021-06-07 22:35:00"): -0.3,
    pd.Timestamp("2021-06-08 13:15:00"): 0.,
    pd.Timestamp("2021-06-14 06:55:00"): -0.25,
    pd.Timestamp("2021-06-15 18:08:00"): 0.,
    pd.Timestamp("2021-06-16 13:00:00"): 0.125,
    pd.Timestamp("2021-06-26 07:35:00"): 0.25,
    pd.Timestamp("2021-06-30 08:40:00"): 0.,
    pd.Timestamp("2021-08-06 00:45:00"): -0.125,
    pd.Timestamp("2021-09-21 08:25:00"): 0.,
    pd.Timestamp("2021-09-22 17:45:00"): -0.075,
    pd.Timestamp("2021-10-24 12:15:00"): -0.,
    pd.Timestamp("2021-10-24 08:40:00"): 0.125,
    pd.Timestamp("2021-10-25 17:55:00"): 0.25,
    pd.Timestamp("2021-10-28 22:40:00"): 0.125,
    pd.Timestamp("2021-10-31 18:10:00"): 0.05,
    pd.Timestamp("2021-11-02 20:40:00"): 0.,
    pd.Timestamp("2021-11-15 19:20:00"): 0.05,
    pd.Timestamp("2021-11-17 09:10:00"): 0.1,
    pd.Timestamp("2021-11-19 14:50:00"): 0.,
    pd.Timestamp("2021-12-24 14:45:00"): 0.1,
    pd.Timestamp("2021-12-30 09:55:00"): 0.05,
    pd.Timestamp("2022-01-02 08:25:00"): 0.1,
    pd.Timestamp("2022-01-11 12:45:00"): 0.,
    pd.Timestamp("2022-01-26 08:10:00"): 0.075,
    pd.Timestamp("2022-02-05 12:10:00"): 0.025,
    pd.Timestamp("2022-02-18 07:50:00"): 0.,
    pd.Timestamp("2022-02-20 10:00:00"): 0.05,
    pd.Timestamp("2022-02-21 21:30:00"): 0.1,
    pd.Timestamp("2022-02-25 10:00:00"): 0.15,
    pd.Timestamp("2022-03-14 10:20:00"): 0.05,
    pd.Timestamp("2022-03-24 08:30:00"): 0.15,
    pd.Timestamp("2022-03-25 09:50:00"): 0.25,
    pd.Timestamp("2022-05-06 17:40:00"): 0.1,
    pd.Timestamp("2022-05-27 08:10:00"): 0.15,
    pd.Timestamp("2022-06-13 15:30:00"): 0.125,
    pd.Timestamp("2022-06-14 08:40:00"): 0.1,
    pd.Timestamp("2022-06-25 08:50:00"): 0.15,
    pd.Timestamp("2022-06-25 11:40:00"): 0.2,
    pd.Timestamp("2022-06-30 15:10:00"): 0.1,
    pd.Timestamp("2022-06-30 16:05:00"): 0.,
    pd.Timestamp("2022-07-14 00:05:00"): 0.1,
    pd.Timestamp("2022-07-15 19:50:00"): 0.,
}
DETERMINER_MULTIPLIER_UPDATES = {
    pd.Timestamp("2020-08-25 17:00:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2020-10-21 21:15:00"): 0.075 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2020-11-16 10:45:00"): 0.0667 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2020-11-25 11:30:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2020-11-27 08:55:00"): 0.15 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2020-12-04 07:00:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2020-12-09 19:50:00"): 0.075 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2020-12-20 23:30:00"): 0.05 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-01-08 08:55:00"): 0.075 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-01-08 09:10:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-01-13 09:20:00"): 0.15 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-01-14 08:00:00"): 0.2 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-01-14 20:35:00"): 0.15 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-01-20 07:40:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-02-08 09:25:00"): 0.125 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-02-15 06:55:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-02-18 23:45:00"): 0.075 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-03-16 08:55:00"): 0.0667 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-04-03 12:00:00"): 0.075 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-05-11 08:15:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-05-15 08:35:00"): 0.15 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-05-23 07:15:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-06-05 12:05:00"): 0.175 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-06-07 22:35:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-06-16 23:20:00"): 0.075 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-06-26 07:35:00"): 0.0667 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-07-08 06:45:00"): 0.075 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-07-25 09:00:00"): 0.1125 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-09-16 08:35:00"): 0.175 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-10-25 17:55:00"): 0.15 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-11-08 08:30:00"): 0.175 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2021-12-24 08:00:00"): 0.125 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-01-02 08:25:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-01-13 08:10:00"): 0.15 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-02-19 23:10:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-02-21 20:00:00"): 0.05 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-02-25 10:00:00"): 0.07 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-02-28 09:35:00"): 0.085 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-03-08 09:00:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-03-15 08:45:00"): 0.15 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-03-24 08:30:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-03-25 09:50:00"): 0.085 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-05-19 08:30:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-05-21 08:55:00"): 0.12 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-05-27 08:10:00"): 0.1 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-06-25 08:50:00"): 0.085 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-06-30 16:05:00"): 0.1  / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-07-14 00:05:00"): 0.12 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-07-21 00:08:05"): 0.15 / RESPONSE_SCALE_BASE,
    pd.Timestamp("2022-07-30 00:07:00"): 0.12 / RESPONSE_SCALE_BASE,
}

MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP_BASE = {
    "only_sad": 0.094,
    "only_non_happy": 0.37,
    "meh": 0.7,
    "only_non_sad": 0.9,
    "only_happy": 0.99,
}

_ordered_cutoffs = sorted(MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP_BASE.values())
_ordered_cutoffs_moods = [
    (c, [k for k, v in MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP_BASE.items() if v == c][0])
    for c in _ordered_cutoffs
]

MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP = {k: v for k, v in MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP_BASE.items()}
MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP.update(
    {
        "interp_only_non_happy__meh__0.50__0.50": 0.535,  # for dict lookup
        "interp_meh__only_non_sad__0.50__0.50": 0.8,  # for dict lookup
        "unrestricted": 0.7,
    }
)

GENERATED_TS_FIRST_STABLE = pd.Timestamp("2020-05-26 19:00:00")
DUPLICATES_BUGFIX_START_TS = pd.Timestamp("2020-12-19 11:00:00")

WINDOW_LENGTH_DAYS = 2.5


def convert_p75_generated_logit_diff_to_user_input_logit_diff(x):
    # one-time empirical (lr fit)
    return 1.24462721 * x - 1.4965600283833032


def convert_user_input_logit_diff_to_p75_generated_logit_diff(x):
    # one-time empirical (lr fit)
    return (x + 1.4965600283833032) / 1.24462721


def mood_buff_v2(x, y):
    prod = pos_sent_to_logit_diff(x) * pos_sent_to_logit_diff(y)
    result = logit_diff_to_pos_sent(np.sign(prod) * np.sqrt(np.abs(prod)))  # in (0, 1)
    return 2 * (result - 0.5)  # in (-1, 1)


def dynamic_mood_value_to_mood_interp(value, verbose=True) -> str:
    upper_name = None
    upper_dist = None
    for tup in _ordered_cutoffs_moods[::-1]:
        if value < tup[0]:
            upper_name = tup[1]
            upper_dist = tup[0] - value

    lower_name = None
    lower_dist = None
    for tup in _ordered_cutoffs_moods:
        if value >= tup[0]:
            lower_name = tup[1]
            lower_dist = value - tup[0]

    if lower_name is None:
        # below lowest
        return _ordered_cutoffs_moods[0][1]

    if upper_name is None:
        # above highest
        return _ordered_cutoffs_moods[-1][1]

    lower_frac = upper_dist / (lower_dist + upper_dist)
    upper_frac = 1.0 - lower_frac

    interp_name = (
        f"interp_{lower_name}__{upper_name}__{lower_frac:.2f}__{upper_frac:.2f}"
    )

    if verbose:
        print(
            f"interpolating between {lower_frac:.1%} {lower_name} and {upper_frac:.1%} {upper_name}"
        )

    return interp_name


class DynamicMoodSystem:
    def __init__(
        self,
        step_sec: float = STEP_SEC,
        tau_sec: float = TAU_SEC,
        tau_sec_2nd: float = TAU_SEC_2ND,
        response_scale_base: float = RESPONSE_SCALE_BASE,
        determiner_center: float = DETERMINER_CENTER,
        determiner_center_updates: dict = DETERMINER_CENTER_UPDATES,
        determiner_multiplier_updates: dict = DETERMINER_MULTIPLIER_UPDATES,
    ):
        self.step_sec = step_sec
        self.tau_sec = tau_sec
        self.tau_sec_2nd = tau_sec_2nd
        self.response_scale_base = response_scale_base
        self.determiner_center = determiner_center
        self.determiner_center_updates = determiner_center_updates
        self.determiner_multiplier_updates = determiner_multiplier_updates

    @property
    def response_scale(self) -> float:
        return self.response_scale_base * (self.step_sec / self.tau_sec_2nd)

    @property
    def system_matrices(self):
        return (
            [
                [-self.step_sec / self.tau_sec, 1],
                [0, -self.step_sec / self.tau_sec_2nd],
            ],
            [[0], [self.response_scale]],
            [[1, 0]],
            [[0]],
        )

    @property
    def lti_system(self) -> StateSpaceContinuous:
        system = lti(*self.system_matrices)
        return system

    @property
    def forcing_system(self) -> StateSpaceContinuous:
        matrices = self.system_matrices
        system = lti(matrices[0], matrices[1], [[0, 1]], matrices[3])
        return system

    def determiner_center_series(self, determiner: pd.Series) -> pd.Series:
        determiner_center_s = pd.Series(
            [self.determiner_center for _ in determiner.index], index=determiner.index
        )
        start = determiner.index.min()
        for time in sorted(self.determiner_center_updates.keys()):
            determiner_center_s.loc[max(start, time):] = self.determiner_center_updates[time]

        return determiner_center_s

    def determiner_multiplier_series(self, determiner: pd.Series) -> pd.Series:
        determiner_multiplier_s = pd.Series(
            [1.0 for _ in determiner.index], index=determiner.index
        )
        for time in sorted(self.determiner_multiplier_updates.keys()):
            determiner_multiplier_s.loc[time:] = self.determiner_multiplier_updates[
                time
            ]

        return determiner_multiplier_s

    def set_centered_scaled_determiner(self,
                                       mood_inputs: pd.DataFrame,
                                       ) -> pd.DataFrame:
        mood_inputs["centered_determiner"] = mood_inputs["determiner"] - self.determiner_center_series(mood_inputs["determiner"])
        mood_inputs["scaled_determiner"] = self.determiner_multiplier_series(mood_inputs["centered_determiner"]) * mood_inputs["centered_determiner"]
        return mood_inputs


def compute_determiner_legacy(row):
    return convert_p75_generated_logit_diff_to_user_input_logit_diff(
        row.p75_generated_logit_diff
    )


def compute_determiner_weighted_avg(row):
    weighted_avg = ((1 - WEIGHTED_AVG_P75_WEIGHT) * row.logit_diff) + (
        WEIGHTED_AVG_P75_WEIGHT * row.p75_generated_logit_diff
    )

    # one time empirical lr fit, see `sentiment_refresh_2021.ipynb`
    weighted_avg_fitted = (0.61029747 * weighted_avg) + 0.4252486735525668

    return weighted_avg_fitted


def compute_dynamic_mood_inputs(
    response_cache: ResponseCache,
    weighted_avg_start_time: pd.Timestamp = WEIGHTED_AVG_START_TIME,
    system: DynamicMoodSystem = None,
) -> pd.DataFrame:
    if system is None:
        system = DynamicMoodSystem()

    df = pd.DataFrame.from_records(
        [
            {
                "timestamp": ident.timestamp,
                "blog_name": ident.blog_name,
                "logit_diff": sent["logit_diff"],
                "generated_logit_diff": sent.get("generated_logit_diff")
                if sent.get("generated_logit_diff")
                else (
                    [
                        pos_sent_to_logit_diff(entry)
                        for entry in sent.get("generated_pos_sent")
                    ]
                    if "generated_pos_sent" in sent
                    else None
                ),
                "p75_generated_logit_diff": sent.get("p75_generated_logit_diff"),
                "text_for_sentiment": sent.get("text_for_sentiment"),
                "generated_ts": sent.get("generated_ts"),
            }
            for ident, sent in response_cache.user_input_sentiments.items()
        ]
    ).drop_duplicates(subset=["timestamp"])

    _filter = df.generated_logit_diff.notnull() & df.p75_generated_logit_diff.isnull()
    df.loc[_filter, "p75_generated_logit_diff"] = df.loc[
        _filter, "generated_logit_diff"
    ].apply(lambda l: np.percentile(l, 75))

    df["time"] = df.timestamp.apply(lambda ts: fromtimestamp_pst(ts))
    _filter = df.generated_ts.notnull()
    df.loc[_filter, "time"] = df.loc[_filter, "generated_ts"]
    df = df.sort_values(by="time")

    _filter = (df["time"] < GENERATED_TS_FIRST_STABLE) | (df["generated_ts"].notnull())
    if sum(_filter) < len(df):
        print(f"keeping {sum(_filter)} of {len(df)} rows")
    df = df[_filter]

    df["using_weighted_avg"] = df["time"] >= weighted_avg_start_time

    can_compute_determiner = df.p75_generated_logit_diff.notnull() & df.generated_ts.notnull()
    df["determiner"] = 0.
    df.loc[can_compute_determiner == False, "determiner"] = df.logit_diff

    compute_as_legacy = can_compute_determiner & (df.using_weighted_avg == False)
    compute_as_weighted_avg = can_compute_determiner & (df.using_weighted_avg == True)
    df.loc[compute_as_legacy, "determiner"] = df.loc[compute_as_legacy, :].apply(compute_determiner_legacy, axis=1)
    df.loc[compute_as_weighted_avg, "determiner"] = df.loc[compute_as_weighted_avg, :].apply(compute_determiner_weighted_avg, axis=1)

    mood_inputs = df.set_index("time")

    mood_inputs = system.set_centered_scaled_determiner(mood_inputs)

    duplicate_bug_filter = ~mood_inputs.index.duplicated(keep="first")
    duplicate_bug_filter = duplicate_bug_filter | (
        mood_inputs.index < DUPLICATES_BUGFIX_START_TS
    )
    mood_inputs = mood_inputs[duplicate_bug_filter]

    return mood_inputs


def get_unit_mood_inputs(response_cache: ResponseCache, uii: UserInputIdentifier):
    unit_rc = ResponseCache(client=None,
                            path="",
                            backup_path="",
                            cache={"user_input_sentiments": {uii: response_cache.get_cached_user_input_sentiment(uii)}}
                            )
    mi = compute_dynamic_mood_inputs(unit_rc)
    umi = mi.iloc[0]
    return umi


def get_unit_mood_effects_from_interval(response_cache: ResponseCache, start_time: datetime, end_time: datetime):
    sents = response_cache.user_input_sentiments

    subset = {
        uid for uid in sents
        if 'generated_ts' in sents[uid]
        and start_time < sents[uid]['generated_ts'] < end_time
    }

    # TODO: make this faster / less silly
    return {uid: get_unit_mood_inputs(response_cache, uid).scaled_determiner for uid in subset}


def show_unit_mood_inputs(response_cache: ResponseCache, uii: UserInputIdentifier):
    umi = get_unit_mood_inputs(response_cache, uii)
    print(f"mood inputs/effects:")
    for col in ['logit_diff', 'p75_generated_logit_diff', 'determiner', 'centered_determiner']:
        print(f"\t{umi[col]: .2f} | {col}")


def make_mood_inputs_readable(mi, show=True):
    cols = ['blog_name', 'text_for_sentiment',
            'logit_diff', 'p75_generated_logit_diff',
            'determiner', 'centered_determiner']
    shortnames = {
        "logit_diff": "ldiff",
        "p75_generated_logit_diff": "p75",
        "text_for_sentiment": "text",
        "determiner": "det",
        "centered_determiner": "cdet"
    }
    readable = mi[cols].rename(columns=shortnames)
    if show:
        with pd.option_context("display.float_format", lambda x: f"{x:.1f}",
                               "display.max_colwidth", 100):
            display(readable)
    else:
        return readable


def apply_daily_mood_offset(
    lti_series: pd.Series, input_is_logit_diff: bool = True
) -> pd.Series:
    daily_base_mood_series = pd.Series(
        lti_series.index.map(lambda dt: random_mood_at_pst_datetime(dt, verbose=False)),
        index=lti_series.index,
    )

    base_mood_value_series = daily_base_mood_series.map(
        MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP
    )
    if input_is_logit_diff:
        base_mood_value_series = base_mood_value_series.apply(pos_sent_to_logit_diff)

    return lti_series + base_mood_value_series


def compute_dynamic_mood_over_interval(
    mood_inputs: pd.DataFrame,
    start_time: datetime = None,
    end_time: datetime = None,
    system: DynamicMoodSystem = None,
    apply_daily_offset: bool = True,
    forcing_system=False,
) -> pd.Series:
    if start_time is None:
        start_time = mood_inputs.index[0]

    if end_time is None:
        end_time = now_pst()

    if system is None:
        system = DynamicMoodSystem()

    # sentiment_centered = determiner - system.determiner_center_series(determiner)
    # sentiment_centered = (
    #     system.determiner_multiplier_series(sentiment_centered) * sentiment_centered
    # )
    sentiment_centered = mood_inputs["scaled_determiner"]
    if start_time > sentiment_centered.index.max():
        sentiment_centered.loc[start_time] = 0.
    sentiment_centered = sentiment_centered.loc[start_time:end_time]
    sentiment_centered_indexed = sentiment_centered.resample(
        f"{system.step_sec}s"
    ).sum()

    extra_ts_ix = pd.date_range(
        sentiment_centered_indexed.index[-1] + pd.Timedelta(seconds=system.step_sec),
        end_time + pd.Timedelta(seconds=system.step_sec),
        freq=f"{system.step_sec}s",
    )

    extra_ts = pd.Series(np.zeros(len(extra_ts_ix)), index=extra_ts_ix)
    sentiment_centered_indexed_extended = pd.concat(
        [sentiment_centered_indexed, extra_ts]
    ).sort_index()

    start_ts = sentiment_centered_indexed_extended.index[0]

    t = (
        sentiment_centered_indexed_extended.index - start_ts
    ).total_seconds().values / system.step_sec
    u = sentiment_centered_indexed_extended.values

    tout, y, x = lsim(
        system.lti_system if not forcing_system else system.forcing_system,
        u,
        t,
        interp=False,
    )
    lti_series = pd.Series(y, index=sentiment_centered_indexed_extended.index)

    if apply_daily_offset and not forcing_system:
        lti_series = apply_daily_mood_offset(lti_series)

    if not forcing_system:
        lti_series = lti_series.apply(logit_diff_to_pos_sent)

    return lti_series


def compute_dynamic_mood_at_time(
    mood_inputs: pd.DataFrame,
    time: datetime = None,
    window_length_days: float = WINDOW_LENGTH_DAYS,  # pass None for unbounded
    system: DynamicMoodSystem = None,
    apply_daily_offset: bool = True,
) -> float:
    if system is None:
        system = DynamicMoodSystem()

    if time is None:
        time = now_pst()

    start_time = None
    if window_length_days is not None:
        start_time = time - pd.Timedelta(days=window_length_days)

    lti_series = compute_dynamic_mood_over_interval(
        mood_inputs=mood_inputs,
        start_time=start_time,
        end_time=time,
        system=system,
        apply_daily_offset=apply_daily_offset,
    )

    time_indexable = pd.Timestamp(time).round(f"{system.step_sec}s")
    return lti_series.loc[time_indexable]


def compute_dynamic_moodspec_at_time(
    response_cache: ResponseCache,
    time: datetime = None,
    window_length_days: float = None,  # pass None for unbounded
    system: DynamicMoodSystem = None,
    verbose: bool = True,
) -> dict:
    mood_inputs = compute_dynamic_mood_inputs(response_cache)
    mood_value = compute_dynamic_mood_at_time(
        mood_inputs, time, window_length_days
    )
    mood_spec = dynamic_mood_value_to_mood_interp(mood_value, verbose=verbose)

    return mood_spec, mood_value


def create_mood_graph(
    response_cache: ResponseCache,
    start_time: datetime = None,
    end_time: datetime = None,
    window_length_days: float = WINDOW_LENGTH_DAYS,
    system: DynamicMoodSystem = None,
    in_logit_diff_space: bool = True,
    font: str = "Menlo",
    save_image: bool = True,
    show_image: bool = False,
) -> str:
    ytrans = pos_sent_to_logit_diff if in_logit_diff_space else lambda x: x

    mood_inputs = compute_dynamic_mood_inputs(response_cache)
    lti_series = compute_dynamic_mood_over_interval(
        mood_inputs,
        start_time - pd.Timedelta(days=window_length_days),
        end_time,
        system,
    ).apply(ytrans)
    lti_series = lti_series.loc[start_time:end_time]

    fig = plt.figure(figsize=(8, 6))
    plt.plot(lti_series.index, lti_series.values, label="Mood", c="k")

    colors = {
        "only_happy": "#000080",
        "only_non_sad": "#8888FF",
        "only_non_happy": "#FF6666",
        "only_sad": "#800000",
    }
    display_names = {
        "only_sad": ":(",
        "only_non_happy": ":|",
        "only_non_sad": ":)",
        "only_happy": ":D",
    }

    for k in ["only_happy", "only_non_sad", "only_non_happy", "only_sad"]:
        plt.axhline(
            ytrans(MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP[k]),
            label=display_names[k],
            ls="--",
            c=colors[k],
            zorder=1.9,
        )

    if in_logit_diff_space:
        default_top = (
            pos_sent_to_logit_diff(MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP["only_happy"])
            + 1.5
        )
        default_bottom = (
            pos_sent_to_logit_diff(MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP["only_sad"])
            - 1.5
        )

        plt.ylim(
            min(default_bottom, lti_series.min() - 1.5), max(default_top, lti_series.max() + 1.5)
        )

    plt.legend(
        fontsize=16,
    )

    plt.tick_params(labelsize=16)
    plt.tick_params(axis="x", labelrotation=80)

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter("%-I %p"))
    plt.tick_params(axis="x", labelrotation=70)

    plt.grid(axis="x")

    ax = plt.gca()
    for t in ax.get_xticklabels():
        t.set_fontname(font)
    for t in ax.get_yticklabels():
        t.set_fontname(font)
    for t in ax.legend_.texts:
        t.set_fontname(font)

    if save_image:
        image_name = now_pst().strftime("%Y-%m-%d-%H-%M-%S") + ".png"
        path = MOOD_IMAGE_DIR + image_name
        plt.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    if show_image:
        plt.show()
    else:
        plt.close(fig)


def counterfactual_mood_graph(
    mood_inputs,
    determiner_centers,
    determiner_multipliers=None,
    n_days=1,
    start_time: datetime = None,
    end_time: datetime = None,
    window_length_days: float = WINDOW_LENGTH_DAYS,
    in_logit_diff_space: bool = True,
    pairs_only: bool = False,
    include_milestones: bool = True,
    system_kwargs=None
) -> str:
    ytrans = pos_sent_to_logit_diff if in_logit_diff_space else lambda x: x

    if end_time is None:
        end_time = now_pst()
    if start_time is None:
        start_time = end_time - pd.Timedelta(days=n_days)

    systems = {"actual": DynamicMoodSystem()}
    if system_kwargs is not None:
        systems[repr(system_kwargs)] = DynamicMoodSystem(**system_kwargs)

    left_time = start_time - pd.Timedelta(days=window_length_days)

    for dc in determiner_centers:
        if dc is not None:
            new_dc_updates = {
                k: v for k, v in DETERMINER_CENTER_UPDATES.items() if k < left_time
            }
            new_dc_updates[left_time] = dc
            if not pairs_only:
                systems[f"dc={dc:.2f}"] = DynamicMoodSystem(
                    determiner_center_updates=new_dc_updates
                )

        if determiner_multipliers is not None:
            for dm in determiner_multipliers:
                new_dm_updates = {
                    k: v
                    for k, v in DETERMINER_MULTIPLIER_UPDATES.items()
                    if k < left_time
                }
                new_dm_updates[left_time] = dm
                dm_s = f"{dm*RESPONSE_SCALE_BASE:.3f}x"
                if dc is not None:
                    systems[f"dc={dc:.2f}, dm={dm_s}"] = DynamicMoodSystem(
                        determiner_center_updates=new_dc_updates,
                        determiner_multiplier_updates=new_dm_updates,
                    )
                if f"dm={dm_s}" not in systems:
                    if not pairs_only:
                        systems[f"dm={dm_s}"] = DynamicMoodSystem(
                            determiner_multiplier_updates=new_dm_updates
                        )

    lti_serieses = {}
    for name, system in tqdm(systems.items()):
        mood_inputs = system.set_centered_scaled_determiner(mood_inputs)
        lti_series = compute_dynamic_mood_over_interval(
            mood_inputs, left_time, end_time, system
        ).apply(ytrans)
        lti_series = lti_series.loc[start_time:end_time]
        lti_serieses[name] = lti_series

    plt.figure(figsize=(8, 6))
    tops, bottoms = [], []
    for name, lti_series in lti_serieses.items():
        tops.append(lti_series.max())
        bottoms.append(lti_series.min())
        ls = "-"
        if "dc=" in name and "dm=" in name:
            ls = "-."
        elif "dm=" in name:
            ls = "--"
        alpha = 1 if "actual" in name else 0.667
        plt.plot(lti_series.index, lti_series.values, label=name, ls=ls, alpha=alpha)

    print(tops)
    print(bottoms)

    colors = {
        "only_happy": "#000080",
        "only_non_sad": "#8888FF",
        "only_non_happy": "#FF6666",
        "only_sad": "#800000",
    }

    for k in ["only_happy", "only_non_sad", "only_non_happy", "only_sad"]:
        plt.axhline(
            ytrans(MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP[k]),
            ls="--",
            c=colors[k],
        )

    golives = {}
    if include_milestones:
        golives.update(MILESTONE_TIMES)
    golives.update(
        {ts: f"dc -> {dc:.2f}" for ts, dc in DETERMINER_CENTER_UPDATES.items()}
    )
    golives.update(
        {
            ts: f"dm -> {dm*RESPONSE_SCALE_BASE:.3f}x"
            for ts, dm in DETERMINER_MULTIPLIER_UPDATES.items()
        }
    )

    for golive, name in golives.items():
        if golive > start_time:
            c = "r" if name.startswith("v") else ("y" if name.startswith("dc") else "k")
            plt.axvline(golive, c=c, ls="-.", label=name)

    if in_logit_diff_space:
        default_top = (
            pos_sent_to_logit_diff(MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP["only_happy"])
            + 1.5
        )
        default_bottom = (
            pos_sent_to_logit_diff(MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP["only_sad"])
            - 1.5
        )

        plt.ylim(min(default_bottom, min(bottoms) - 1.5), max(default_top, max(tops) + 1.5))

    plt.legend(
        fontsize=12,
    )
    plt.tick_params(labelsize=12)
    plt.tick_params(axis="x", labelrotation=60)

    plt.show()
