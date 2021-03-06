"""dynamic user-input-responsive part of mood, and mood graphs"""
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

from scipy.signal import lsim, lti, StateSpace
from scipy.signal.ltisys import StateSpaceContinuous

from response_cache import ResponseCache, UserInputType, PostIdentifier, ReplyIdentifier, CachedResponseType
from mood import get_mood_by_name, random_mood_at_pst_datetime, logit_diff_to_pos_sent, pos_sent_to_logit_diff

STEP_SEC = 30 * 1
TAU_SEC = 3600 * 12
TAU_SEC_2ND = 60 * 60

RESPONSE_SCALE_BASE = 0.1 # 0.2 #0.5
DETERMINER_CENTER = -2.4 # -1.5 #-2

MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP = {
 'only_sad': 0.094,
 'only_non_happy': 0.37,
 'meh': 0.7,
 'unrestricted': 0.7,
 'only_non_sad': 0.9,
 'only_happy': 0.99
}

_ordered_cutoffs = sorted(MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP.values())
_ordered_cutoffs_moods = [(c, [k for k, v in MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP.items() if v == c][0])
                          for c in _ordered_cutoffs]

GENERATED_TS_FIRST_STABLE = pd.Timestamp('2020-05-26 19:00:00')

def logit_diff_to_pos_sent(x):
    return 1/(1+np.exp(-x))

def pos_sent_to_logit_diff(x, eps=1e-4):
    return -np.log(1/max(x, eps)-1)

def convert_p75_generated_logit_diff_to_user_input_logit_diff(x):
    # one-time empirical (lr fit)
    return 1.24462721*x - 1.4965600283833032

def convert_user_input_logit_diff_to_p75_generated_logit_diff(x):
    # one-time empirical (lr fit)
    return (x+1.4965600283833032)/1.24462721

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

    lower_mood = get_mood_by_name(lower_name)
    upper_mood = get_mood_by_name(upper_name)

    lower_frac = upper_dist / (lower_dist + upper_dist)
    upper_frac = 1. - lower_frac

    interp_name = f"interp_{lower_name}__{upper_name}__{lower_frac:.2f}__{upper_frac:.2f}"

    if verbose:
        print(f"interpolating between {lower_frac:.1%} {lower_name} and {upper_frac:.1%} {upper_name}")

    return interp_name


class DynamicMoodSystem:
    def __init__(self,
                 step_sec: float=STEP_SEC,
                 tau_sec: float=TAU_SEC,
                 tau_sec_2nd: float=TAU_SEC_2ND,
                 response_scale_base: float=RESPONSE_SCALE_BASE,
                 determiner_center: float=DETERMINER_CENTER,
                 ):
        self.step_sec = step_sec
        self.tau_sec = tau_sec
        self.tau_sec_2nd = tau_sec_2nd
        self.response_scale_base = response_scale_base
        self.determiner_center = determiner_center

    @property
    def response_scale(self) -> float:
        return self.response_scale_base * (self.step_sec / self.tau_sec_2nd)

    @property
    def system_matrices(self):
        return ([[-self.step_sec/self.tau_sec, 1], [0, -self.step_sec/self.tau_sec_2nd]],
        [[0], [self.response_scale]],
        [[1, 0]],
        [[0]])

    @property
    def lti_system(self) -> StateSpaceContinuous:
        system = lti(*self.system_matrices)
        return system

    @property
    def forcing_system(self) -> StateSpaceContinuous:
        matrices = self.system_matrices
        system = lti(matrices[0], matrices[1], [[0, 1]], matrices[3])
        return system


def compute_dynamic_mood_inputs(response_cache: ResponseCache) -> pd.DataFrame:
    df = pd.DataFrame.from_records(
    [{"timestamp": ident.timestamp,
      "blog_name": ident.blog_name,
      "pos_sent": sent["prob"] if sent["label"]=="1" else 1.-sent["prob"],
      "pos_logit": sent["logits"][0],
      "neg_logit": sent["logits"][1],
      "logit_diff": sent["logits"][0] - sent["logits"][1],
      "generated_logit_diff": [pos_sent_to_logit_diff(entry)
                               for entry in sent.get("generated_pos_sent")
                               ] if "generated_pos_sent" in sent else None,
      "text_for_sentiment": sent.get("text_for_sentiment"),
      "generated_ts": sent.get("generated_ts")}
     for ident, sent in response_cache.user_input_sentiments.items()
    ]
    ).drop_duplicates(subset=["timestamp"])

    df["time"] = df.timestamp.apply(lambda ts: datetime.fromtimestamp(ts))
    _filter = df.generated_ts.notnull()
    df.loc[_filter, "time"] = df.loc[_filter, "generated_ts"]
    df = df.sort_values(by="time")

    _filter = (df["time"] < GENERATED_TS_FIRST_STABLE) | (df["generated_ts"].notnull())
    if sum(_filter) < len(df):
        print(f"keeping {sum(_filter)} of {len(df)} rows")
    df = df[_filter]

    _filter = df.generated_logit_diff.notnull()
    df.loc[_filter, "p75_generated_logit_diff"] = df.loc[_filter, "generated_logit_diff"].apply(
        lambda l: np.percentile(l, 75)
    )

    mood_inputs = df.set_index("time")
    return mood_inputs


def apply_daily_mood_offset(lti_series: pd.Series, input_is_logit_diff: bool=True) -> pd.Series:
    daily_base_mood_series = pd.Series(
        lti_series.index.map(lambda dt: random_mood_at_pst_datetime(dt, verbose=False)),
        index=lti_series.index
    )

    base_mood_value_series = daily_base_mood_series.map(MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP)
    if input_is_logit_diff:
        base_mood_value_series = base_mood_value_series.apply(pos_sent_to_logit_diff)

    return lti_series + base_mood_value_series


def compute_dynamic_mood_over_interval(mood_inputs: pd.DataFrame,
                                       start_time: datetime=None,
                                       end_time: datetime=None,
                                       system: DynamicMoodSystem=None,
                                       apply_daily_offset: bool=True,
                                       use_p75: bool=True,
                                       forcing_system=False) -> pd.Series:
    if start_time is None:
        start_time = mood_inputs.index[0]

    if end_time is None:
        end_time = datetime.now()

    if system is None:
        system = DynamicMoodSystem()

    if use_p75:
        determiner = mood_inputs.apply(
            lambda row: convert_p75_generated_logit_diff_to_user_input_logit_diff(row.p75_generated_logit_diff)
            if (np.isfinite(row.p75_generated_logit_diff) and (row.generated_ts == row.generated_ts))
            else row.logit_diff,
            axis=1
        )
    else:
        determiner = mood_inputs.logit_diff

    sentiment_centered = (determiner - system.determiner_center)
    sentiment_centered = sentiment_centered.loc[start_time:]
    sentiment_centered_indexed = sentiment_centered.resample(f"{system.step_sec}s").sum()

    extra_ts_ix = pd.date_range(
            sentiment_centered_indexed.index[-1]+pd.Timedelta(seconds=system.step_sec),
            end_time + pd.Timedelta(seconds=system.step_sec),
            freq=f"{system.step_sec}s"
            )

    extra_ts = pd.Series(np.zeros(len(extra_ts_ix)), index=extra_ts_ix)
    sentiment_centered_indexed_extended = pd.concat([sentiment_centered_indexed, extra_ts]).sort_index()

    start_ts = sentiment_centered_indexed_extended.index[0]

    t = (sentiment_centered_indexed_extended.index - start_ts).total_seconds().values / system.step_sec
    u = sentiment_centered_indexed_extended.values

    tout, y, x = lsim(system.lti_system if not forcing_system else system.forcing_system, u, t, interp=False)
    lti_series = pd.Series(y, index=sentiment_centered_indexed_extended.index)

    if apply_daily_offset and not forcing_system:
        lti_series = apply_daily_mood_offset(lti_series)

    if not forcing_system:
        lti_series = lti_series.apply(logit_diff_to_pos_sent)

    return lti_series



def compute_dynamic_mood_at_time(mood_inputs: pd.DataFrame,
                                 time: datetime=None,
                                 window_length_days: float=None,  # pass None for unbounded
                                 system: DynamicMoodSystem=None,
                                 apply_daily_offset: bool=True,
                                 use_p75: bool=True) -> float:
    if system is None:
        system = DynamicMoodSystem()

    if time is None:
        time = datetime.now()

    start_time = None
    if window_length_days is not None:
        start_time = time - pd.Timedelta(days=window_length_days)

    lti_series = compute_dynamic_mood_over_interval(
        mood_inputs=mood_inputs,
        start_time=start_time,
        end_time=time,
        system=system,
        apply_daily_offset=apply_daily_offset,
        use_p75=use_p75
    )

    time_indexable = pd.Timestamp(time).round(f"{system.step_sec}s")
    return lti_series.loc[time_indexable]


def compute_dynamic_moodspec_at_time(response_cache: ResponseCache,
                                     time: datetime=None,
                                     window_length_days: float=None,  # pass None for unbounded
                                     system: DynamicMoodSystem=None,
                                     use_p75: bool=True,
                                     verbose: bool=True
                                     ) -> dict:
    mood_inputs = compute_dynamic_mood_inputs(response_cache)
    mood_value = compute_dynamic_mood_at_time(mood_inputs, time, window_length_days, use_p75=use_p75)
    mood_spec = dynamic_mood_value_to_mood_interp(mood_value, verbose=verbose)

    return mood_spec, mood_value


def create_mood_graph(response_cache: ResponseCache,
                      start_time: datetime=None,
                      end_time: datetime=None,
                      system: DynamicMoodSystem=None) -> str:
    mood_inputs = compute_dynamic_mood_inputs(response_cache)
    lti_series = compute_dynamic_mood_over_interval(mood_inputs, start_time, end_time, system)

    plt.figure(figsize=(8, 6))
    plt.plot(lti_series.index, lti_series.values, label="Mood", c='k')

    colors = {"only_happy": "#000080",
              "only_non_sad": "#8888FF",
              "only_non_happy": "#FF6666",
              "only_sad": "#800000",
              }
    display_names = {"only_sad": ":(",
                     "only_non_happy": ":|",
                     "only_non_sad": ":)",
                     "only_happy": ":D"}

    for k in ["only_happy", "only_non_sad", "only_non_happy", "only_sad"]:
        plt.axhline(MOOD_NAME_TO_DYNAMIC_MOOD_VALUE_MAP[k], label=display_names[k], ls='--', c=colors[k])

    plt.legend(fontsize=16, );
    plt.tick_params(labelsize=16)
    plt.tick_params(axis='x', labelrotation=80)

    image_name = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".png"
    path = "mood_images/" + image_name
    plt.savefig(path, bbox_inches='tight')
    return path
