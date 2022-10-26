from typing import NamedTuple

from config.autoresponder_config import *  # TODO: move elsewhere?
from util.util import typed_namedtuple_to_dict


def copy_and_update_config(cls, config, **kwargs):
    old_d = typed_namedtuple_to_dict(config)
    new_d = {k: kwargs.get(k) if k in kwargs else v for k, v in old_d.items()}
    return cls(**new_d)


def is_repeating_criterion(unique_token_frac):
    return unique_token_frac < 0.2


SamplingParams = NamedTuple(
    'SamplingParams',
    temperature=float,
    top_k=int,
    top_p=float,
    middle_p=float,
    chop_lowest=float,
    chop_highest=float,
    mirostat=bool,
    breakruns=bool,
    breakruns_tau=float,
    breakruns_decay=float,
    typical_sampling=bool,
    typical_sampling_mass=float,
    typical_sampling_min_tokens_to_keep=int,
    avoid_unk_caption=bool,
    breakruns_off_within_images=bool,
    breakruns_modified_within_images=bool,
    breakruns_temp_modifier=float,
)


SamplingConfig = NamedTuple(
    'SamplingConfig',
    use_first_step=bool,
    first_step_length=int,
    post_window_length=int,
    first_step_params=SamplingParams,
    params=SamplingParams,
    disable_prints=bool,
    max_ctx_fits_on_gpu=int,
    max_continue_steps=int,
    max_continue_tokens=int,
    mirostat_lr=float,
    mirostat_v2=bool,
    mirostat_trunc=int,
)


DEFAULT_SAMPLING_CONFIG = SamplingConfig(
    first_step_params=SamplingParams(
        temperature=first_step_temperature,
        top_k=first_step_top_k,
        top_p=first_step_top_p,
        middle_p=first_step_middle_p,
        chop_lowest=first_step_chop_lowest,
        chop_highest=first_step_chop_highest,
        mirostat=first_step_mirostat,
        breakruns=BREAKRUNS,
        breakruns_tau=FIRST_STEP_BREAKRUNS_TAU,
        breakruns_decay=FIRST_STEP_BREAKRUNS_DECAY,
        typical_sampling=TYPICAL_SAMPLING,
        typical_sampling_mass=TYPICAL_SAMPLING_MASS,
        typical_sampling_min_tokens_to_keep=TYPICAL_SAMPLING_MIN_TOKENS_TO_KEEP,
        avoid_unk_caption=False,
        breakruns_off_within_images=False,
        breakruns_modified_within_images=False,
        breakruns_temp_modifier=0.,
    ),
    params=SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        middle_p=middle_p,
        chop_lowest=chop_lowest,
        chop_highest=chop_highest,
        mirostat=MIRO,
        breakruns=BREAKRUNS,
        breakruns_tau=BREAKRUNS_TAU,
        breakruns_decay=BREAKRUNS_DECAY,
        typical_sampling=TYPICAL_SAMPLING,
        typical_sampling_mass=TYPICAL_SAMPLING_MASS,
        typical_sampling_min_tokens_to_keep=TYPICAL_SAMPLING_MIN_TOKENS_TO_KEEP,
        avoid_unk_caption=False,
        breakruns_off_within_images=False,
        breakruns_modified_within_images=False,
        breakruns_temp_modifier=0.,
    ),
    disable_prints=True,
    first_step_length=first_step_length,
    post_window_length=length,
    max_ctx_fits_on_gpu=max_feed_size_with_cache,
    max_continue_steps=MAX_CONTINUE_STEPS,
    max_continue_tokens=MAX_CONTINUE_TOKENS,
    mirostat_lr=MIRO_LR,
    mirostat_v2=MIRO_V2,
    mirostat_trunc=MIRO_TRUNC,
    use_first_step=USE_FIRST_STEP,
)
