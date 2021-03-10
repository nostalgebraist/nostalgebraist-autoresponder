# HPARAMS, FLAGS, ETC
import os
import subprocess

from bot_config import BotSpecificConstants
from autoresponder_static import *
from autoresponder_static_v8 import *

V8 = True
V8_2 = True
V9 = True
V9_1 = True
V9_1R2 = True
V9_1R3 = True
V9_1R4 = True
V10 = True

bot_specific_constants = BotSpecificConstants.load()
BUCKET_NAME = bot_specific_constants.BUCKET_NAME
BRIDGE_SERVICE_REMOTE_HOST = bot_specific_constants.BRIDGE_SERVICE_REMOTE_HOST
bridge_service_port = bot_specific_constants.bridge_service_port

startdir = os.getcwd()
os.chdir("/")

EOT_WORKAROUND = True
EOT_PREPEND = True
SELECTOR_CAN_SEE_PROMPTS = True
SELECTOR_LR_CALIB_INPUT = "logit_diff"
SELECTOR_LEFT_STRIP_NEWLINE_IN_FORUMLIKE = False

# FORUMLIKE = True  # now set in autoresponder_static
FORUMLIKE_REVIEW_PROB = 0.15
FORUMLIKE_FIC_PROB = 0.15

EVEN_BETTER_LENGTH = True

BATCHONE = True

RANDOM_SAMPLING_PARAMS_ON_STARTUP = False  # True = experimental

DO_ALT_TIMESTAMPS = False  # True is for analytics

eot_end_segment = EOT_FULL if EOT_WORKAROUND else "<|"

final_munge_before_neural = final_munge_before_neural_v10
final_munge_after_neural = final_munge_after_neural_v10

model_name = "autoresponder_v10"
model_path = os.path.join("models", model_name, "model-135.hdf5")

dataset = "data/v10/ALL_data_v10_nost_tuning.npz"
ckpt_select = "selector/v10/v16/.hdf5"
ckpt_sentiment = "sentiment/v10/v2/.hdf5"
ckpt_autoreviewer = "draft_autoreviewer/v10/v3/.hdf5"

TRUNCATE_AT_RIGHT = False
SELECTOR_EOT_PREPEND = True

gs_command_get_dataset = (
    f"gsutil -m cp gs://{BUCKET_NAME}/data/v10/ALL_data_v10_nost_tuning.npz /data/v10/"
)

gs_command_get_encoder = f"gsutil -m cp gs://{BUCKET_NAME}/checkpoint_gs_sync/autoresponder_v10_nost_tuning_f/encoder.json /models/autoresponder_v10/"
gs_command_get_encoder += f"; gsutil -m cp gs://{BUCKET_NAME}/checkpoint_gs_sync/autoresponder_v10_nost_tuning_f/vocab.bpe /models/autoresponder_v10/"

gs_command_get_model = f"gsutil -m cp gs://{BUCKET_NAME}/checkpoint_gs_sync/autoresponder_v10_nost_tuning_f/model-135.hdf5 /models/autoresponder_v10/"

gs_command_get_selector = (
    f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v10_selector/* /selector/v10/"
)
gs_command_get_sentiment = (
    f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v10_sentiment/* /sentiment/v10/"
)
gs_command_get_autoreviewer = (
    f"gsutil -m cp -R gs://{BUCKET_NAME}/draft_autoreviewer/v10/* /draft_autoreviewer/v10/"
)

length_select = 825

SELECT_VIA_GENERATOR_LONGLENGTH = True
SENTIMENT_VIA_GENERATOR_LONGLENGTH = True

length_sentiment = 204


def _gpu_type():
    try:
        gpu_info = subprocess.check_output("nvidia-smi").decode("utf-8")
    except:
        return "small"
    if "P100" in gpu_info:
        return "big"
    else:
        return "small"


if BATCHONE:
    batch_size = 1

    if _gpu_type() == "big":
        max_ctx_fits_on_gpu = 1020
    else:
        max_ctx_fits_on_gpu = 1020

else:
    batch_size = 2

    if _gpu_type() == "big":
        max_ctx_fits_on_gpu = 940  # 825
    else:
        max_ctx_fits_on_gpu = 840  # 1020  #800


# sets max context size, for long prompts we want to cut off to allow bot to write at least this many tokens
required_continuation_room = 100

if EVEN_BETTER_LENGTH:
    better_length = False
    length = required_continuation_room
    # MAX_CONTINUE_TOKENS=2210 if _gpu_type() == "big" else 1600
    MAX_CONTINUE_TOKENS = 1600

    # MAX_CONTINUE_STEPS = 100  # disable via huge
    MAX_CONTINUE_STEPS = (
        2 + 6
    )  # first_sample_op + sample_op_fill_window + 6 x (sample_op_beyond_window)
else:
    better_length = True
    length = max_ctx_fits_on_gpu
    MAX_CONTINUE_TOKENS = 12 * 1024  # disable via huge
    MAX_CONTINUE_STEPS = 12

### Sampling

MIRO_V2 = True
MIRO_TRUNC = 2000  # unused in MIRO_V2

if RANDOM_SAMPLING_PARAMS_ON_STARTUP:
    import numpy as np

    miro_prob = 0.5

    sampling_param_bundles_miro = [
        {
            "MIRO": True,
            "MIRO_LR": _mirolr,
            "T": 1,
            "p": 0,
            "chop_lowest": None,
            "chop_highest": None,
        }
        for _mirolr in [0.05, 0.05, 0.1]
    ]

    sampling_param_bundles_no_miro = [
        {
            "MIRO": False,
            "MIRO_LR": 0.05,
            "T": _T,
            "p": _p,
            "chop_lowest": _chop_lowest,
            "chop_highest": _chop_highest,
        }
        for _T, _p, _chop_lowest, _chop_highest in [
            (1, 0.9, None, None),
            (1, 0.925, None, None),
            (0.95, 0.95, None, None),
            (0.95, 0.9, None, None),
            (0.9, 0.9, None, None),
            (0.9, 0.97, None, None),
            (1, 0, (1 - 0.9), 0.03),  # chop_lowest 1-x = top_p x
            (0.9, 0, (1 - 0.9), 0.05),  # chop_lowest 1-x = top_p x
        ]
    ]

    miro_roll = np.random.rand()
    if miro_roll < miro_prob:
        msg = f"\nSAMPLING: using miro (miro_roll {miro_roll} < miro_prob {miro_prob}"
        choose_from_bundles = sampling_param_bundles_miro
    else:
        msg = f"\nSAMPLING: not using miro (miro_roll {miro_roll} >= miro_prob {miro_prob}"
        choose_from_bundles = sampling_param_bundles_no_miro
    print(msg)

    bundle_ix = int(np.random.choice(list(range(len(choose_from_bundles)))))
    chosen_bundle = choose_from_bundles[bundle_ix]

    msg = f"\nSAMPLING: using bundle {bundle_ix}/{len(choose_from_bundles)}\n\t{chosen_bundle}"
    print(msg)

    MIRO = chosen_bundle["MIRO"]
    MIRO_LR = chosen_bundle["MIRO_LR"]
    temperature = chosen_bundle["T"]
    top_p = chosen_bundle["p"]
    chop_lowest = chosen_bundle["chop_lowest"]
    chop_highest = chosen_bundle["chop_highest"]

    top_k = 0
    middle_p = 0

    MIRO_ONLY_ON_CONTINUE = MIRO
else:
    MIRO = True
    MIRO_ONLY_ON_CONTINUE = True

    temperature = 1
    top_k = 0
    top_p = 0
    middle_p = 0

    # unused
    EXPERIMENTAL_TOP_P = True
    EXPERIMENTAL_TOP_P_2 = True
    EXPERIMENTAL_TOP_P_3 = True
    EXPERIMENTAL_MIDDLE_P_TWEAK = False
    EXPERIMENTAL_MIDDLE_P_ASYM = True

    chop_lowest, chop_highest = None, None

    # MIRO_TRUNC = 50000
    MIRO_TRUNC = 2000

    # MIRO_LR = 0.2
    # MIRO_LR = 0.1  # ! experimental
    MIRO_LR = 0.05

if V8_2:
    MIRO_TARGET_LOW = 2.0
    MIRO_TARGET_HIGH = 2.8
    MIRO_TARGET_ALL = [
        1.8,
        1.9,
        1.9,
        2.0,
        2.0,
        2.1,
        2.1,
        2.1,
        2.2,
        2.2,
        2.2,
        2.3,
        2.3,
        2.3,
        2.4,
        2.4,
        2.5,
        2.6,
        2.7,
        2.8,
        2.9,
        3.0,
    ]

else:
    MIRO_TARGET_LOW = 2.0
    MIRO_TARGET_HIGH = 2.8
    MIRO_TARGET_ALL = [
        2.0,
        2.0,
        2.1,
        2.1,
        2.1,
        2.1,
        2.2,
        2.2,
        2.2,
        2.3,
        2.3,
        2.4,
        2.4,
        2.5,
        2.6,
        2.7,
        2.8,
        2.8,
    ]


if MIRO_ONLY_ON_CONTINUE:
    pre_continue_mirostat = False
    pre_continue_length = 30 # 80  # 50
    pre_continue_temperature = 0.9  # 1
    pre_continue_top_k = 0
    pre_continue_top_p = 0.9  # 0.925
    pre_continue_middle_p = 0
    pre_continue_chop_lowest = None
    pre_continue_chop_highest = None
else:
    pre_continue_mirostat = MIRO
    pre_continue_length = length
    pre_continue_temperature = temperature
    pre_continue_top_k = top_k
    pre_continue_top_p = top_p
    pre_continue_middle_p = middle_p
    pre_continue_chop_lowest = chop_lowest
    pre_continue_chop_highest = chop_highest

print((MIRO, length, temperature, top_p, middle_p, chop_lowest, chop_highest))
print(
    (
        pre_continue_mirostat,
        pre_continue_length,
        pre_continue_temperature,
        pre_continue_top_p,
        pre_continue_middle_p,
        pre_continue_chop_lowest,
        pre_continue_chop_highest,
    )
)

if SELECT_VIA_GENERATOR_LONGLENGTH:
    length_select = max_ctx_fits_on_gpu

if SENTIMENT_VIA_GENERATOR_LONGLENGTH:
    length_sentiment = max_ctx_fits_on_gpu

os.chdir(startdir)
