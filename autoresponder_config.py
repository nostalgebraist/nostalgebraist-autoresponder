# HPARAMS, FLAGS, ETC
# TODO: refactor this terrible, terrible file
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
V10_1 = True
V10_1_torch = True  # !!
V11 = True  # !!!! -- gptneo, th
V11_INSURANCE = False

USE_AUTOREVIEWER = True
if V11:
    # TODO: fill
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.158,  # v11/v11: predict true accept rate: ~50%, false accept rate ~6.7%
        "reject_above": 0.461,  # v11/v11: predict true reject rate: ~59%, false reject rate ~5%
    }
elif V10_1_torch:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.158,  # v10_1_torch/v1: predict true accept rate: ~38%, false accept rate ~6.7%
        "reject_above": 0.467,  # v10_1_torch/v1: predict true reject rate: ~54%, false reject rate ~6%
    }

else:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.145,  # v10_1/v11: predict true accept rate: ~40%, false accept rate ~6.7%
        "reject_above": 0.541,  # v10_1/v11: predict true reject rate: ~49%, false reject rate ~6%
    }

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

if V10_1:
    final_munge_before_neural = final_munge_before_neural_v10_1
    final_munge_after_neural = final_munge_after_neural_v10_1
else:
    final_munge_before_neural = final_munge_before_neural_v10
    final_munge_after_neural = final_munge_after_neural_v10

if V11:
    model_name = "neo_ar_2_7B_v0_nost_tuning_f"
    model_path = os.path.join("/", model_name)
elif V10_1_torch:
    model_name = "torch__autoresponder_v10_1"
    model_path = os.path.join("/", model_name)
elif V10_1:
    model_name = "autoresponder_v10_1"
    model_path = os.path.join("models", model_name, "model-141.hdf5")
else:
    model_name = "autoresponder_v10"
    model_path = os.path.join("models", model_name, "model-135.hdf5")

if V11:
    ckpt_select = "selector/v11/v2/"
    ckpt_sentiment = "sentiment/v11/v2/"
    ckpt_autoreviewer = "draft_autoreviewer/v11/v1/"
elif V10_1_torch:
    ckpt_select = "selector/v10_1_torch/v1/"
    ckpt_sentiment = "sentiment/v10_1_torch/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v10_1_torch/v1/"
elif V10_1:
    ckpt_select = "selector/v10_1/v8/.hdf5"
    ckpt_sentiment = "sentiment/v10_1/v1/.hdf5"
    ckpt_autoreviewer = "draft_autoreviewer/v10_1/v11/.hdf5"
else:
    ckpt_select = "selector/v10/v17/.hdf5"
    ckpt_sentiment = "sentiment/v10/v2/.hdf5"
    ckpt_autoreviewer = "draft_autoreviewer/v10/v3/.hdf5"

TRUNCATE_AT_RIGHT = False
SELECTOR_EOT_PREPEND = True

gs_command_get_encoder = f"gsutil -m cp gs://{BUCKET_NAME}/checkpoint_gs_sync/autoresponder_v10_nost_tuning_f/encoder.json /models/{model_name}/"
gs_command_get_encoder += f"; gsutil -m cp gs://{BUCKET_NAME}/checkpoint_gs_sync/autoresponder_v10_nost_tuning_f/vocab.bpe /models/{model_name}/"

if V11:
    gs_command_get_model = f"gsutil -m cp gs://{BUCKET_NAME}/tf_to_torch/neo_ar_2_7B_v0_nost_tuning_f/* {model_path}"

    gs_command_get_selector = (
        f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v11_selector/* /selector/v11/"
    )
    gs_command_get_sentiment = f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v11_sentiment/* /sentiment/v11/"
    gs_command_get_autoreviewer = f"gsutil -m cp -R gs://{BUCKET_NAME}/draft_autoreviewer/v11/* /draft_autoreviewer/v11/"
elif V10_1_torch:
    gs_command_get_model = f"gsutil -m cp gs://{BUCKET_NAME}/tf_to_torch/{model_name}/* {model_path}"

    gs_command_get_selector = (
        f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v10_1_torch_selector/* /selector/v10_1_torch/"
    )
    gs_command_get_sentiment = f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v10_1_torch_sentiment/* /sentiment/v10_1_torch/"
    gs_command_get_autoreviewer = f"gsutil -m cp -R gs://{BUCKET_NAME}/draft_autoreviewer/v10_1_torch/* /draft_autoreviewer/v10_1_torch"
elif V10_1:
    gs_command_get_model = f"gsutil -m cp gs://{BUCKET_NAME}/checkpoint_gs_sync/autoresponder_v10_1_nost_tuning_f/model-141.hdf5 /models/autoresponder_v10_1/"

    gs_command_get_selector = f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v10_1_selector/* /selector/v10_1/"
    gs_command_get_sentiment = f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v10_1_sentiment/* /sentiment/v10_1/"
    gs_command_get_autoreviewer = f"gsutil -m cp -R gs://{BUCKET_NAME}/draft_autoreviewer/v10_1/* /draft_autoreviewer/v10_1/"
else:
    gs_command_get_model = f"gsutil -m cp gs://{BUCKET_NAME}/checkpoint_gs_sync/autoresponder_v10_nost_tuning_f/model-135.hdf5 /models/autoresponder_v10/"

    gs_command_get_selector = (
        f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v10_selector/* /selector/v10/"
    )
    gs_command_get_sentiment = f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v10_sentiment/* /sentiment/v10/"
    gs_command_get_autoreviewer = f"gsutil -m cp -R gs://{BUCKET_NAME}/draft_autoreviewer/v10/* /draft_autoreviewer/v10/"

length_select = 825

SELECT_VIA_GENERATOR_LONGLENGTH = True
SENTIMENT_VIA_GENERATOR_LONGLENGTH = True

length_sentiment = 204


def _gpu_type():
    try:
        gpu_info = subprocess.check_output("nvidia-smi").decode("utf-8")
    except:
        return "small"
    if "P100" in gpu_info or "V100" in gpu_info:
        return "big"
    else:
        return "small"


GPU_TYPE = _gpu_type()

batch_size = 1

max_ctx_fits_on_gpu = 1020

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

BREAKRUNS = True
BREAKRUNS_TAU = 0.02  # 0.03
BREAKRUNS_DECAY = 0.0
BREAKRUNS_DEBUG = False

temperature = 0.9  # 0.85
top_k = 0
top_p = 0.95
middle_p = 0

FIRST_STEP_BREAKRUNS = True  # disable via tau=0
FIRST_STEP_BREAKRUNS_TAU = 0.0
FIRST_STEP_BREAKRUNS_DECAY = 0.0

MIRO_V2 = False
MIRO_TRUNC = 2000  # unused in MIRO_V2

MIRO = False
USE_FIRST_STEP = False

# unused
EXPERIMENTAL_TOP_P = True
EXPERIMENTAL_TOP_P_2 = True
EXPERIMENTAL_TOP_P_3 = True
EXPERIMENTAL_MIDDLE_P_TWEAK = False
EXPERIMENTAL_MIDDLE_P_ASYM = True

chop_lowest, chop_highest = None, None

# MIRO_LR = 0.2
# MIRO_LR = 0.1
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


if USE_FIRST_STEP and RANDOM_SAMPLING_PARAMS_ON_STARTUP:
    import numpy as np

    sampling_param_bundles = [
        {
            "T": _T,
            "p": _p,
            "chop_lowest": _chop_lowest,
            "chop_highest": _chop_highest,
        }
        for _T, _p, _chop_lowest, _chop_highest in [
            (1, 0.9, None, None),
            (0.9, 0, (1 - 0.9), 0.075),  # chop_lowest 1-x = top_p x
            (1, 0, (1 - 0.925), 0.075),  # chop_lowest 1-x = top_p x
        ]
    ]

    bundle_ix = int(np.random.choice(list(range(len(sampling_param_bundles)))))
    chosen_bundle = sampling_param_bundles[bundle_ix]

    first_step_mirostat = False
    first_step_temperature = chosen_bundle["T"]
    first_step_top_p = chosen_bundle["p"]
    first_step_chop_lowest = chosen_bundle["chop_lowest"]
    first_step_chop_highest = chosen_bundle["chop_highest"]

    first_step_length = int(np.random.choice([35, 55, 85, 105]))

    # unused
    first_step_top_k = 0
    first_step_middle_p = 0
elif USE_FIRST_STEP:
    first_step_mirostat = False
    first_step_length = 25  # 20  # 30  # 80  # 50
    first_step_temperature = 1  # 0.9
    first_step_top_k = 0
    first_step_top_p = 0.9  # 0.925
    first_step_middle_p = 0
    first_step_chop_lowest = None
    first_step_chop_highest = None
else:
    first_step_mirostat = MIRO
    first_step_length = length
    first_step_temperature = temperature
    first_step_top_k = top_k
    first_step_top_p = top_p
    first_step_middle_p = middle_p
    first_step_chop_lowest = chop_lowest
    first_step_chop_highest = chop_highest

print((MIRO, length, temperature, top_p, middle_p, chop_lowest, chop_highest))
print(
    (
        first_step_mirostat,
        first_step_length,
        first_step_temperature,
        first_step_top_p,
        first_step_middle_p,
        first_step_chop_lowest,
        first_step_chop_highest,
    )
)

max_ctx_fits_on_gpu = 2048 if V11 else 1024

if SELECT_VIA_GENERATOR_LONGLENGTH:
    length_select = max_ctx_fits_on_gpu

if SENTIMENT_VIA_GENERATOR_LONGLENGTH:
    length_sentiment = max_ctx_fits_on_gpu

GPT_NEO_T = 0.95 if BREAKRUNS else 1.0
GPT_NEO_TOP_P = 1.
GPT_NEO_TOP_K = 0
GPT_NEO_MAX_LENGTH = 2048 if V11 else 1024

head_inference_batch_size = 1 if V11 else None

# # V11 loads on cpu initially then transfers to gpu
# TENSOR_LOAD_DEVICE = 'cpu' if V11 else 'cuda:0'

# lazy init should allow this everywhere
# TODO: get rid of this, no longer needed in ultra_defensive_load
TENSOR_LOAD_DEVICE = 'cuda:0'

if GPU_TYPE == "big":
    MODELS_SERVED = {"generator", "selector", "sentiment", "autoreviewer"}
elif V11_INSURANCE:
    MODELS_SERVED = {"selector", "sentiment", "autoreviewer"}
elif V11:
    MODELS_SERVED = {"generator", "selector", "sentiment", "autoreviewer"}
    # MODELS_SERVED = {"generator", }
else:
    # pre-v11
    MODELS_SERVED = {"generator", "selector", "sentiment", "autoreviewer"}

os.chdir(startdir)
