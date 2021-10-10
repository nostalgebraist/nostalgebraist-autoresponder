# HPARAMS, FLAGS, ETC
# TODO: refactor this terrible, terrible file
import subprocess

from tumblr_to_text.classic.autoresponder_static_v8 import *

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
V11_2 = True  # nost tuning: spacefix + quotes + dedup
V12 = True  # gpt-j
V12_2 = True  # gpt-j nost tuning
V12_3 = True  # higher lr
V12_4 = True  # fixed lr schedule for gpt-j + skip nost tuning
V12_5 = True  # many incremental improvements to gpt-j lr / dataset / etc + fixed "Posts by"
V12_6 = True  # fix for issue in https://github.com/EleutherAI/gpt-neo/pull/230 + batch size 32
V12_7 = True  # XXXX
V12_8 = True  # XXXX
V12_9 = True  # XXXX
V12_10 = True  # XXXX
V12_11 = True  # XXXX
V12_12 = True
V12_13 = False


BUCKET_NAME = ""
if not V12_7:
    # before switch to HF CDN
    import config.bot_config_singleton
    bot_specific_constants = config.bot_config_singleton.bot_specific_constants
    BUCKET_NAME = bot_specific_constants.BUCKET_NAME


USE_AUTOREVIEWER = True

LOGGING_FLAGS = {
    "side_judg_inputs": False,
    "parse_continuation": False
}

if V12_13:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.,  # v12_13/v1: predict true accept rate: ~XX%, false accept rate ~6.7%
        "reject_above": 0.,  # v12_13/v1: predict true reject rate: ~XX%, false reject rate ~6%
    }
elif V12_12:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.149,  # v12_12/v1: predict true accept rate: ~40%, false accept rate ~6.7%
        "reject_above": 0.558,  # v12_12/v1: predict true reject rate: ~49%, false reject rate ~6%
    }
elif V12_11:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.146,  # v12_11/v2: predict true accept rate: ~40%, false accept rate ~6.7%
        "reject_above": 0.599,  # v12_11/v2: predict true reject rate: ~46%, false reject rate ~6%
    }
elif V12_10:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.141,  # v12_10/v1: predict true accept rate: ~35%, false accept rate ~6.7%
        "reject_above": 0.622,  # v12_10/v1: predict true reject rate: ~47%, false reject rate ~6%
    }
elif V12_9:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.146,  # v12_9/v1: predict true accept rate: ~38%, false accept rate ~6.7%
        "reject_above": 0.587,  # v12_9/v1: predict true reject rate: ~46%, false reject rate ~6%
    }
elif V12_8:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.128,  # v12_8/v1: predict true accept rate: ~29%, false accept rate ~6.7%
        "reject_above": 0.579,  # v12_8/v1: predict true reject rate: ~43%, false reject rate ~6%
    }
elif V12_7:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.113,  # v12_7/v1: predict true accept rate: ~34%, false accept rate ~6.7%
        "reject_above": 0.561,  # v12_7/v1: predict true reject rate: ~45%, false reject rate ~6%
    }
elif V12_6:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.136,  # v12_6/v1: predict true accept rate: ~39%, false accept rate ~6.7%
        "reject_above": 0.619,  # v12_6/v1: predict true reject rate: ~47%, false reject rate ~6%
    }
elif V12_5:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.121,  # v12_5/v2: predict true accept rate: ~36%, false accept rate ~6.7%
        "reject_above": 0.604,  # v12_5/v2: predict true reject rate: ~47%, false reject rate ~6%
    }
elif V12_4:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.130,  # v12_4/v1: predict true accept rate: ~40%, false accept rate ~6.7%
        "reject_above": 0.561,  # v12_4/v1: predict true reject rate: ~47%, false reject rate ~6%
    }
elif V12_3:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.097,  # v12_3/v1: predict true accept rate: ~41%, false accept rate ~6.7%
        "reject_above": 0.606,  # v12_3/v1: predict true reject rate: ~41%, false reject rate ~6%
    }
elif V12_2:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.115,  # v12_2/v3: predict true accept rate: ~38%, false accept rate ~6.7%
        "reject_above": 0.603,  # v12_2/v3: predict true reject rate: ~44%, false reject rate ~6%
    }
elif V12:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.139,  # v12/v3: predict true accept rate: ~43%, false accept rate ~6.7%
        "reject_above": 0.546,  # v12/v3: predict true reject rate: ~50%, false reject rate ~6%
    }
elif V11_2:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.150,  # v11_2/v1: predict true accept rate: ~36%, false accept rate ~6.7%
        "reject_above": 0.481,  # v11_2/v1: predict true reject rate: ~48%, false reject rate ~6%
    }
elif V11:
    AUTOREVIEWER_CUTOFFS = {
        "accept_below": 0.134,  # v11/v4: predict true accept rate: ~33%, false accept rate ~6.7%
        "reject_above": 0.599,  # v11/v4: predict true reject rate: ~47%, false reject rate ~6%
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

startdir = os.getcwd()
os.chdir("/")

SELECTOR_LR_CALIB_INPUT = "logit_diff"
SELECTOR_LEFT_STRIP_NEWLINE_IN_FORUMLIKE = False

FORUMLIKE_REVIEW_PROB = 0.15
FORUMLIKE_FIC_PROB = 0.15

EVEN_BETTER_LENGTH = True

BATCHONE = True

RANDOM_SAMPLING_PARAMS_ON_STARTUP = False

HF_REPO_NAME = "nostalgebraist/nostalgebraist-autoresponder-6_1b"
HF_FILES_GZIPPED = True
model_path = None

if V12_13:
    HF_FILES_GZIPPED = False
    HF_REPO_NAME = "nostalgebraist/nostalgebraist-autoresponder-6_1b-staging"
    model_name = "arj-x7-1213"
elif V12_12:
    HF_FILES_GZIPPED = False
    HF_REPO_NAME = "nostalgebraist/nostalgebraist-autoresponder-6_1b"
    model_name = "arj-x5-no-nbar-1094"
elif V12_11:
    HF_FILES_GZIPPED = False
    HF_REPO_NAME = "nostalgebraist/nostalgebraist-autoresponder-6_1b-staging"
    model_name = "arj-x3-twplus2-alldata-988"
elif V12_10:
    model_name = "arj-x3-twplus-alldata-2051"
elif V12_9:
    HF_REPO_NAME = "nostalgebraist/nostalgebraist-autoresponder-6_1b-staging"
    model_name = "arj-x3-alldata-4385"
elif V12_8:
    model_name = "arj-x2-tw-repack-alldata-3801"
elif V12_7:
    HF_REPO_NAME = "nostalgebraist/nostalgebraist-autoresponder-6_1b-staging"
    model_name = "arj-x2-tw-scaled-alldata-3801"
elif V12_6:
    model_name = "arj-v10-3-batch32-alldata-4001"
elif V12_5:
    model_name = "arj-merged-minu-shuf-alldata-2001"
elif V12_4:
    model_name = "arj-v0-ostate"
elif V12_3:
    model_name = "nost-tuning-arj-cl"
elif V12_2:
    model_name = "arj-v0-nost-tuning-f16"
elif V12:
    model_name = "arj-v0-2801-f16"
elif V11_2:
    model_name = "neo_ar_2_7B_v0_nost_tuning_quotes_dedup_f"
elif V11:
    model_name = "neo_ar_2_7B_v0_nost_tuning_f"
elif V10_1_torch:
    model_name = "torch__autoresponder_v10_1"
elif V10_1:
    model_name = "autoresponder_v10_1"
    model_path = os.path.join("models", model_name, "model-141.hdf5")
else:
    model_name = "autoresponder_v10"
    model_path = os.path.join("models", model_name, "model-135.hdf5")

if not model_path:
    model_path = os.path.join("/", model_name)

if V12_13:
    ckpt_select = "selector/v12_13/v1/"
    ckpt_sentiment = "sentiment/v12_13/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v12_13/v1/"
elif V12_12:
    ckpt_select = "selector/v12_12/v2/"
    ckpt_sentiment = "sentiment/v12_12/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v12_12/v1/"
elif V12_11:
    ckpt_select = "selector/v12_11/v3/"
    ckpt_sentiment = "sentiment/v12_11/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v12_11/v2/"
elif V12_10:
    ckpt_select = "selector/v12_10/v1/"
    ckpt_sentiment = "sentiment/v12_10/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v12_10/v1/"
elif V12_9:
    ckpt_select = "selector/v12_9/v1/"
    ckpt_sentiment = "sentiment/v12_9/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v12_9/v1/"
elif V12_8:
    ckpt_select = "selector/v12_8/v1/"
    ckpt_sentiment = "sentiment/v12_8/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v12_8/v1/"
elif V12_7:
    ckpt_select = "selector/v12_7/v2/"
    ckpt_sentiment = "sentiment/v12_7/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v12_7/v1/"
elif V12_6:
    ckpt_select = "selector/v12_6/v4__layer9/"
    ckpt_sentiment = "sentiment/v12_6/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v12_6/v1/"
elif V12_5:
    ckpt_select = "selector/v12_5/v2/"
    ckpt_sentiment = "sentiment/v12_5/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v12_5/v2/"
elif V12_4:
    ckpt_select = "selector/v12_4/v1/"
    ckpt_sentiment = "sentiment/v12_4/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v12_4/v1/"
elif V12_3:
    ckpt_select = "selector/v12_3/v1/"
    ckpt_sentiment = "sentiment/v12_3/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v12_3/v1/"
elif V12_2:
    ckpt_select = "selector/v12_2/v2/"
    ckpt_sentiment = "sentiment/v12_2/v1/"
    ckpt_autoreviewer = "draft_autoreviewer/v12_2/v3/"
elif V12:
    ckpt_select = "selector/v12/v3/"
    ckpt_sentiment = "sentiment/v12/v2/"
    ckpt_autoreviewer = "draft_autoreviewer/v12/v3/"
elif V11_2:
    ckpt_select = "selector/v11_2/v2/"
    ckpt_sentiment = "sentiment/v11_2/v2/"
    ckpt_autoreviewer = "draft_autoreviewer/v11_2/v1/"
elif V11:
    ckpt_select = "selector/v11/v7/"
    ckpt_sentiment = "sentiment/v11/v2/"
    ckpt_autoreviewer = "draft_autoreviewer/v11/v4/"
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

gs_command_get_encoder = f"gsutil -m cp gs://{BUCKET_NAME}/checkpoint_gs_sync/autoresponder_v10_nost_tuning_f/encoder.json /models/{model_name}/"
gs_command_get_encoder += f"; gsutil -m cp gs://{BUCKET_NAME}/checkpoint_gs_sync/autoresponder_v10_nost_tuning_f/vocab.bpe /models/{model_name}/"

if V12:
    suffix = "_6" if V12_6 else ("_5" if V12_5 else ("_4" if V12_4 else ("_3" if V12_3 else ("_2" if V12_2 else ""))))

    gs_command_get_model = f"gsutil -m cp gs://{BUCKET_NAME}/gpt-j-th/{model_name}/* {model_path}"

    gs_command_get_selector = (
        f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v12{suffix}_selector/* /selector/v12{suffix}/"
    )
    gs_command_get_sentiment = f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v12{suffix}_sentiment/* /sentiment/v12{suffix}/"
    gs_command_get_autoreviewer = f"gsutil -m cp -R gs://{BUCKET_NAME}/draft_autoreviewer/v12{suffix}/* /draft_autoreviewer/v12{suffix}/"
elif V11_2:
    gs_command_get_model = f"gsutil -m cp gs://{BUCKET_NAME}/tf_to_torch/neo_ar_2_7B_v0_nost_tuning_quotes_dedup_f/* {model_path}"

    gs_command_get_selector = (
        f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v11_2_selector/* /selector/v11_2/"
    )
    gs_command_get_sentiment = f"gsutil -m cp -R gs://{BUCKET_NAME}/ar_model_v10/v11_2_sentiment/* /sentiment/v11_2/"
    gs_command_get_autoreviewer = f"gsutil -m cp -R gs://{BUCKET_NAME}/draft_autoreviewer/v11_2/* /draft_autoreviewer/v11_2/"
elif V11:
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
    if "A100" in gpu_info:
        return "bigger"
    elif "P100" in gpu_info or "V100" in gpu_info:
        return "big"
    else:
        return "small"


GPU_TYPE = _gpu_type()

batch_size = 1

max_ctx_fits_on_gpu = 1020

# sets max context size, for long prompts we want to cut off to allow bot to write at least this many tokens
required_continuation_room = 256  # if GPU_TYPE == "big" else 512

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

temperature = 0.95
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

max_ctx_fits_on_gpu = 2048 if V11 else 1024

if SELECT_VIA_GENERATOR_LONGLENGTH:
    length_select = max_ctx_fits_on_gpu

if SENTIMENT_VIA_GENERATOR_LONGLENGTH:
    length_sentiment = max_ctx_fits_on_gpu

# TODO: collapse these w/ "regular" T, top_p, etc
# (wow this file needs cleanup badly...)
GPT_NEO_T = 1.
GPT_NEO_TOP_P = 0.95
GPT_NEO_TOP_K = 0
GPT_NEO_MAX_LENGTH = 2048 if V11 else 1024

if V11 and (GPU_TYPE == "small"):
    # TODO: figure out if this OOM happened due to something in transformers 4.6.0
    #
    # https://github.com/huggingface/transformers/compare/v4.5.1...v4.6.0
    GPT_NEO_MAX_LENGTH = 1900

if V12 and (GPU_TYPE == "small"):
    GPT_NEO_MAX_LENGTH = 1536

head_inference_batch_size = 8 if GPU_TYPE == "bigger" else 1
head_load_device = 'cuda:0' if GPU_TYPE == "bigger" else 'cpu'

MODELS_SERVED = {"generator", "selector", "sentiment", "autoreviewer"}

os.chdir(startdir)
