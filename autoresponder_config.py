# HPARAMS, FLAGS, ETC
import os
import json
import subprocess

from autoresponder_static import *
from autoresponder_static_v8 import *

V8 = True
V8_2 = True
V9 = True # ! experimental

startdir = os.getcwd()
os.chdir("/")

SELECT_VIA_GENERATOR = True
EOT_WORKAROUND = True
EOT_PREPEND = True
SELECTOR_CAN_SEE_PROMPTS = True
SELECTOR_LR_CALIB_INPUT = "logit_diff"
SELECTOR_LEFT_STRIP_NEWLINE_IN_FORUMLIKE = False

# FORUMLIKE = True  # now set in autoresponder_static
FORUMLIKE_REVIEW_PROB = 0.15
FORUMLIKE_FIC_PROB = 0.15

EVEN_BETTER_LENGTH = True

BATCHONE = True  # ! experimental

eot_end_segment = EOT_FULL if EOT_WORKAROUND else "<|"

if V8:
  final_munge_before_neural = final_munge_before_neural_v8
  final_munge_after_neural = final_munge_after_neural_v8
else:
  final_munge_before_neural = final_munge_before_neural_v7
  final_munge_after_neural = lambda text, *args, **kwargs: text

if V9:
    model_name = "autoresponder_v9_experimental_nost_transfer_take3"

    dataset = "data/data_v9_nost_tuning.npz"
    ckpt_select = "autoresponder_v9_selector/v6/.hdf5"
    ckpt_sentiment = "autoresponder_v9_sentiment/v3/.hdf5"

    TRUNCATE_AT_RIGHT = True
    SELECTOR_EOT_PREPEND = True
elif V8:
    model_name = "autoresponder_v8_opt"

    if V8_2:
      dataset = "data/data_v8_2_recat.npz"
      ckpt_select = "autoresponder_v8_2_selector/v7/.hdf5"
      ckpt_sentiment = "autoresponder_v8_2_sentiment/v1/.hdf5"
    else:
      dataset = "data/data_v8_1_extended.npz"
      ckpt_select = "autoresponder_v8_selector/v9/.hdf5"
      ckpt_sentiment = "autoresponder_v8_sentiment/v1/.hdf5"
    TRUNCATE_AT_RIGHT = True
    SELECTOR_EOT_PREPEND = True
elif FORUMLIKE_V2:
    model_name = "autoresponder_v7_2"
    dataset = "data/data_with_fixes_retune.npz"

    ckpt_select = "autoresponder_v7_2_selector/v23/.hdf5"
    ckpt_sentiment = "autoresponder_v7_2_sentiment/v2/.hdf5"
    TRUNCATE_AT_RIGHT = True
    SELECTOR_EOT_PREPEND = False
elif FORUMLIKE:
    model_name = "autoresponder_v7"
    dataset = "data/autoresponder_v7_1/data_with_reviews.npz"

    ckpt_select = "autoresponder_v7_selector/v6/.hdf5"
    ckpt_sentiment = "autoresponder_v7_sentiment/v4/.hdf5"
    TRUNCATE_AT_RIGHT = True
    SELECTOR_EOT_PREPEND = False
    GLOBAL_DEBUG = False
else:
    model_name = "autoresponder_v6_normed_prime"
    dataset = "data/autoresponder_v6_normed_prime/data.npz"

    ckpt_select = "autoresponder_v6_normed_prime_selector/v2_7/.hdf5"
    ckpt_sentiment = "autoresponder_v6_normed_prime_sentiment/v1/.hdf5"
    SELECTOR_EOT_PREPEND = False
    TRUNCATE_AT_RIGHT = False

with open(ckpt_select.rpartition("/")[0] + "/metadata.json", "r") as f:
  select_metadata = json.load(f)

select_scope = select_metadata['select_scope']

if V9:
    layer_nums = [7, 23]
    do_resid = False
    norm_layers_after = False
    use_mlp = True
    resid_mlp = True
    direct_mlp = False
    mlp_proj = True
    mlp_ratio = 2
    use_length_channel = False
    use_length_channel_v2 = False
    add_position_emb_later_layers = False
    add_prompt_cont_embs = False
    norm_final_output = False
    n_head_select = select_metadata['n_head']
    length_select = 825
    SELECT_VIA_GENERATOR_LONGLENGTH = True
    MULTI_LR_CALIB = True
elif V8:
    layer_nums = [7, 23]
    do_resid = False
    norm_layers_after = False
    use_mlp = True
    resid_mlp = True
    direct_mlp = False
    mlp_proj = True
    mlp_ratio = 2
    use_length_channel = False
    use_length_channel_v2 = False
    add_position_emb_later_layers = False
    add_prompt_cont_embs = False
    norm_final_output = False
    n_head_select = select_metadata['n_head']
    length_select = 825
    SELECT_VIA_GENERATOR_LONGLENGTH = True
    MULTI_LR_CALIB = True
else:
    layer_nums = [7, 23]
    do_resid = False
    norm_layers_after = False
    use_mlp = True
    resid_mlp = True
    direct_mlp = False
    mlp_proj = True
    mlp_ratio = 2
    use_length_channel = False
    use_length_channel_v2 = False
    add_position_emb_later_layers = False
    add_prompt_cont_embs = False
    norm_final_output = False
    n_head_select = 40
    length_select = 825
    SELECT_VIA_GENERATOR_LONGLENGTH = True
    MULTI_LR_CALIB = True

SENTIMENT_VIA_GENERATOR = True
SENTIMENT_VIA_GENERATOR_LONGLENGTH = True

with open(ckpt_sentiment.rpartition("/")[0] + "/metadata.json", "r") as f:
  sentiment_select_metadata = json.load(f)

sentiment_select_scope = sentiment_select_metadata['select_scope']

if V9:
  layer_nums_sentiment = [7, 23]
  use_mlp_sentiment = True
  use_length_channel_sentiment = False
  use_length_channel_v2_sentiment = False
  length_sentiment = 204
  norm_final_output_sentiment = False
  n_head_sentiment = sentiment_select_metadata['n_head']
elif V8:
  layer_nums_sentiment = [8-1, 24-1]
  use_mlp_sentiment = True
  use_length_channel_sentiment = False
  use_length_channel_v2_sentiment = False
  length_sentiment = 204
  norm_final_output_sentiment = False
  n_head_sentiment = sentiment_select_metadata.get('n_head', 25)
else:
  layer_nums_sentiment = [8-1, 24-1]
  use_mlp_sentiment = False
  use_length_channel_sentiment = False
  use_length_channel_v2_sentiment = False
  length_sentiment = 204
  norm_final_output_sentiment = True
  n_head_sentiment = 25

def _gpu_type():
  gpu_info = subprocess.check_output("nvidia-smi").decode('utf-8')
  if "P100" in gpu_info:
    return "big"
  else:
    return "small"

if BATCHONE:
  batch_size = 1

  if _gpu_type() == "big":
    max_ctx_fits_on_gpu=1020
  else:
    max_ctx_fits_on_gpu=1020

else:
  batch_size = 2

  if _gpu_type() == "big":
    max_ctx_fits_on_gpu=940  #825
  else:
    max_ctx_fits_on_gpu=840  #1020  #800


# sets max context size, for long prompts we want to cut off to allow bot to write at least this many tokens
required_continuation_room = 100

if EVEN_BETTER_LENGTH:
    better_length = False
    length = required_continuation_room
    MAX_CONTINUE_TOKENS=2210 if _gpu_type() == "big" else 1600

    MAX_CONTINUE_STEPS=100  # disable via huge
else:
    better_length = True
    length=max_ctx_fits_on_gpu
    MAX_CONTINUE_TOKENS=12*1024  # disable via huge
    MAX_CONTINUE_STEPS=12

MIRO = True
MIRO_V2 = True
MIRO_ONLY_ON_CONTINUE = True

EXPERIMENTAL_TOP_P = True
EXPERIMENTAL_TOP_P_2 = True
EXPERIMENTAL_TOP_P_3 = True
EXPERIMENTAL_MIDDLE_P_TWEAK = False
EXPERIMENTAL_MIDDLE_P_ASYM = True

chop_lowest, chop_highest = None, None

# MIRO_TRUNC = 50000
MIRO_TRUNC = 2000  # ! experimental

# MIRO_LR = 0.2
# MIRO_LR = 0.1  # ! experimental
MIRO_LR = 0.05  # ! experimental

if V8_2:
  MIRO_TARGET_LOW = 2.0
  MIRO_TARGET_HIGH = 2.8
  MIRO_TARGET_ALL = [
    1.8,
    1.9, 1.9,
    2.0, 2.0,
    2.1, 2.1, 2.1,
    2.2, 2.2, 2.2,
    2.3, 2.3, 2.3,
    2.4, 2.4,
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
    2.0, 2.0,
    2.1, 2.1, 2.1, 2.1,
    2.2, 2.2, 2.2,
    2.3, 2.3,
    2.4, 2.4,
    2.5,
    2.6,
    2.7,
    2.8, 2.8,
    ]

if MIRO:
    temperature=1
    top_k=0
    top_p=0
    middle_p=0
elif EXPERIMENTAL_TOP_P:
  if EXPERIMENTAL_TOP_P_3:
    temperature=1
    top_k=0
    top_p=0.925
    middle_p=0
  elif EXPERIMENTAL_TOP_P_2:
    temperature=1
    top_k=0
    top_p=0.9
    middle_p=0
  else:
    temperature=0.95
    top_k=0
    top_p=0.925
    middle_p=0
elif EXPERIMENTAL_MIDDLE_P_ASYM:
  temperature=1
  top_k=0
  top_p=0.
  middle_p=0.
  chop_lowest=(1-0.925)  # chop_lowest 1-x = top_p x
  chop_highest=0.03
elif EXPERIMENTAL_MIDDLE_P_TWEAK:
  temperature=0.95
  top_k=0
  top_p=0
  middle_p=0.925
else:
  temperature=0.95
  top_k=0
  top_p=0
  middle_p=0.85

if MIRO_ONLY_ON_CONTINUE:
  pre_continue_mirostat=False
  pre_continue_length=50
  pre_continue_temperature=1
  pre_continue_top_k=0
  pre_continue_top_p=0.925
  pre_continue_middle_p=0
  pre_continue_chop_lowest=None
  pre_continue_chop_highest=None
else:
  pre_continue_mirostat=MIRO
  pre_continue_length=length
  pre_continue_temperature=temperature
  pre_continue_top_k=top_k
  pre_continue_top_p=top_p
  pre_continue_middle_p=middle_p
  pre_continue_chop_lowest=chop_lowest
  pre_continue_chop_highest=chop_highest

print((MIRO, length, temperature, top_p, middle_p, chop_lowest, chop_highest))
print((pre_continue_mirostat, pre_continue_length, pre_continue_temperature,
       pre_continue_top_p, pre_continue_middle_p,
       pre_continue_chop_lowest, pre_continue_chop_highest))

if SELECT_VIA_GENERATOR_LONGLENGTH:
  length_select = max_ctx_fits_on_gpu

if SENTIMENT_VIA_GENERATOR_LONGLENGTH:
  length_sentiment = max_ctx_fits_on_gpu

os.chdir(startdir)
