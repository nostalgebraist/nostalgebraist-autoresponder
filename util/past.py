"""information about my particular bot's history and related helpers"""
import pandas as pd

MILESTONE_TIMES = {
    pd.Timestamp("2020-09-30"): "v8_golive",
    pd.Timestamp("2020-10-03"): "miro_golive",  # c34029f old repo
    pd.Timestamp("2020-11-11"): "miro_v2_golive",  # 09b5f4 old repo
    pd.Timestamp("2020-11-22"): "v9_golive",
    pd.Timestamp("2020-12-05"): "v9_1_golive",
    pd.Timestamp("2020-12-07"): "v9_1R2_golive",
    pd.Timestamp("2021-01-15"): "v10_golive",
    pd.Timestamp("2021-03-12 16:10"): "v10_1_golive",
    pd.Timestamp("2021-04-01"): "moderator2_golive",
    pd.Timestamp("2021-04-02 11:10"): "autoreview_golive",
    pd.Timestamp("2021-05-06 21:57"): "v11_golive",
    pd.Timestamp("2021-06-06 17:12"): "v11_2_golive",
    pd.Timestamp("2021-06-13 17:22"): "v12_golive",
    pd.Timestamp("2021-06-14 08:52"): "v12_2_golive",
}

MAJOR_MILESTONE_TIMES = {
    ts: name
    for ts, name in MILESTONE_TIMES.items()
    if any([
        subs in name for subs in ['v8', 'v9', 'v10', 'v11', 'v12']
    ])
}

VERSION_EXPLAINERS = {
    "v7": """
    Switched to 'forumlike' separators instead of Chinese characters.
    """,

    "v8": """
    Modified the 'forumlike' formatting, putting tags and interlocutor names at the start, adding times, etc.
    """,

    "v9": """
    First to use Colossal Rattumb Corpus.

    Used Corpus v0 (6.2e7 tokens) and v8 separators.

    Model size dropped to 774M.

    First success at true TPU training.
    Used TPUv2, so could only fit a batch size of 8 and only with a smaller model.

    Corpus pretrain: batch 8, 1 epoch.
    Transfer: [fill in later]

    Corpus model name: 'autoresponder_v9_experimental_const_lr' @ 7600 steps.
    Transfer model name: 'autoresponder_v9_experimental_nost_transfer_take3'.
    """,

    "v9_1": """
    Used Corpus v1 (1.1e8 tokens) and v8 separators.
    First v9 variant, with shortest Corpus pretrain (2 epochs) among v9 variants.

    Model size 1558M.

    Corpus pretrain: batch 128, 2 epochs.
    Transfer: [fill in later]

    Corpus model name: 'autoresponder_v9_v1_1558M' @ 1620 steps.
    Transfer model name: 'autoresponder_v9_v1_1558M_nost_tuning'.
    """,

    "v9_1R2": """
    Variant of "v9_1" with longer Corpus pretrain: 2 epochs --> 3 epochs.

    1558M, Corpus v1, batch 128, 3 epochs.

    Corpus model name: 'autoresponder_v9_v1_1558M' @ 2550 steps.
    Transfer model name: 'autoresponder_v9_v1_1558M_nost_tuning2'.
    """,
    "v9_1R3": """
    Variant of "v9_1" with longer Corpus pretrain: 2 epochs --> 5.3 epochs.

    1558M, Corpus v1, batch 128, 5.3 epochs.

    Corpus model name: 'autoresponder_v9_v1_1558M' @ 4400 steps.
    Transfer model name: 'autoresponder_v9_v1_1558M_nost_tuning3'.
    """,

    "v9_1R3": """
    Variant of "v9_1."

    Same Corpus pretrain as v9_1R3.

    Different version of transfer dataset ("floornight fix" -- TODO: remember what this means).
    """,
    "v10": """
    Used Corpus v2 (1.3e8 tokens) and v10 separators.

    First version to fix Adam bias bug --> effective summed learning rate raised by a factor of [fill in] over [fill in].

    First version to use validation sets and noise scale.

    1558M, Corpus v2, batch 120, 4 epochs.

    Corpus model name: 'autoresponder_v10' @ 4400 steps.
    Transfer model name: 'autoresponder_v10_nost_tuning_f'.
    """,
    "v11": """TODO (gpt-neo)""",
    "v11_2": """TODO (nost tuning quotes/dedup/etc)""",
    "v12": """TODO (gpt-j)""",
    "v12_2": """TODO (gpt-j) nost tuning""",
}

VERSION_METADATA = {
    "v9": {
        "model_size": "774M",
        "corpus_name": "v0",
        "pretrain_tokens": 61820671,
        "transfer_tokens": 5803579,
        "pretrain_batch_size": 8,
        "pretrain_steps": 7600,
        "transfer_batch_size": 8,
        "transfer_steps": 2400,
        "pretrain_max_lr": 1.5e-5,
        "pretrain_summed_lr": 0.08662222774251596,
        "transfer_max_lr": 4e-6,
        "transfer_summed_lr": 0.005283145101413014,
        "adam_bias_fixed": False,
        "pretrain_model_name": "autoresponder_v9_experimental_const_lr",
        "transfer_model_name": "autoresponder_v9_experimental_nost_transfer_take3",
    },
    "v9_1": {
        "model_size": "1558M",
        "corpus_name": "v1",
        "pretrain_tokens": 108773237,
        "transfer_tokens": 5803579,
        "pretrain_batch_size": 128,
        "pretrain_steps": 1620,
        "transfer_batch_size": 32,
        "transfer_steps": 712,
        "pretrain_max_lr": 2.5e-5,
        "pretrain_summed_lr": 0.019797048284180538,
        "transfer_max_lr": 1e-5,
        "transfer_summed_lr": 0.0038529999999999997,
        "adam_bias_fixed": False,
        "pretrain_model_name": "autoresponder_v9_v1_1558M",
        "transfer_model_name": "autoresponder_v9_v1_1558M_nost_tuning",
    },
    "v9_1R2": {
        "model_size": "1558M",
        "corpus_name": "v1",
        "pretrain_tokens": 108773237,
        "transfer_tokens": 5803579,
        "pretrain_batch_size": 128,
        "pretrain_steps": 2250,
        "transfer_batch_size": 32,
        "transfer_steps": 534,
        "pretrain_max_lr": 2.5e-5,
        "pretrain_summed_lr": 0.02655127465978588,
        "transfer_max_lr": 1e-5,
        "transfer_summed_lr": 0.0034080000000000004,
        "adam_bias_fixed": False,
        "pretrain_model_name": "autoresponder_v9_v1_1558M",
        "transfer_model_name": "autoresponder_v9_v1_1558M_nost_tuning2",
    },
    "v9_1R3": {
        "model_size": "1558M",
        "corpus_name": "v1",
        "pretrain_tokens": 108773237,
        "transfer_tokens": 5803579,
        "pretrain_batch_size": 128,
        "pretrain_steps": 4400,
        "transfer_batch_size": 32,
        "transfer_steps": 534,
        "pretrain_max_lr": 2.5e-5,
        "pretrain_summed_lr": 0.04310071099445527,
        "transfer_max_lr": 6.2e-6,
        "transfer_summed_lr": 0.0025728,
        "adam_bias_fixed": False,
        "pretrain_model_name": "autoresponder_v9_v1_1558M",
        "transfer_model_name": "autoresponder_v9_v1_1558M_nost_tuning3",
    },
    "v9_1R4": {
        "model_size": "1558M",
        "corpus_name": "v1",
        "pretrain_tokens": 108773237,
        "transfer_tokens": 5743071,
        "pretrain_batch_size": 128,
        "pretrain_steps": 4400,
        "transfer_batch_size": 32,
        "transfer_steps": 464,
        "pretrain_max_lr": 2.5e-5,
        "pretrain_summed_lr": 0.04310071099445527,
        "transfer_max_lr": 9.3e-06,
        "transfer_summed_lr": 0.0028661,
        "adam_bias_fixed": False,
        "pretrain_model_name": "autoresponder_v9_v1_1558M",
        "transfer_model_name": "autoresponder_v9_v1_1558M_nost_tuning3",
    },
    "v10": {
        "model_size": "1558M",
        "corpus_name": "v2",
        "pretrain_tokens": 133285930,
        "transfer_tokens": 5290881,
        "pretrain_batch_size": 120,
        "pretrain_steps": 4400,
        "transfer_batch_size": 120,
        "transfer_steps": 135,
        "pretrain_max_lr": 4.76e-05,
        "pretrain_summed_lr": 0.12498540000000001,
        "transfer_max_lr": 1e-5,
        "transfer_summed_lr": 0.0012367000000000003,
        "adam_bias_fixed": True,
        "pretrain_model_name": "autoresponder_v10",
        "transfer_model_name": "autoresponder_v10_nost_tuning_f",
    },
}
