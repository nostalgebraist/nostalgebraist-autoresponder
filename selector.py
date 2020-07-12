"""
Runs the selector, regularly polling the bridge service and handling selection needs.

This was copied out of a jupyter notebook and hasn't been edited much since then, so
code quality is even uglier than usual :(
"""
import numpy as np, pandas as pd
import pickle
import sys
from textwrap import wrap

from copy import deepcopy
from datetime import datetime

import requests, time
from tqdm import tqdm

Q_CHAR = "会"
A_CHAR = "域"
T_CHAR = "职"
ORIG_POST_CHAR = "翰"

AB_TEST_A_SEQUENCE = "\uFFFA"
AB_TEST_B_SEQUENCE = "\uFFFB"

from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_predict, cross_validate
from sklearn.compose import TransformedTargetRegressor

from scipy.special import softmax

from simpletransformers.classification import ClassificationModel, MultiLabelClassificationModel

from bot_config import BotSpecificConstants
from sentiment import SentimentCache

bot_specific_constants = BotSpecificConstants.load()
selector_url = bot_specific_constants.bridge_service_url + "/pollselector"
generator_url = bot_specific_constants.bridge_service_url + "/pollgenerator"

RETENTION_STACK = {}
RESULT_STACK = {}
wrapped = None

MODEL_NAME = "bert_6_25_20_WITH_FIX"
VERIFY_MODEL = False

RETENTION_CUTOFF = 0.6
ENFORCE_RETENTION_CUTOFF = True

SELECT_VIA_GENERATOR = True
EOT_WORKAROUND = True
eot_end_segment = "<|endoftext|>" if EOT_WORKAROUND else "<|"

BERT_CONFIG_DEFAULT = {"model_type": "roberta",
                       "model_name": "roberta-large",
                       "args": {'reprocess_input_data': True, "fp16": False,
                                'train_batch_size': 8, 'eval_batch_size': 8,
                                'gradient_accumulation_steps': 2,
                                'learning_rate': 1e-5,
                                'max_seq_length': 256,
                                'sliding_window': False,
                                'num_train_epochs': 4,
                                'warmup_steps': 0,
                                'warmup_ratio': 0.06,
                                'weight_decay': 0.025,
                                'logging_steps': 0,
                                'max_grad_norm': 10000.,
                                'adam_epsilon': 1e-6,
                                'silent': False,
                                'overwrite_output_dir': True,
                                'evaluate_during_training': False,
                                'use_early_stopping': False,
                                'save_model_every_epoch': False,
                                'save_optimizer_and_scheduler': False,
                                'save_steps': 0,
                                },
                       "kwargs": {"use_cuda": False,
                                  "num_labels": 2}}


class SimpleTransformerClassificationEstimator(BaseEstimator,):
  def __init__(self, model_maker):
    self.model_maker = model_maker
    self.model_ = None
    self.classes_ = None

  def fit(self, X, y):
    train_df = pd.DataFrame({"text": X, "target": y})[["text", "target"]]
    self.model_ = self.model_maker()

    self.model_.train_model(train_df, **self.extra_metrics)

    self.classes_ = [0, 1]
    return self

  def predict(self, X):
    preds, logits = self.model_.predict(X)
    return preds

  def predict_proba(self, X):
    preds, logits = self.model_.predict(X)
    if self.model_.args['sliding_window']:
        logits = [np.mean(l, axis=0) for l in logits]
    proba = softmax(logits, axis=1)
    return proba


def make_bert(output_dir: str, bert_config: dict=BERT_CONFIG_DEFAULT, add_timestamp=False, regression=True, overrides: dict=None):
    bert_config_ = deepcopy(bert_config)

    output_dir_ = output_dir
    if add_timestamp:
      output_dir_ += "_" + datetime.now().strftime("%H-%M-%S")
    bert_config_["args"]["output_dir"] = output_dir_
    bert_config_["args"]["best_model_dir"] = output_dir_ + "/best_model"

    if regression or bert_config['args'].get('multi_label'):
      bert_config_["kwargs"]["num_labels"] = 1
    else:
      bert_config_["kwargs"]["num_labels"] = 2

    if overrides is not None:
      for k, v in overrides.items():
        bert_config_["args"][k] = v

    if regression:
      constructor = RegressionMode
    elif bert_config['args'].get('multi_label'):
      constructor = MultiLabelClassificationModel
      print('using MultiLabelClassificationModel')
    else:
      constructor = ClassificationModel
    bert = constructor(model_type=bert_config_["model_type"],
                            model_name=bert_config_["model_name"],
                            args=bert_config_["args"],
                            **bert_config_["kwargs"])
    return bert

def load_selector_model(model_name, verify=False):
    global wrapped
    bert = ClassificationModel(BERT_CONFIG_DEFAULT["model_type"], model_name, **BERT_CONFIG_DEFAULT["kwargs"])
    # cf https://github.com/ThilinaRajapakse/simpletransformers/issues/103
    bert.args['max_seq_length'] = 256

    wrapped = SimpleTransformerClassificationEstimator(model_maker=lambda:make_bert(model_name, add_timestamp=True, regression=False))
    wrapped.model_ = bert

    if verify:
        verify_new_model()

def load_retention():
    global RETENTION_STACK
    with open("retention_stack.pkl.gz", "rb") as f:
        RETENTION_STACK = pickle.load(f)

def logit_diff(sentiment):
    pos_logit = sentiment["logits"][0] if sentiment is not None else 0
    neg_logit = sentiment["logits"][1] if sentiment is not None else 0
    return pos_logit - neg_logit

def pos_sent(sentiment):
    if sentiment is None:
        return 0.
    return sentiment["prob"] if sentiment["label"] == "1" else 1.-sentiment["prob"]

def show_note_preds(texts, preds):
  for tpe, pred in zip(texts, preds):
    print(f"\tpredicted notes: {pred:.1f}\n")
    print("\n~_~_~_~_~_\n")
    print("\n".join(wrap(tpe)))
    print("\n~_~_~_~_~_\n")

def show_note_probas(texts, probas, continuation_sentiments=None, other_proba=None):
    if continuation_sentiments is None:
        sent_segments = ["" for _ in texts]
    else:
        sent_segments = [f", pos_sent {pos_sent(sent):.1%}" for sent in continuation_sentiments]

    if other_proba is None:
        other_proba_segments = ["" for _ in texts]
    else:
        other_proba_segments = [f", other_proba {p:.1%}" if p is not None else "other_proba None"
                                for p in other_proba]

    for tpe, proba, sseg, opseg in zip(texts, probas, sent_segments, other_proba_segments):
        print(f"\tpredicted prob: {proba:.1%}{opseg}{sseg}\n")
        print("\n~_~_~_~_~_\n")
        print("\n".join(wrap(tpe)))
        print("\n~_~_~_~_~_\n")

def verify_new_model():
    global wrapped
    with open("reward/textpost_examples.pkl.gz", "rb") as f:
        textpost_examples = pickle.load(f)
    textpost_examples = [s.lstrip(ORIG_POST_CHAR) for s in textpost_examples]
    proba_tpe = wrapped.predict_proba(textpost_examples)[:, 1]
    show_note_probas(textpost_examples, proba_tpe)

def parse_continuation(continuation: str, verbose=True):
    if verbose:
        print(f"parsing the following raw output:\n------------------\n{continuation}\n------------------\n")

    # split out tags, if present
    post, _ , tag_text = continuation.partition(T_CHAR)
    tag_text = tag_text.partition(eot_end_segment)[0]  # drop stuff after eot_end_segment
    tag_text = tag_text.partition('<|')[0]  # temporarily support old EOT format

    tags = []
    if len(tag_text) > 0:
        tags = [s.rstrip(" ") for s in tag_text.split("#")]

    # handle mistake i made in AR V6 :(
    if "#original fiction" in post:
        post_after_fic_tag = post[post.index("#original fiction"):]
        if len(post_after_fic_tag.split()) < 10:
            fic_tags = [s.rstrip(" ") for s in post_after_fic_tag.split("#")]
            print(f"converting {post_after_fic_tag} to {fic_tags}")
            tags = fic_tags + tags
            post = post[:post.index("#original fiction")]

    post = post.lstrip(ORIG_POST_CHAR) # TODO: fix this in get_prompted_continuation_with_length_proportional_sampling
    parsed = {"post": post, "tags": tags}
    return parsed


def winndow_probabilities(proba, lower=0.15, upper=0.65):
    proba_ = proba.copy()
    exclusion_mask = np.zeros_like(proba, dtype=bool)

    if (proba_ > upper).any():
        exclude_upper = (proba <= upper)
        print(f"winnowing {exclude_upper.sum()} of {len(proba)} with p<{upper}")
        exclusion_mask[exclude_upper] = True
    elif (proba_ > lower).any():
        exclude_lower = (proba <= lower)
        print(f"winnowing {exclude_lower.sum()} of {len(proba)} with p<{lower}")
        exclusion_mask[exclude_lower] = True

    proba_[exclusion_mask] = 0

    return proba_


def get_continuation_sentiments(continuations, sleep_time=0.2):
    sc = SentimentCache.load()
    continuations_stripped = []
    for c in continuations:
        if T_CHAR in c:
            c_stripped = c.partition(T_CHAR)[0]
        else:
            c_stripped = c
        continuations_stripped.append(c_stripped)

    continuation_sentiments = [sc.query(c, sleep_time=sleep_time) for c in tqdm(continuations_stripped)]
    sc.save()
    return continuation_sentiments


def sentiment_screen(continuations, mood, selection_proba=None):
    if selection_proba is None:
        selection_proba = [None for _ in continuations]

    all_continuation_sentiments = get_continuation_sentiments(continuations)

    score_fn = mood['score_fn']
    if score_fn == "logit_diff":
        scores = np.asarray([logit_diff(sentiment) for sentiment in all_continuation_sentiments])
    elif score_fn == "pos_sentiment":
        scores = np.asarray([pos_sent(sentiment) for sentiment in all_continuation_sentiments])
    else:
        raise ValueError(f"score_fn {score_fn} not understood")

    min_allowed_score = mood["min_allowed_score"]
    max_allowed_score = mood["max_allowed_score"]

    print(f"{score_fn}: {scores}\n")

    exclusion_mask = np.zeros_like(scores, dtype=bool)

    if (scores >= min_allowed_score).any():
        exclude_lower = (scores < min_allowed_score)
        print(f"excluding {exclude_lower.sum()} of {len(scores)} with {score_fn}<{min_allowed_score}")
        exclusion_mask[exclude_lower] = True
    else:
        print(f"couldn't find any with {score_fn}>={min_allowed_score}, highest is {scores.max()}")

    if (scores <= max_allowed_score).any():
        exclude_upper = (scores > max_allowed_score)
        print(f"excluding {exclude_upper.sum()} of {len(scores)} with {score_fn}>{max_allowed_score}")
        exclusion_mask[exclude_upper] = True
    else:
        print(f"couldn't find any with {score_fn}<={max_allowed_score}, lowest is {scores.min()}")

    retained_continuation_sentiments = [sent
                                        for mask, sent in zip(exclusion_mask, all_continuation_sentiments)
                                        if not mask]

    retained_continuations = [cont
                              for mask, cont in zip(exclusion_mask, continuations)
                              if not mask]

    retained_selection_proba = [p for mask, p in zip(exclusion_mask, selection_proba) if not mask]

    return retained_continuations, retained_continuation_sentiments, retained_selection_proba, all_continuation_sentiments


def sentiment_screen_legacy(proba, continuations, mood):
    continuation_sentiments = get_continuation_sentiments(continuations)

    score_fn = mood['score_fn']
    if score_fn == "logit_diff":
        scores = np.asarray([logit_diff(sentiment) for sentiment in continuation_sentiments])
    elif score_fn == "pos_sentiment":
        scores = np.asarray([pos_sent(sentiment) for sentiment in continuation_sentiments])
    else:
        raise ValueError(f"score_fn {score_fn} not understood")

    proba_ = proba.copy()
    exclusion_mask = np.zeros_like(proba, dtype=bool)

    min_allowed_score = mood["min_allowed_score"]
    max_allowed_score = mood["max_allowed_score"]

    print(f"proba: {proba}\nscores: {scores}\n")

    if (scores >= min_allowed_score).any():
        exclude_lower = (scores < min_allowed_score)
        print(f"excluding {exclude_lower.sum()} of {len(proba)} with {score_fn}<{min_allowed_score}")
        exclusion_mask[exclude_lower] = True
    else:
        print(f"couldn't find any with {score_fn}>={min_allowed_score}, highest is {scores.max()}")

    if (scores <= max_allowed_score).any():
        exclude_upper = (scores > max_allowed_score)
        print(f"excluding {exclude_upper.sum()} of {len(proba)} with {score_fn}>{max_allowed_score}")
        exclusion_mask[exclude_upper] = True
    else:
        print(f"couldn't find any with {score_fn}<={max_allowed_score}, lowest is {scores.min()}")

    proba_[exclusion_mask] = 0

    return proba_, continuation_sentiments


def serve_selection(data):
  global RETENTION_STACK
  global RETENTION_STACK_PROBA
  global wrapped
  continuations = data["continuations"]
  selection_proba = data.get("selection_proba")
  if selection_proba is not None:
      print(f"len(selection_proba): {len(selection_proba)} vs len(continuations): {len(continuations)}")
  else:
      print("selection_proba is None")

  kwargs = data["kwargs"]
  mood = kwargs.get("mood")

  strategy = "proportional"
  if "strategy" in kwargs:
    strategy = kwargs['strategy']

  if (data['type'] == 'textpost') and (strategy != "uniform"):
      continuations += sorted(RETENTION_STACK)
      if selection_proba is not None:
          if RETENTION_STACK_PROBA is not None:
              selection_proba += RETENTION_STACK_PROBA
          else:
              selection_proba += [None for _ in RETENTION_STACK]
  base_id = data["base_id"]

  do_mood_screen = False
  if mood is not None:
        do_mood_screen = mood.get("name") != "unrestricted"

  if do_mood_screen:
    continuations, continuation_sentiments, selection_proba, all_continuation_sentiments = sentiment_screen(
            continuations, mood, selection_proba
        )
  else:
        continuation_sentiments = get_continuation_sentiments(continuations)
        all_continuation_sentiments = continuation_sentiments

  if SELECT_VIA_GENERATOR:
      proba = np.asarray(selection_proba)
      show_note_probas(continuations, proba, continuation_sentiments)
  else:
  proba = wrapped.predict_proba([s.lstrip("翰") for s in continuations])[:, 1]
      show_note_probas(continuations, proba, continuation_sentiments, other_proba=selection_proba)

  if strategy == "argmax":
    #choice_ix = preds.argmax()
    choice_ix = proba.argmax()
  elif strategy == "proportional" or strategy == "proportional_winnowed":
    #note_preds = np.exp(preds)-1
    #probs = note_preds / sum(note_preds)

    if strategy == "proportional_winnowed":
        proba_winnowed = winndow_probabilities(proba)
    else:
        proba_winnowed = proba

    probs = proba_winnowed / sum(proba_winnowed)
    print(f"choosing between preds {proba_winnowed}\nprobs {probs}")
    choice_ix = np.random.choice(list(range(len(probs))), p=probs)
  elif strategy == "uniform":
    print("choosing randomly with uniform distribution")
    choice_ix = np.random.choice(list(range(len(continuations))))
  else:
    raise ValueError(f"strategy {strategy}")

  continuation = continuations[choice_ix]
  chosen_proba = proba[choice_ix]
  chosen_pos_sent = pos_sent(continuation_sentiments[choice_ix])
  print(f"\nselecting #{choice_ix} with pred {chosen_proba:.1%}, pos_sent {chosen_pos_sent:.1%}:\n{continuation}\n")

  if data['type'] == 'textpost':
      for i, p in enumerate(proba):
        if p > RETENTION_CUTOFF and continuations[i] not in RETENTION_STACK:
            RETENTION_STACK.add(continuations[i])

      if continuation in RETENTION_STACK:
        RETENTION_STACK.remove(continuation)

  parsed = parse_continuation(continuation)
  parsed["proba"] = float(chosen_proba)
  parsed["pos_sentiment"] = float(chosen_pos_sent)
  parsed["all_pos_sentiment"] = [float(pos_sent(s)) for s in all_continuation_sentiments]

  parsed["base_id"] = base_id

  if 'AB_fork' in kwargs:
    fork = kwargs['AB_fork']
    parsed['AB_fork'] = fork

    post = parsed['post']
    non_newline_ixs = [ix for ix, c in enumerate(post) if c != "\n"]
    if len(non_newline_ixs) > 0:
        newline_switch_ix = max(non_newline_ixs) + 1
        trailing_newlines = post[newline_switch_ix:]
        post = post[:newline_switch_ix]
    else:
        trailing_newlines = ""
  else:
    print(f"not AB testing, have kwargs {kwargs}")

  print(f"sending back: {parsed}")

  with open("retention_stack.pkl.gz", "wb") as f:
    pickle.dump(RETENTION_STACK, f)

  with open("retention_stack_backup.pkl.gz", "wb") as f:
    pickle.dump(RETENTION_STACK, f)

  return parsed


def select_one(data):
    global wrapped
    texts = data['texts']

    proba = wrapped.predict_proba([s.lstrip("翰") for s in texts])[:, 1]

    selection_proba = [float(p) for p in proba]
    results = {"selection_proba": selection_proba}

    print(f"sending back: {results}")
    return results

def apply_retention_cutoff():
    global RETENTION_STACK
    global RETENTION_STACK_PROBA

    n_before_stack, n_before_proba = len(RETENTION_STACK), len(RETENTION_STACK_PROBA)
    retain = [p > RETENTION_CUTOFF for p in RETENTION_STACK_PROBA]

    new_stack = [s for s, r in zip(sorted(RETENTION_STACK), retain) if r]
    new_proba = [p for p, r in zip(RETENTION_STACK_PROBA, retain) if r]

    RETENTION_STACK = set(new_stack)
    RETENTION_STACK_PROBA = new_proba

    n_after_stack, n_after_proba = len(RETENTION_STACK), len(RETENTION_STACK_PROBA)

    all_unchanged = (n_before_stack == n_after_stack) and (n_before_proba == n_after_proba)
    if not all_unchanged:
        print(f"before: {n_before_stack} in RETENTION_STACK, {n_before_proba} in RETENTION_STACK_PROBA")
        print(f"after: {n_after_stack} in RETENTION_STACK, {n_after_proba} in RETENTION_STACK_PROBA")

def poll():
  global RESULT_STACK
  global RETENTION_STACK
  global RETENTION_STACK_PROBA

  r = requests.post(selector_url, json={"results": RESULT_STACK,
                                        "retention_stack": sorted(RETENTION_STACK),})

  received_data = r.json()
  PROMPT_STACK = received_data["SELECTION_PROMPT_STACK"]
  RETENTION_STACK_PROBA = received_data["RETENTION_STACK_PROBA"]

  if ENFORCE_RETENTION_CUTOFF and RETENTION_STACK_PROBA is not None:
      apply_retention_cutoff()

  RESULT_STACK = {k: v for k, v in RESULT_STACK.items() if k in PROMPT_STACK}  # clean out already used results

  print(f"got prompt stack: {PROMPT_STACK}")

  for prompt_id, data in PROMPT_STACK.items():
    print("selecting...")
    if data.get('raw_selection_request', False):
        RESULT_STACK[prompt_id] = select_one(data)
    else:
        RESULT_STACK[prompt_id] = serve_selection(data)

  if len(RESULT_STACK) > 0:
    requests.post(selector_url, json={"results": RESULT_STACK,
                                      "n_retention": len(RETENTION_STACK)})
  print(f"done generating for this poll, {len(RETENTION_STACK)} on RETENTION_STACK")

  if len(PROMPT_STACK) > 0 and not data.get('raw_selection_request', False):
    r = requests.post(generator_url, json={"results": RESULT_STACK})
    time.sleep(1)

import time

def loop_poll(period=60):
  while True:
    try:
      poll()
    except Exception as e:
      print(f"{type(e)}: {e}")
      time.sleep(period*10)
    time.sleep(period)

def selector_main_loop():
    global RESULT_STACK
    global RETENTION_STACK

    load_retention()
    if not SELECT_VIA_GENERATOR:
        load_selector_model(MODEL_NAME, verify=VERIFY_MODEL)

    loop_poll(period=5)

if __name__ == "__main__":
    sys.exit(selector_main_loop())
