"""
Coordinating switchboard that handles communication between the generator, selector, and
tumblr API compontents.
"""
import json
import os
import numpy as np
import re
import uuid

from flask import Flask, escape, request, jsonify

from bot_config import BotSpecificConstants
from mood import get_mood_by_name, load_logit_diff_sample, estimate_expected_rejections
from bridge_shared import bridge_service_unique_id

bot_specific_constants = BotSpecificConstants.load()
bridge_service_url = bot_specific_constants.bridge_service_url

logit_diff_sample_series = load_logit_diff_sample()
EXPECTED_REJECTION_MULT = 0.5
TEXTPOST_N_CANDIDATES_TARGET = 20

Q_CHAR = "会"
A_CHAR = "域"
T_CHAR = "职"

UNAME_CHAR = "友"
ORIG_POST_CHAR = "翰"

RAW_SELECT_VIA_GENERATOR = True
RETENTION_PROBA_VIA_GENERATOR = True

AB_TEST_SELECTOR = True
AB_TEST_A_SEQUENCE = "\uFFF9"
AB_TEST_B_SEQUENCE = "\uFFF9\uFFFA\uFFFB"

PROMPT_STACK = {}
RESULT_STACK = {}

SELECTION_PROMPT_STACK = {}
GENERATION_RESULT_STACK = {}

RETENTION_STACK = None
N_RETENTION = None
RETENTION_PROBA_ID = None

GENERATIONS_PER_REQUEST = 3

### FLASK
app = Flask(__name__)

def make_raw_select(texts, new_id, v8_timestamps=None):
    global PROMPT_STACK
    global SELECTION_PROMPT_STACK

    if RAW_SELECT_VIA_GENERATOR:
        PROMPT_STACK[new_id] = {"texts": texts, "v8_timestamps": v8_timestamps, "type": "raw_select"}
        print(f"made raw_select: {PROMPT_STACK[new_id]}")
    else:
        SELECTION_PROMPT_STACK[new_id] = {"texts": texts, "raw_selection_request": True}


@app.route('/raw_select', methods=['POST'])
def raw_select():
    global PROMPT_STACK
    global SELECTION_PROMPT_STACK

    texts = request.json["texts"]
    new_id = request.json["id"]
    v8_timestamps = request.json.get("v8_timestamps", None)

    make_raw_select(texts, new_id, v8_timestamps)

    return jsonify({"collision": False})


@app.route('/', methods=['POST'])
def answer():
    global PROMPT_STACK

    prompt = request.form["question"]
    new_id = request.form["id"]
    mood = request.form.get("mood")
    exact_prompt = request.form.get("exact_prompt", False)
    v8_timestamp = request.form.get("v8_timestamp", "")
    forced_tags_string = request.form.get("forced_tags_string", "")
    write_fic_override = bool(int(request.form.get("write_fic_override", 0)))
    return_all_conts = bool(int(request.form.get("return_all_conts", False)))
    selector_cut_to_final_exchange = bool(int(request.form.get("selector_cut_to_final_exchange", False)))

    if not exact_prompt:
        prompt = UNAME_CHAR + request.form["asking_name"] + Q_CHAR + "\n" + prompt + "\n" + A_CHAR
    elif (not exact_prompt):
        prompt = Q_CHAR + prompt + "\n" + A_CHAR
    print(f'got prompt: {prompt}')

    kwargs = {"best_of": 13, "verbose": True, "V5": True,
              "mood": get_mood_by_name(mood),
              "return_all_conts": return_all_conts,
              "selector_cut_to_final_exchange": selector_cut_to_final_exchange,
              "forced_tags_string": forced_tags_string,
              "write_fic_override": write_fic_override}

    if kwargs["write_fic_override"]:
        kwargs["best_of"] = 8

    expected_rejection_frac = estimate_expected_rejections(
        min_logit_diff=kwargs['mood']['min_allowed_score'],
        max_logit_diff=kwargs['mood']['max_allowed_score'],
        logit_diff_sample_series=logit_diff_sample_series
        )

    raw_extra_best_of = int(np.round(kwargs["best_of"]/(1-expected_rejection_frac))) - kwargs["best_of"]
    discounted_extra_best_of = int(np.round(raw_extra_best_of * EXPECTED_REJECTION_MULT))

    print(f"expecting to reject {expected_rejection_frac:.1%}, need {raw_extra_best_of} extra over best_of={kwargs['best_of']}")
    kwargs['best_of'] += discounted_extra_best_of
    print(f"discounting to {discounted_extra_best_of} --> best_of={kwargs['best_of']}")

    if any([d.get("base_id") == new_id for d in PROMPT_STACK.values()]):
        return jsonify({"collision": True})
    kwargs["strategy"] = "proportional_winnowed"
    kwargs["avoid_if_under"] = 10
    kwargs["avoid_half_if_under"] = 15
    kwargs["avoid_if_cut_off"] = False
    kwargs["split_on_control_char"] = True
    kwargs["avoid_initial_blockquote"] = True
    kwargs["avoid_if_profane"] = False
    if request.form["asking_name"] == "bukbot":
        kwargs["avoid_if_profane"] = True
    if True:
            fork = "B" if np.random.rand() > 1 else "A"
        if fork == "A":
            strategy = "proportional_winnowed"
        elif fork == "B":
            strategy = "uniform"
        else:
            print(f"!!fork {fork} not understood".upper())
            strategy = "proportional"
        kwargs["strategy"] = strategy
        kwargs["AB_fork"] = fork
    generation_id = str(uuid.uuid4())
    PROMPT_STACK[generation_id] = {"type": "answer", "prompt": prompt, "kwargs": kwargs, "base_id": new_id, "v8_timestamp": v8_timestamp}
    PROMPT_STACK[generation_id]["n_desired"] = PROMPT_STACK[generation_id]["kwargs"]["best_of"]
    PROMPT_STACK[generation_id]["kwargs"]["best_of"] = GENERATIONS_PER_REQUEST
    print(f"desiring {PROMPT_STACK[generation_id]['n_desired']}, per request {PROMPT_STACK[generation_id]['kwargs']['best_of']}")
    return jsonify({"collision": False})

@app.route('/textpost', methods=['POST'])
def textpost():
    global PROMPT_STACK
    global RETENTION_STACK
    global N_RETENTION
    global RETENTION_PROBA_ID

    new_id = request.form["id"]
    mood = request.form.get("mood")
    v8_timestamp = request.form.get("v8_timestamp", "")
    return_all_conts = bool(int(request.form.get("return_all_conts", False)))

    kwargs = {"best_of": 10,
              "prompt_from_dataset": True, "verbose": True, "V5": True,
              "mood": get_mood_by_name(mood),
              "return_all_conts": return_all_conts}

    if any([d.get("base_id") == new_id for d in PROMPT_STACK.values()]):
        return jsonify({"collision": True})
    kwargs["strategy"] = "proportional"
    kwargs["avoid_if_under"] = 20
    kwargs["avoid_half_if_under"] = 40
    kwargs["avoid_if_cut_off"] = False
    kwargs["avoid_initial_blockquote"] = True
    if True:
        fork = "B" if np.random.rand() > 1 else "A"
        strategy = "proportional_winnowed" if fork == "A" else "uniform"
        kwargs["strategy"] = strategy
        kwargs["AB_fork"] = fork

    n_candidates_target = TEXTPOST_N_CANDIDATES_TARGET

    # TODO: DRY
    expected_rejection_frac = estimate_expected_rejections(
        min_logit_diff=kwargs['mood']['min_allowed_score'],
        max_logit_diff=kwargs['mood']['max_allowed_score'],
        logit_diff_sample_series=logit_diff_sample_series
        )

    raw_extra_best_of = int(np.round(n_candidates_target/(1-expected_rejection_frac))) - n_candidates_target
    discounted_extra_best_of = int(np.round(raw_extra_best_of * EXPECTED_REJECTION_MULT))

    print(f"expecting to reject {expected_rejection_frac:.1%}, need {raw_extra_best_of} extra over n_candidates_target={n_candidates_target}")
    n_candidates_target += discounted_extra_best_of
    print(f"discounting to {discounted_extra_best_of} --> n_candidates_target={n_candidates_target}")

    if N_RETENTION is not None:
        n_candidates_target = max(0, n_candidates_target - N_RETENTION)
        print(f"with {N_RETENTION} on stack, only need {n_candidates_target}")

        if RETENTION_STACK is not None and RETENTION_PROBA_VIA_GENERATOR:
                url = "http://0.0.0.0:5000/raw_select"
            data = {"texts": RETENTION_STACK}
            new_retention_proba_id = bridge_service_unique_id(url, data)

            if new_retention_proba_id != RETENTION_PROBA_ID:
                make_raw_select(data["texts"], new_retention_proba_id)
                if RETENTION_PROBA_ID in RESULT_STACK:
                    RESULT_STACK.pop(RETENTION_PROBA_ID)
                RETENTION_PROBA_ID = new_retention_proba_id

    kwargs["best_of"] = n_candidates_target

    print(f"AB test: fork {fork}, N_RETENTION {N_RETENTION}, kwargs {kwargs}")

    generation_id = str(uuid.uuid4())
    PROMPT_STACK[generation_id] = {"type": "textpost", "kwargs": kwargs, "base_id": new_id, "v8_timestamp": v8_timestamp}
    PROMPT_STACK[generation_id]["n_desired"] = PROMPT_STACK[generation_id]["kwargs"]["best_of"]
    PROMPT_STACK[generation_id]["kwargs"]["best_of"] = GENERATIONS_PER_REQUEST
    print(f"desiring {PROMPT_STACK[generation_id]['n_desired']}, per request {PROMPT_STACK[generation_id]['kwargs']['best_of']}")
    return jsonify({"collision": False})

@app.route('/rawcont', methods=['POST'])
def raw_cont():
    global PROMPT_STACK
    new_id = request.form["id"]
    prompt = request.form["prompt"]
    best_of = int(request.form["best_of"])
    prompt_from_dataset = request.form.get("prompt_from_dataset", False)

    kwargs = {"best_of": best_of, "prompt_from_dataset": prompt_from_dataset, "verbose": True, }

    if new_id in PROMPT_STACK:
            return jsonify({"collision": True})

    PROMPT_STACK[new_id] = {"type": "raw_continuations", "prompt": prompt, "kwargs": kwargs}
    PROMPT_STACK[generation_id]["n_desired"] = PROMPT_STACK[generation_id]["kwargs"]["best_of"]
    PROMPT_STACK[generation_id]["kwargs"]["best_of"] = GENERATIONS_PER_REQUEST

    return jsonify({"collision": False})


@app.route('/pollgenerator', methods=['POST'])
def pollgenerator():
    global PROMPT_STACK
    global GENERATION_RESULT_STACK
    global RESULT_STACK

    print(request.json)
    posted_results = request.json["results"]

    handled_selection_ids = set()

    for result_id, result in posted_results.items():
        if result_id in PROMPT_STACK:
            if PROMPT_STACK[result_id].get("type") == "raw_select":
                RESULT_STACK[result_id] = result
                handled_selection_ids.add(result_id)
                continue
            elif PROMPT_STACK[result_id].get("type") != "raw_continuations":
                if result_id not in GENERATION_RESULT_STACK:
                    GENERATION_RESULT_STACK[result_id] = result
                else:
                    if all([new_ in GENERATION_RESULT_STACK[result_id].get('continuations', [])
                            for new_ in result.get('continuations', [])]):
                        print("duplicate detected, skipping")
                    else:
                        for list_key in ['continuations', 'selection_proba', 'sentiment_logit_diffs', 'mirotarg']:
                            if list_key not in GENERATION_RESULT_STACK[result_id]:
                                GENERATION_RESULT_STACK[result_id][list_key] = []

                            GENERATION_RESULT_STACK[result_id][list_key].extend(result.get(list_key, []))
                GENERATION_RESULT_STACK[result_id]["done"] = False
            else:
                # TODO: move this over to be like the above block, if we ever plan to use "raw_continuations" again
                RESULT_STACK[result_id] = result

            n_desired = PROMPT_STACK[result_id]["n_desired"]
            n_acquired = len(GENERATION_RESULT_STACK[result_id]['continuations'])
            n_remaining = n_desired - n_acquired

            if PROMPT_STACK[result_id]['kwargs']['best_of'] > n_remaining:
                print(f"{result_id}: updating best_of from {PROMPT_STACK[result_id]['kwargs']['best_of']} to {n_remaining}")
                PROMPT_STACK[result_id]['kwargs']['best_of'] = n_remaining
            if n_remaining <= 0:
                print(f"done with {result_id}: have {n_acquired} of {n_desired}")
                GENERATION_RESULT_STACK[result_id]["done"] = True
            else:
                print(f"continuing {result_id}: have {n_acquired} of {n_desired}")

    for result_id in GENERATION_RESULT_STACK.keys():
        if GENERATION_RESULT_STACK[result_id]["done"] and result_id in PROMPT_STACK:
            PROMPT_STACK.pop(result_id)

    for result_id in handled_selection_ids:
        PROMPT_STACK.pop(result_id)

    return jsonify(PROMPT_STACK)

@app.route('/pollselector', methods=['POST'])
def pollselector():
    global GENERATION_RESULT_STACK
    global SELECTION_PROMPT_STACK
    global RESULT_STACK
    global RETENTION_STACK
    global N_RETENTION
    global RETENTION_PROBA_ID

    transferred_ids = set()
    for id_, result in GENERATION_RESULT_STACK.items():
        if result.get("done"):
            SELECTION_PROMPT_STACK[result["base_id"]] = result
            transferred_ids.add(id_)

    for id_ in transferred_ids:
        GENERATION_RESULT_STACK.pop(id_)

    posted_results = request.json["results"]
    RETENTION_STACK = request.json.get("retention_stack")
    if RETENTION_STACK is not None:
        if RETENTION_STACK is not None and RETENTION_PROBA_VIA_GENERATOR:
            url = bridge_service_url + "/raw_select"
            data = {"texts": RETENTION_STACK}
            new_retention_proba_id = bridge_service_unique_id(url, data)

            if new_retention_proba_id != RETENTION_PROBA_ID:
                make_raw_select(data["texts"], new_retention_proba_id)
                RETENTION_PROBA_ID = new_retention_proba_id

        N_RETENTION = len(RETENTION_STACK)
        print(f"N_RETENTION: {N_RETENTION}")
    print(posted_results)

    handled_ids = set()

    for result_id, result in posted_results.items():
        if result_id in SELECTION_PROMPT_STACK:
            RESULT_STACK[result_id] = result
            handled_ids.add(result_id)

    for result_id in handled_ids:
        SELECTION_PROMPT_STACK.pop(result_id)

    RETENTION_STACK_PROBA = None
    if RETENTION_PROBA_ID in RESULT_STACK:
        RETENTION_STACK_PROBA = RESULT_STACK[RETENTION_PROBA_ID]["selection_proba"]

    return jsonify({"SELECTION_PROMPT_STACK": SELECTION_PROMPT_STACK,
                    "RETENTION_STACK_PROBA": RETENTION_STACK_PROBA})


@app.route('/getresult', methods=['POST'])
def getresult():
    global PROMPT_STACK
    global RESULT_STACK
    global SELECTION_PROMPT_STACK
    global SELECTION_RESULT_STACK

    response = {"done": False, "result": ""}

    desired_id = request.form["id"]

    if desired_id in RESULT_STACK:
        response["done"] = True
        response["result"] = RESULT_STACK.pop(desired_id)
        print(f"got result for {desired_id}")

    return jsonify(response)


if __name__=="__main__":
    try:
        app.run(host="0.0.0.0", debug=True)
    except KeyboardInterrupt:
        print("closing session...")
        sess_.close()
        raise KeyboardInterrupt
