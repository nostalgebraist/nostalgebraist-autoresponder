from autoresponder_config import *
from autoresponder_static import *
from autoresponder_static_v8 import *


def finalize_prompt_for_neural(
    prompt,
    override_disable_forumlike=False,
    forced_tags_string=None,
    write_fic_override=False,
):
    if GLOBAL_DEBUG:
        print(f"in finalize_prompt_for_neural, got prompt: {repr(prompt)}")
    prompt = final_munge_before_neural(
        prompt,
        override_disable_forumlike=override_disable_forumlike,
        left_strip_newline=SELECTOR_LEFT_STRIP_NEWLINE_IN_FORUMLIKE,
        forced_tags_string=forced_tags_string,
        write_fic_override=write_fic_override,
    )
    prompt = prompt.replace(EOT_FULL, "")
    if EOT_PREPEND:
        prompt = EOT_FULL + prompt
    if GLOBAL_DEBUG:
        print(f"finalize_prompt_for_neural, using prompt (munged): {repr(prompt)}")
    return prompt


class MLModelInterface:
    def __init__(self):
        raise NotImplementedError

    def do(self, method, *args, **kwargs):
        data = {"model": self.name, "method": method, "args": args, "kwargs": kwargs}
        new_id = bridge_service_unique_id(bridge_service_url, data)

        data_to_send = dict()
        data_to_send.update(data)
        data_to_send["id"] = new_id

        requests.post(bridge_service_url + "/requestml", data=data_to_send)

        return new_id


class GeneratorModelInterface(MLModelInterface):
    def __init__(self):
        self.name = "generator"

    def write(self, *args, **kwargs):
        return self.do("write", *args, **kwargs)

    def done_writing(self, *args, **kwargs):
        return self.do("done_writing", *args, **kwargs)


class SideJudgmentModelInterface(MLModelInterface):
    def __init__(self, name):
        self.name = name

    def predict_proba(self, *args, **kwargs):
        return self.do("predict_proba", *args, **kwargs)

    def _predict(self, *args, **kwargs):
        return self.do("_predict", *args, **kwargs)


generator_model = GeneratorModelInterface()
selector_est = SideJudgmentModelInterface("selector")
sentiment_est = SideJudgmentModelInterface("sentiment")


def request_prompted_continuation(
    prompt: str,
    verbose=False,
    mirotarg=None,  # TODO: allow vary across batch, add noise inside this fn
):
    if mirotarg is None:
        mirotarg = np.random.choice(MIRO_TARGET_ALL)

    return generator_model.write(prompt, mirotarg=mirotarg, verbose=verbose)


def parse_continuation(continuation: str, verbose=True, wrap=False):
    if verbose:
        print(
            f"parsing the following raw output:\n------------------\n{fill(continuation)}\n------------------\n"
        )

    # split out tags, if present
    if V8:
        post, _, tag_text = continuation.partition("\n")
    else:
        post, _, tag_text = continuation.partition(T_CHAR)
    tags = []
    if len(tag_text) > 0:
        tags = [s.rstrip(" ") for s in tag_text.split("#")]

    post = post.lstrip(ORIG_POST_CHAR)
    parsed = {"post": post, "tags": tags}
    return parsed


def get_textpost_prompt():
    overrides = {}

    roll = np.random.rand()
    if roll < FORUMLIKE_REVIEW_PROB:
        prompt = CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"]
        overrides["v8_timestamp"] = ""
        overrides["v10_timestamp"] = ""
    elif roll < FORUMLIKE_REVIEW_PROB + FORUMLIKE_FIC_PROB:
        prompt = CONTROL_SEG_CONFIG["ORIG_FICTION_CHAR_FORUMLIKE"]
        overrides["v8_timestamp"] = ""
        overrides["v10_timestamp"] = ""
        overrides["tag_string_raw"] = "#original fiction"
    else:
        prompt = CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"]
    print(f"using prompt={repr(prompt)}")

    return prompt, overrides


profane_substrings = {
    "shit",
    "fuck",
    "sex",
    "crap",
    "hell",
    "damn",
    "vagina",
    "penis",
    "genital",
    "piss",
    "gay",
}


def basic_n_continuations(
    prompt,
    N,
    avoid_if_under=20,
    avoid_half_if_under=40,
    avoid_if_cut_off=True,
    split_on_control_char=False,
    use_textpost_prompt=False,
    avoid_initial_blockquote=False,
    continue_if_cut_off=False,
    avoid_if_profane=False,
    v8_timestamp="",
    v10_timestamp="",
    max_continue_steps=MAX_CONTINUE_STEPS,
    mirotarg=None,
    forced_tags_string=None,
    write_fic_override=False,
    override_disable_forumlike=False,
    verbose=False,
):
    continuation_side_data = []

    if mirotarg is None:
        mirotarg = np.random.choice(MIRO_TARGET_ALL)

    relevant_timestamp = v10_timestamp if V10 else v8_timestamp

    if use_textpost_prompt:
        prompt, textpost_overrides = get_textpost_prompt()
        v8_timestamp = textpost_overrides.get("v8_timestamp", v8_timestamp)
        v10_timestamp = textpost_overrides.get("v10_timestamp", v10_timestamp)
        relevant_timestamp = v10_timestamp if V10 else v8_timestamp

        if V8:
            ts_string = format_segment_v8_time(
                relevant_timestamp, control_seg_config=CONTROL_SEG_CONFIG
            )
            if CONTROL_SEG_CONFIG["flags"]["add_control_prefix_to_forced_tag_strings"]:
                tag_string = format_segment_v8_tags(
                    textpost_overrides.get("tag_string_raw", ""),
                    control_seg_config=CONTROL_SEG_CONFIG,
                )
            else:
                tag_string = textpost_overrides.get("tag_string_raw", "")
            prompt = globally_format_v8(
                doc_tagless=prompt,
                ts_string=ts_string,
                interlocutor_string=format_segment_v8_interlocutors(""),
                tag_string=tag_string,
                control_seg_config=CONTROL_SEG_CONFIG,
            )
    elif V8:
        prompt = join_time_sidechannel(prompt, relevant_timestamp)

    prompt = finalize_prompt_for_neural(
        prompt,
        override_disable_forumlike=use_textpost_prompt or override_disable_forumlike,
        forced_tags_string=forced_tags_string,
        write_fic_override=write_fic_override,
    )

    if GLOBAL_DEBUG:
        print(f"in basic_n_continuations, using prompt: {repr(prompt)}")
    continuations = []

    bridge_id = request_prompted_continuation(
        prompt,
        verbose=verbose,
        mirotarg=mirotarg,
    )

    while len(continuations) < N:
        time.sleep(1)
        response = requests.post(
            bridge_service_url + "/getresult", data={"id": bridge_id}
        ).json()

        this_batch_continuations = response["result"][len(continuations) :]

        for c in this_batch_continuations:
            if contains_control_chars(c, control_seg_config=CONTROL_SEG_CONFIG):
                if split_on_control_char:
                    # min_ix = min([i for i, char in enumerate(c) if char in {Q_CHAR, A_CHAR, ORIG_POST_CHAR, UNAME_CHAR}])
                    min_ix = first_control_char(
                        c, control_seg_config=CONTROL_SEG_CONFIG
                    )[1]
                    csub = c[:min_ix]
                    print(f"splitting on control char:")
                    print(
                        f"\t{len(c)} chars, {len(c.split(' '))} words-->\n\t{len(csub)} chars, {len(csub.split(' '))} words"
                    )
                    c = csub
                else:
                    print(f"rejecting because control char: \n{fill(c)}\n")
                    continue

            roll = np.random.rand()
            if len(c.partition("\n")[2].split(" ")) < avoid_if_under:
                print(f"rejecting because length under {avoid_if_under}: \n{fill(c)}\n")
            elif (
                len(c.partition("\n")[2].split(" ")) < avoid_half_if_under
            ) and roll < 0.5:
                print(
                    f"rejecting because length under {avoid_half_if_under} and roll {roll}: \n{fill(c)}\n"
                )
            elif (not c.endswith(eot_end_segment)) and avoid_if_cut_off:
                print(f"rejecting because cut off: \n{fill(c)}\n")
            elif (
                c.partition("\n")[2].lstrip(" \n").startswith("<blockquote")
            ) and avoid_initial_blockquote:
                print(f"rejecting because initial blockquote: \n{fill(c)}\n")
            elif len([char for char in c if char == T_CHAR]) >= 2:
                print(f"rejecting because multiple T_CHAR: \n{fill(c)}\n")
            elif (
                any([subs in c.lower() for subs in profane_substrings])
                and avoid_if_profane
            ):
                print(f"rejecting because profane: \n{fill(c)}\n")
            elif normalize_for_generator(
                c.partition(T_CHAR)[0].strip(whitespace)
            ) in normalize_for_generator(prompt):
                print(f"rejecting because repeating myself: \n{fill(c)}\n")
            else:
                if len(c.partition("\n")[2].split(" ")) < avoid_half_if_under:
                    print(
                        f"keeping with roll {roll}, although length under {avoid_half_if_under}"
                    )
                continuations.append(c)
                continuation_side_data.append({"mirotarg": mirotarg})

    requests.post(bridge_service_url + "/done", data={"id": bridge_id})

    bridge_id = generator_model.done_writing(prompt)
    requests.post(bridge_service_url + "/done", data={"id": bridge_id})

    continuations_ = []
    for continuation in continuations:
        if use_textpost_prompt:
            continuation = prompt + continuation
            if EOT_PREPEND and continuation.startswith("<|endoftext|>"):
                continuation = continuation[len("<|endoftext|>") :]
            if FORUMLIKE and continuation.startswith(ORIG_POST_CHAR_CHINESE):
                continuation = CONTROL_SEG_CONFIG[
                    "ORIG_POST_CHAR_FORUMLIKE"
                ] + continuation.lstrip(ORIG_POST_CHAR_CHINESE)
        continuations_.append(continuation)
    continuations = continuations_

    return continuations, continuation_side_data


def logit_diff_to_pos_sent(x):
    return 1 / (1 + np.exp(-x))


def show_note_probas(texts, probas, sentiment_logit_diffs=None, console_width=110):
    if sentiment_logit_diffs is None:
        for tpe, proba in zip(texts, probas):
            print(f"\tpredicted prob: {proba:.1%}\n")
            print("\n~_~_~_~_~_\n")
            print("\n".join(wrap(tpe, replace_whitespace=False, width=console_width)))
            print("\n~_~_~_~_~_\n")
    else:
        for tpe, proba, ld in zip(texts, probas, sentiment_logit_diffs):
            print(
                f"\tpredicted prob: {proba:.1%}, sentiment_logit_diff {ld:.4f}, pos_sent {logit_diff_to_pos_sent(ld):.1%}\n"
            )
            print("\n~_~_~_~_~_\n")
            print("\n".join(wrap(tpe, replace_whitespace=False, width=console_width)))
            print("\n~_~_~_~_~_\n")


def predict_select(data, debug=False, override_disable_forumlike=False):
    selector_input = []
    for text in data.selector_input:
        for end_segment in {
            eot_end_segment,
            "<|",
        }:  # explicitly support old <| thing, for now
            if text.endswith(end_segment):
                text = text[: -len(end_segment)]
        if T_CHAR not in text and (not V8):
            text = text + T_CHAR

        text = final_munge_before_neural(
            text,
            override_disable_forumlike=override_disable_forumlike,
            left_strip_newline=SELECTOR_LEFT_STRIP_NEWLINE_IN_FORUMLIKE,
        )

        if EOT_PREPEND:
            if (not SELECTOR_EOT_PREPEND) and text.startswith(EOT_FULL):
                text = text[len(EOT_FULL) :]
            if SELECTOR_EOT_PREPEND and (not text.startswith(EOT_FULL)):
                text = EOT_FULL + text

        selector_input.append(text)
    data.loc[:, "selector_input"] = selector_input

    bridge_id = selector_est.predict_proba(data)

    response = None
    while response is None:
        time.sleep(1)
        response = requests.post(
            bridge_service_url + "/getresult", data={"id": bridge_id}
        ).json()

    result = response["result"]
    probs = result[:, 1]
    return probs


def predict_sentiment(data, debug=False):
    selector_input = []
    for text in data.selector_input:
        for end_segment in {
            eot_end_segment,
            "<|",
        }:  # explicitly support old <| thing, for now
            if text.endswith(end_segment):
                text = text[: -len(end_segment)]
        if T_CHAR not in text:
            text = text + T_CHAR
        text = text.partition(T_CHAR)[0]
        if NORMALIZE:
            text = normalize_for_generator(text)
        text = re.sub(r"\<.*?\>", "", text)  # sentiment-specific

        if EOT_PREPEND:
            if (not SELECTOR_EOT_PREPEND) and text.startswith(EOT_FULL):
                text = text[len(EOT_FULL) :]
            if SELECTOR_EOT_PREPEND and (not text.startswith(EOT_FULL)):
                text = EOT_FULL + text

        selector_input.append(text)
    data.loc[:, "selector_input"] = selector_input

    bridge_id = sentiment_est._predict(data, key="logits")

    response = None
    while response is None:
        time.sleep(1)
        response = requests.post(
            bridge_service_url + "/getresult", data={"id": bridge_id}
        ).json()

    logits = response["result"]

    logit_diffs = logits[:, 1:] - logits[:, :1]

    return logit_diffs


RESULT_STACK = {}


def _make_alt_timestamps(v10_timestamp):
    if v10_timestamp is None:
        return []

    alts = []
    months = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "November",
        "December",
    ]
    years = ["2019", "2020", "2021"][::-1]
    for year in years:
        for month in months:
            alts.append(v10_timestamp.replace("January", month).replace("2021", year))
    return alts


def answer_from_gpt2_service(data: dict, ts=None, no_timestamp=False):
    if ts is None:
        ts = datetime.now()
    data["v8_timestamp"] = timestamp_to_v8_format(ts)
    data["v10_timestamp"] = timestamp_to_v10_format(ts)
    if DO_FAKE_V10_YEAR_MONTH:
        data["v10_timestamp"] = (
            " ".join(data["v10_timestamp"].split(" ")[:2]) + " " + FAKE_V10_YEAR_MONTH
        )
    if no_timestamp:
        data["v8_timestamp"] = ""
        data["v10_timestamp"] = ""

    if BEAMSPLIT_TESTING_FLAG:
        data["na_beamsplit"] = True

    result_generator = old_bridge_call__answer(data=data)

    result, _, _ = serve_selection(
        data=result_generator, side_judgment_cache=side_judgment_cache
    )

    # for logging, add any input fields that didn't make the round trip
    for k, v in data.items():
        if k not in result:
            print(f"adding key {k}")
            result[k] = v

    return result


def text_post_from_gpt2_service(loop_persistent_data, mood=None, ts=None):
    data = {"mood": mood}

    if ts is None:
        ts = datetime.now()
    data["v8_timestamp"] = timestamp_to_v8_format(ts)
    data["v10_timestamp"] = timestamp_to_v10_format(ts)
    if DO_FAKE_V10_YEAR_MONTH:
        data["v10_timestamp"] = (
            " ".join(data["v10_timestamp"].split(" ")[:2]) + " " + FAKE_V10_YEAR_MONTH
        )

    if BEAMSPLIT_TESTING_FLAG:
        data["na_beamsplit"] = True

    data["n_retention"] = len(loop_persistent_data.retention_stack)

    result_generator = old_bridge_call__textpost(data=data)

    result, retention_stack, retention_stack_proba = serve_selection(
        data=result_generator,
        side_judgment_cache=loop_persistent_data.side_judgment_cache,
        retention_stack=loop_persistent_data.retention_stack,
        retention_stack_proba=loop_persistent_data.retention_stack_proba,
    )

    save_retention(retention_stack)

    loop_persistent_data.retention_stack = retention_stack
    loop_persistent_data.retention_stack_proba = retention_stack_proba

    # for logging, add any input fields that didn't make the round trip
    for k, v in data.items():
        if k not in result:
            print(f"adding key {k}")
            result[k] = v

    return result, loop_persistent_data


def old_bridge_call__answer(data):
    global PROMPT_STACK

    prompt = data["question"]
    new_id = data["id"]
    mood = data.get("mood")
    exact_prompt = data.get("exact_prompt", False)
    v8_timestamp = data.get("v8_timestamp", "")
    v10_timestamp = data.get("v10_timestamp", "")
    forced_tags_string = data.get("forced_tags_string", "")
    write_fic_override = bool(int(data.get("write_fic_override", 0)))
    write_review_override = bool(int(data.get("write_review_override", 0)))
    return_all_conts = bool(int(data.get("return_all_conts", False)))
    selector_cut_to_final_exchange = bool(
        int(data.get("selector_cut_to_final_exchange", False))
    )

    if not exact_prompt:
        prompt = (
            UNAME_CHAR
            + data["asking_name"]
            + DEFAULT_CSC["ASK_CHAR"]
            + "\n"
            + prompt
            + "\n"
            + A_CHAR
        )
    elif not exact_prompt:
        prompt = Q_CHAR + prompt + "\n" + A_CHAR
    print(f"got prompt: {prompt}")

    kwargs = {
        "best_of": 13 if (not TRADE_QUALITY_FOR_SPEED) else 10,
        "verbose": True,
        "V5": True,
        "mood": get_mood_by_name(mood),
        "return_all_conts": return_all_conts,
        "selector_cut_to_final_exchange": selector_cut_to_final_exchange,
        "forced_tags_string": forced_tags_string,
        "write_fic_override": write_fic_override,
    }

    if kwargs["write_fic_override"] or write_review_override:
        kwargs["best_of"] = 8 if not (TRADE_QUALITY_FOR_SPEED) else 6

    expected_rejection_frac = estimate_expected_rejections(
        min_logit_diff=kwargs["mood"]["min_allowed_score"],
        max_logit_diff=kwargs["mood"]["max_allowed_score"],
        logit_diff_sample_series=logit_diff_sample_series,
    )

    raw_extra_best_of = (
        int(np.round(kwargs["best_of"] / (1 - expected_rejection_frac)))
        - kwargs["best_of"]
    )
    discounted_extra_best_of = int(
        np.round(raw_extra_best_of * EXPECTED_REJECTION_MULT)
    )

    print(
        f"expecting to reject {expected_rejection_frac:.1%}, need {raw_extra_best_of} extra over best_of={kwargs['best_of']}"
    )
    kwargs["best_of"] += discounted_extra_best_of
    print(f"discounting to {discounted_extra_best_of} --> best_of={kwargs['best_of']}")

    if any([d.get("base_id") == new_id for d in PROMPT_STACK.values()]):
        return jsonify({"collision": True})
    kwargs["strategy"] = "proportional_winnowed"
    kwargs["avoid_if_under"] = 10
    if kwargs["write_fic_override"]:
        kwargs["avoid_if_under"] = 100
    kwargs["avoid_half_if_under"] = 15
    kwargs["avoid_if_cut_off"] = False
    kwargs["split_on_control_char"] = True
    kwargs["avoid_initial_blockquote"] = True
    kwargs["avoid_if_profane"] = False
    if data["asking_name"] == "bukbot":
        kwargs["avoid_if_profane"] = True
    if True:
        fork = "B" if np.random.rand() > 1 else "A"
    # strategy = "proportional_winnowed"
    strategy = "eps_greedy"
    eps = 0.1
    kwargs["strategy"] = strategy
    kwargs["eps"] = eps

    kwargs["AB_fork"] = fork
    generation_id = str(uuid.uuid4())
    PROMPT_STACK[generation_id] = {
        "type": "answer",
        "prompt": prompt,
        "kwargs": kwargs,
        "base_id": new_id,
        "v8_timestamp": v8_timestamp,
        "v10_timestamp": v10_timestamp,
    }
    PROMPT_STACK[generation_id]["n_desired"] = PROMPT_STACK[generation_id]["kwargs"][
        "best_of"
    ]
    PROMPT_STACK[generation_id]["kwargs"]["best_of"] = GENERATIONS_PER_REQUEST
    print(
        f"desiring {PROMPT_STACK[generation_id]['n_desired']}, per request {PROMPT_STACK[generation_id]['kwargs']['best_of']}"
    )
    return serve_answer(PROMPT_STACK[generation_id])


def old_bridge_call__textpost(data):
    global PROMPT_STACK

    new_id = data["id"]
    mood = data.get("mood")
    v8_timestamp = data.get("v8_timestamp", "")
    v10_timestamp = data.get("v10_timestamp", "")
    return_all_conts = bool(int(data.get("return_all_conts", False)))
    n_retention = int(data.get("n_retention"))

    kwargs = {
        "best_of": 10,
        "prompt_from_dataset": True,  # TODO: remove
        "verbose": True,
        "V5": True,
        "mood": get_mood_by_name(mood),
        "return_all_conts": return_all_conts,
    }

    if any([d.get("base_id") == new_id for d in PROMPT_STACK.values()]):
        return jsonify({"collision": True})
    kwargs["strategy"] = "proportional"
    kwargs["avoid_if_under"] = 20
    kwargs["avoid_half_if_under"] = 40
    kwargs["avoid_if_cut_off"] = False
    kwargs["avoid_initial_blockquote"] = True
    if True:
        fork = "B" if np.random.rand() > 1 else "A"
        # strategy = "proportional_winnowed"
        strategy = "eps_greedy"
        eps = 0.25
        kwargs["strategy"] = strategy
        kwargs["eps"] = eps
        kwargs["AB_fork"] = fork

    n_candidates_target = TEXTPOST_N_CANDIDATES_TARGET

    # TODO: DRY
    expected_rejection_frac = estimate_expected_rejections(
        min_logit_diff=kwargs["mood"]["min_allowed_score"],
        max_logit_diff=kwargs["mood"]["max_allowed_score"],
        logit_diff_sample_series=logit_diff_sample_series,
    )

    raw_extra_best_of = (
        int(np.round(n_candidates_target / (1 - expected_rejection_frac)))
        - n_candidates_target
    )
    discounted_extra_best_of = int(
        np.round(raw_extra_best_of * EXPECTED_REJECTION_MULT)
    )

    print(
        f"expecting to reject {expected_rejection_frac:.1%}, need {raw_extra_best_of} extra over n_candidates_target={n_candidates_target}"
    )
    n_candidates_target += discounted_extra_best_of
    print(
        f"discounting to {discounted_extra_best_of} --> n_candidates_target={n_candidates_target}"
    )

    if n_retention is not None:
        n_candidates_target = max(0, n_candidates_target - n_retention)
        print(f"with {n_retention} on stack, only need {n_candidates_target}")

    kwargs["best_of"] = n_candidates_target

    print(f"AB test: fork {fork}, n_retention {n_retention}, kwargs {kwargs}")

    generation_id = str(uuid.uuid4())
    PROMPT_STACK[generation_id] = {
        "type": "textpost",
        "kwargs": kwargs,
        "base_id": new_id,
        "v8_timestamp": v8_timestamp,
        "v10_timestamp": v10_timestamp,
    }
    PROMPT_STACK[generation_id]["n_desired"] = PROMPT_STACK[generation_id]["kwargs"][
        "best_of"
    ]
    PROMPT_STACK[generation_id]["kwargs"]["best_of"] = GENERATIONS_PER_REQUEST
    print(
        f"desiring {PROMPT_STACK[generation_id]['n_desired']}, per request {PROMPT_STACK[generation_id]['kwargs']['best_of']}"
    )
    return serve_textpost(PROMPT_STACK[generation_id])


def serve_answer(data):
    print("\n------------\n")
    print("serving answer for\n")
    for k, v in data.items():
        print(f"\t{k}: {v}")
    print("\n------------\n")
    prompt = data["prompt"].rstrip(whitespace)

    if EOT_PREPEND and not V8:
        prompt = "<|endoftext|>" + prompt

    kwargs = data["kwargs"]
    avoid_if_under = kwargs.get("avoid_if_under", 20)
    avoid_half_if_under = kwargs.get("avoid_half_if_under", 40)
    avoid_if_cut_off = kwargs.get("avoid_if_cut_off", True)
    split_on_control_char = kwargs.get("split_on_control_char", False)
    avoid_initial_blockquote = kwargs.get("avoid_initial_blockquote", True)
    avoid_if_profane = kwargs.get("avoid_if_profane", False)

    continue_if_cut_off = kwargs.get("continue_if_cut_off", True)
    if continue_if_cut_off:
        avoid_if_cut_off = False

    selector_cut_to_final_exchange = kwargs.get("selector_cut_to_final_exchange", False)
    forced_tags_string = kwargs.get("forced_tags_string", None)
    write_fic_override = kwargs.get("write_fic_override", False)
    print(f"write_fic_override: {write_fic_override}")

    v8_timestamp = data.get("v8_timestamp", "")
    v10_timestamp = data.get("v10_timestamp", "")
    relevant_timestamp = v10_timestamp if V10 else v8_timestamp

    override_disable_forumlike = False
    if prompt.startswith(CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"]):
        override_disable_forumlike = True

    continuations, continuation_side_data = basic_n_continuations(
        prompt,
        N=kwargs["best_of"],
        avoid_if_under=avoid_if_under,
        avoid_half_if_under=avoid_half_if_under,
        avoid_if_cut_off=avoid_if_cut_off,
        use_textpost_prompt=False,
        split_on_control_char=split_on_control_char,
        avoid_initial_blockquote=avoid_initial_blockquote,
        continue_if_cut_off=continue_if_cut_off,
        avoid_if_profane=avoid_if_profane,
        v8_timestamp=v8_timestamp,
        v10_timestamp=v10_timestamp,
        forced_tags_string=forced_tags_string,
        write_fic_override=write_fic_override,
        override_disable_forumlike=override_disable_forumlike,
    )
    parsed = data.copy()
    parsed["continuations"] = [final_munge_after_neural(c) for c in continuations]
    parsed["mirotarg"] = [cd.get("mirotarg") for cd in continuation_side_data]

    if SELECTOR_CAN_SEE_PROMPTS:
        if selector_cut_to_final_exchange and not override_disable_forumlike:
            prompt_cut = cut_to_final_exchange_chinese(prompt)
            selector_inputs = [
                prompt_cut + final_munge_after_neural(c) for c in continuations
            ]
        else:
            selector_inputs = [prompt + c for c in continuations]
    else:
        if FORUMLIKE:
            prompt_forumlike = substitute_forumlike(
                normalize_for_generator(prompt),
                shuffle=False,
                infer_first=False,
                left_strip_newline=SELECTOR_LEFT_STRIP_NEWLINE_IN_FORUMLIKE,
            )
            prompt_finalchar = prompt_forumlike[
                last_control_char(
                    prompt_forumlike,
                    incl_number=False,
                    control_seg_config=CONTROL_SEG_CONFIG,
                )[1] :
            ]
            selector_inputs = [prompt_finalchar + c for c in continuations]
        else:
            selector_inputs = [A_CHAR + c for c in continuations]

    if DO_ALT_TIMESTAMPS:
        for alt_ts in _make_alt_timestamps(v10_timestamp):
            alt_selector_inputs = pd.DataFrame(
                {
                    "selector_input": [
                        join_time_sidechannel(s, alt_ts) for s in selector_inputs
                    ]
                }
            )
            entry_selection_results = predict_select(alt_selector_inputs, debug=True)
            listkey = f"alt_selection_proba__{alt_ts.replace(' ', '_')}"
            parsed[listkey] = [float(p) for p in entry_selection_results]

    selector_inputs = [
        join_time_sidechannel(s, relevant_timestamp) for s in selector_inputs
    ]
    selector_inputs = pd.DataFrame(
        {
            "selector_input": selector_inputs,
            "prompt_finalchar": [f"{A_CHAR}a" for _ in range(len(selector_inputs))],
        }
    )
    if GLOBAL_DEBUG:
        print(f"passing to predict_select: {selector_inputs}")
    selection_results = predict_select(
        selector_inputs,
        debug=True,
        override_disable_forumlike=override_disable_forumlike,
    )
    parsed["selection_proba"] = [float(p) for p in selection_results]
    selector_inputs = pd.DataFrame({"selector_input": parsed["continuations"]})
    sentiment_results = predict_sentiment(selector_inputs, debug=True)
    parsed["sentiment_logit_diffs"] = [float(p) for p in sentiment_results]
    show_note_probas(
        continuations,
        probas=parsed["selection_proba"],
        sentiment_logit_diffs=parsed["sentiment_logit_diffs"],
    )

    if GLOBAL_DEBUG:
        print(f"sending back: {parsed}")

    return parsed


def serve_textpost(data):
    prompt = ""
    kwargs = data["kwargs"]
    avoid_if_under = kwargs.get("avoid_if_under", 20)
    avoid_half_if_under = kwargs.get("avoid_half_if_under", 40)
    avoid_if_cut_off = kwargs.get("avoid_if_cut_off", True)
    split_on_control_char = kwargs.get("split_on_control_char", True)
    avoid_initial_blockquote = kwargs.get("avoid_initial_blockquote", False)

    continue_if_cut_off = kwargs.get("continue_if_cut_off", True)
    if continue_if_cut_off:
        avoid_if_cut_off = False

    try:
        continuations, continuation_side_data = basic_n_continuations(
            prompt,
            N=kwargs["best_of"],
            avoid_if_under=avoid_if_under,
            avoid_half_if_under=avoid_half_if_under,
            avoid_if_cut_off=avoid_if_cut_off,
            split_on_control_char=split_on_control_char,
            use_textpost_prompt=True,
            avoid_initial_blockquote=avoid_initial_blockquote,
            v8_timestamp=data.get("v8_timestamp"),
            v10_timestamp=data.get("v10_timestamp", ""),
            continue_if_cut_off=continue_if_cut_off,
        )
    except Exception as e:
        if EVEN_BETTER_LENGTH:
            raise (e)
        print(f"got {e}, trying without continue_if_cut_off")
        continuations, continuation_side_data = basic_n_continuations(
            prompt,
            N=kwargs["best_of"],
            avoid_if_under=avoid_if_under,
            avoid_half_if_under=avoid_half_if_under,
            avoid_if_cut_off=avoid_if_cut_off,
            split_on_control_char=split_on_control_char,
            use_textpost_prompt=True,
            avoid_initial_blockquote=avoid_initial_blockquote,
            v8_timestamp=data.get("v8_timestamp"),
            v10_timestamp=data.get("v10_timestamp", ""),
            continue_if_cut_off=False,
        )
    parsed = data.copy()
    parsed["continuations"] = [final_munge_after_neural(c) for c in continuations]
    parsed["mirotarg"] = [cd.get("mirotarg") for cd in continuation_side_data]

    if FORUMLIKE:
        selector_inputs = [c for c in continuations]
        for alt_char in [
            CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"],
            CONTROL_SEG_CONFIG["ORIG_FICTION_CHAR_FORUMLIKE"],
        ]:
            selector_inputs = [
                s.replace(alt_char, CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"])
                for s in selector_inputs
            ]
    else:
        selector_inputs = [A_CHAR + c for c in continuations]
    selector_inputs = pd.DataFrame(
        {
            "selector_input": selector_inputs,
            "prompt_finalchar": [
                ORIG_POST_CHAR_CHINESE for _ in range(len(selector_inputs))
            ],
        }
    )
    if GLOBAL_DEBUG:
        print(f"passing to predict_select: {selector_inputs}")
    selection_results = predict_select(
        selector_inputs,
        debug=True,
        override_disable_forumlike=True,
    )
    parsed["selection_proba"] = [float(p) for p in selection_results]

    selector_inputs = pd.DataFrame({"selector_input": parsed["continuations"]})
    sentiment_results = predict_sentiment(selector_inputs, debug=True)
    parsed["sentiment_logit_diffs"] = [float(p) for p in sentiment_results]
    show_note_probas(
        continuations,
        probas=parsed["selection_proba"],
        sentiment_logit_diffs=parsed["sentiment_logit_diffs"],
    )

    if GLOBAL_DEBUG:
        print(f"sending back: {parsed}")

    return parsed


def serve_raw_select(data):
    texts = data["texts"]

    # texts = [s.lstrip("翰") for s in texts]
    if V8:
        vX_timestamp = (
            data.get("v10_timestamp", "") if V10 else data.get("v8_timestamp", "")
        )
        texts = [join_time_sidechannel(s, vX_timestamp) for s in texts]
        texts = [
            s
            if len(find_all_control_chars_chinese(s)) > 0
            else ORIG_POST_CHAR_CHINESE + s
            for s in texts
        ]
        texts = [final_munge_before_neural(s) for s in texts]
    else:
        if FORUMLIKE:
            for alt_char in [
                CONTROL_SEG_CONFIG["REVIEW_CHAR_FORUMLIKE"],
                CONTROL_SEG_CONFIG["ORIG_FICTION_CHAR_FORUMLIKE"],
            ]:
                texts = [
                    s.replace(alt_char, CONTROL_SEG_CONFIG["ORIG_POST_CHAR_FORUMLIKE"])
                    for s in texts
                ]
        texts = [s if ORIG_POST_CHAR in s else ORIG_POST_CHAR + s for s in texts]
    results = {}

    selector_inputs = texts
    if GLOBAL_DEBUG:
        print(f"passing to predict_select: {selector_inputs}")
    selector_inputs = pd.DataFrame(
        {
            "selector_input": selector_inputs,
            "prompt_finalchar": [
                ORIG_POST_CHAR_CHINESE for _ in range(len(selector_inputs))
            ],
        }
    )
    selection_results = predict_select(
        selector_inputs, debug=True, override_disable_forumlike=True
    )
    results["selection_proba"] = [float(p) for p in selection_results]

    selector_inputs = pd.DataFrame(
        {"selector_input": [final_munge_after_neural(s) for s in texts]}
    )
    sentiment_results = predict_sentiment(selector_inputs, debug=True)
    results["sentiment_logit_diffs"] = [float(p) for p in sentiment_results]
    show_note_probas(
        texts,
        probas=results["selection_proba"],
        sentiment_logit_diffs=results["sentiment_logit_diffs"],
    )

    print(f"texts: {texts}\nresults: {results}\n")

    return results
