from config.autoresponder_config import *

import torch, requests

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import config.bot_config_singleton
bot_specific_constants = config.bot_config_singleton.bot_specific_constants

bridge_service_port = bot_specific_constants.bridge_service_port
BRIDGE_SERVICE_REMOTE_HOST = bot_specific_constants.BRIDGE_SERVICE_REMOTE_HOST

model_name = 'flan-t5-xl'
MODELS_SERVED = {'flan'}


def no_init(loading_code):
    def dummy(self):
        return

    modules = [torch.nn.Linear, torch.nn.Embedding, torch.nn.LayerNorm]
    original = {}
    for mod in modules:
        original[mod] = mod.reset_parameters
        mod.reset_parameters = dummy

    result = loading_code()
    for mod in modules:
        mod.reset_parameters = original[mod]

    return result


def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        "google/flan-t5-xl", torch_dtype=torch.float16)
    model.cuda()
    return model, tokenizer


model, tokenizer = no_init(load_model)


def call_flan_t5(model, prompts, max_length=60, temperature=1.0, top_p=0.9, **kwargs):
    single_prompt = 0

    if isinstance(prompts, str):
        prompts = [prompts]
        single_prompt = 1

    ins = {k: v.cuda() for k, v in tokenizer(
        prompts, return_tensors='pt').items()}

    out = model.generate(**ins, do_sample=True, max_length=max_length,
                         temperature=temperature, top_p=top_p, **kwargs)

    out = tokenizer.batch_decode(out.cpu(), skip_special_tokens=True)

    if single_prompt:
        out = out[0]

    return out


def poll(
    dummy=False,
    ports=[
        bridge_service_port,
    ],
    routes=[
        "pollml",
    ],
    show_memory=True,
    multirequest_sequence_in_process=False,
):
    global CLOSED_REQUESTS

    for port, route in zip(ports, routes):
        r = requests.get(
            f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
        )

        PROMPT_STACK = {prompt_id: data for prompt_id,
                        data in r.json().items()}

        RESULT_STACK = {}

        for prompt_id, data in PROMPT_STACK.items():
            if prompt_id in CLOSED_REQUESTS:
                RESULT_STACK[prompt_id] = CLOSED_REQUESTS[prompt_id]
                continue

            if data["model"] not in MODELS_SERVED:
                continue

            requested_args, requested_kwargs = data.get("args", []), data.get(
                "kwargs", {}
            )

            with torch.inference_mode():
                result = call_flan_t5(model, prompts, 
                    *requested_args, **requested_kwargs
                )

            if isinstance(result, np.ndarray):
                result = result.tolist()

            RESULT_STACK[prompt_id] = {"result": result}

            model_info = {
                "model_name": model_name,

            }
            RESULT_STACK[prompt_id]["model_info"] = model_info


        if len(RESULT_STACK) > 0:
            requests.post(
                f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
                json=RESULT_STACK if not dummy else {},
            )

            gc.collect()

        open_request_ids = set()
        for prompt_id in PROMPT_STACK:
            if prompt_id in RESULT_STACK:
                CLOSED_REQUESTS[prompt_id] = RESULT_STACK[prompt_id]

        return open_request_ids


def loop_poll(
    period=1,
    dummy=False,
    ports=[
        bridge_service_port,
    ],
    routes=[
        "pollml",
    ],
    show_memory=True,
    n_loops=None,
    use_almostdone=True,
    multirequest_sequence_in_process=False,
):
    loop_counter = 0
    open_request_ids = set()

    def _should_stop(loop_counter, open_request_ids):
        if n_loops is not None:
            return (loop_counter >= n_loops) and (open_request_ids == set())
        return False

    while not _should_stop(loop_counter, open_request_ids):
        open_request_ids = poll(
            dummy=dummy, ports=ports, routes=routes, show_memory=show_memory,
            multirequest_sequence_in_process=multirequest_sequence_in_process
        )
        if len(open_request_ids) == 0 or dummy:
            time.sleep(period)
        else:
            time.sleep(0.2)
        loop_counter += 1
