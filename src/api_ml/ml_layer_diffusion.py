import sys
import time

import requests
import numpy as np
from transformer_utils.util.tfm_utils import get_local_path_from_huggingface_cdn

import improved_diffusion.pipeline
from multimodal.diffusion_helpers import run_pipeline

from config.autoresponder_config import *

from util.util import typed_namedtuple_to_dict, collect_and_show, show_gpu

import config.bot_config_singleton
bot_specific_constants = config.bot_config_singleton.bot_specific_constants

bridge_service_port = bot_specific_constants.bridge_service_port
BRIDGE_SERVICE_REMOTE_HOST = bot_specific_constants.BRIDGE_SERVICE_REMOTE_HOST


CLOSED_REQUESTS = {}


# constants
HF_REPO_NAME_DIFFUSION = 'nostalgebraist-autoresponder-diffusion'
model_path_diffusion = 'nostalgebraist-autoresponder-diffusion'
timestep_respacing_sres1 = '250'
timestep_respacing_sres2 = '250'

DIFFUSION_DEFAULTS = dict(batch_size=8, n_samples=8)

# download
if not os.path.exists(model_path_diffusion):
    model_tar_name = 'model.tar'
    model_tar_path = get_local_path_from_huggingface_cdn(
        HF_REPO_NAME_DIFFUSION, model_tar_name
    )
    subprocess.run(f"tar -xf {model_tar_path} && rm {model_tar_path}", shell=True)

checkpoint_path_sres1 = os.path.join(model_path_diffusion, "sres1.pt")
config_path_sres1 = os.path.join(model_path_diffusion, "config_sres1.json")

checkpoint_path_sres2 = os.path.join(model_path_diffusion, "sres2.pt")
config_path_sres2 = os.path.join(model_path_diffusion, "config_sres2.json")

# load
sampling_model_sres1 = improved_diffusion.pipeline.SamplingModel.from_config(
    checkpoint_path=checkpoint_path_sres1,
    config_path=config_path_sres1,
    timestep_respacing=timestep_respacing_sres1
)

sampling_model_sres2 = improved_diffusion.pipeline.SamplingModel.from_config(
    checkpoint_path=checkpoint_path_sres2,
    config_path=config_path_sres2,
    timestep_respacing=timestep_respacing_sres2
)

pipeline = improved_diffusion.pipeline.SamplingPipeline(sampling_model_sres1, sampling_model_sres2)



def poll(
    dummy=False,
    ports=[
        bridge_service_port,
    ],
    routes=[
        "polldiffusion",
    ],
    show_memory=True,
):
    global CLOSED_REQUESTS

    for port, route in zip(ports, routes):
        r = requests.get(
            f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
        )

        PROMPT_STACK = {prompt_id: data for prompt_id, data in r.json().items()}

        RESULT_STACK = {}

        for prompt_id, data in PROMPT_STACK.items():
            if prompt_id in CLOSED_REQUESTS:
                RESULT_STACK[prompt_id] = CLOSED_REQUESTS[prompt_id]
                continue

            args = DIFFUSION_DEFAULTS
            args.update(data)

            print(f"running: {args}")

            result = run_pipeline(**args)

            if isinstance(result, np.ndarray):
                result = result.tolist()

            RESULT_STACK[prompt_id] = {"result": result}

        if len(RESULT_STACK) > 0:
            requests.post(
                f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
                json=RESULT_STACK if not dummy else {},
            )

            collect_and_show()
            if show_memory:
                show_gpu()

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
        "polldiffusion",
    ],
    show_memory=True,
):
    open_request_ids = set()
    while True:
        open_request_ids = poll(dummy=dummy, ports=ports, routes=routes, show_memory=show_memory)
        if len(open_request_ids) == 0 or dummy:
            time.sleep(period)
        else:
            time.sleep(0.2)


if __name__ == "__main__":
    sys.exit(loop_poll())
