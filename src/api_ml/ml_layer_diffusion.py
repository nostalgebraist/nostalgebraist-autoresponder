import sys
import time
from io import BytesIO

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



# constants
HF_REPO_NAME_DIFFUSION = 'nostalgebraist/nostalgebraist-autoresponder-diffusion'
model_path_diffusion = 'nostalgebraist-autoresponder-diffusion'
timestep_respacing_sres1 = '40,70,80,40,20'
timestep_respacing_sres2 = '30,15,30,75'

DIFFUSION_DEFAULTS = dict(
    batch_size=2,
    n_samples=2,
    delete_under=-1,
    keep_only_if_above=2,
    truncate_length=380,
    threshold=65,
    clf_free_guidance=True,
    clf_free_guidance_sres=False,
    guidance_scale=1,
    guidance_scale_sres=0,
    use_plms=False,
    use_plms_sres=False,
    guidance_after_step_base=1000,
)

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
    for port, route in zip(ports, routes):
        r = requests.get(
            f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
        )

        data = r.json()
        if data is None or len(data) == 0:
            continue

        args = {k: v for k, v in DIFFUSION_DEFAULTS.items()}
        args.update(data)

        print(f"running: {args}")

        t1 = time.time()
        result = run_pipeline(pipeline, **args)  # PIL Image
        delta_t = time.time() - t1
        print(f"pipeline took {delta_t:.1f}s")

        with BytesIO() as output:
            result.save(output, "png")
            b = output.getvalue()

        if not dummy:
            requests.post(
                f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
                data=b
            )

        collect_and_show()
        if show_memory:
            show_gpu()


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
    n_loops=None,
):
    loop_counter = 0

    def _should_stop(loop_counter):
        if n_loops is not None:
            return loop_counter >= n_loops
        return False

    while not _should_stop(loop_counter):
        poll(dummy=dummy, ports=ports, routes=routes, show_memory=show_memory)
        time.sleep(period)
        loop_counter += 1

if __name__ == "__main__":
    sys.exit(loop_poll())
