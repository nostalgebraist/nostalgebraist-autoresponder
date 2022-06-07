import sys
import time
from io import BytesIO
from PIL import Image

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
timestep_respacing_sres1 = '100,150'
timestep_respacing_sres1p5 = '90,60,60,20,20'
timestep_respacing_sres2 = '150,50,25,25'
timestep_respacing_sres3 = '250'

TRUNCATE_LENGTH = 380

# download
if not os.path.exists(model_path_diffusion):
    model_tar_name = 'model.tar'
    model_tar_path = get_local_path_from_huggingface_cdn(
        HF_REPO_NAME_DIFFUSION, model_tar_name
    )
    subprocess.run(f"tar -xf {model_tar_path} && rm {model_tar_path}", shell=True)

checkpoint_path_sres1 = os.path.join(model_path_diffusion, "sres1.pt")
config_path_sres1 = os.path.join(model_path_diffusion, "config_sres1.json")

checkpoint_path_sres1p5 = os.path.join(model_path_diffusion, "sres1p5.pt")
config_path_sres1p5 = os.path.join(model_path_diffusion, "config_sres1p5.json")

using_sres1p5 = os.path.exists(checkpoint_path_sres1p5) and os.path.exists(config_path_sres1p5)

checkpoint_path_sres2 = os.path.join(model_path_diffusion, "sres2.pt")
config_path_sres2 = os.path.join(model_path_diffusion, "config_sres2.json")

checkpoint_path_sres3 = os.path.join(model_path_diffusion, "sres3.pt")
config_path_sres3 = os.path.join(model_path_diffusion, "config_sres3.json")

# shared model for steps 2 and 3
checkpoint_path_sres2_3 = os.path.join(model_path_diffusion, "sres2_3.pt")

if os.path.exists(checkpoint_path_sres2_3):
    checkpoint_path_sres2 = checkpoint_path_sres2_3
    checkpoint_path_sres3 = checkpoint_path_sres2_3

using_sres3 = os.path.exists(checkpoint_path_sres3) and os.path.exists(config_path_sres3)

# load
sampling_model_sres1 = improved_diffusion.pipeline.SamplingModel.from_config(
    checkpoint_path=checkpoint_path_sres1,
    config_path=config_path_sres1,
    timestep_respacing=timestep_respacing_sres1
)

sampling_model_sres1p5 = None

if using_sres1p5:
    sampling_model_sres1p5 = improved_diffusion.pipeline.SamplingModel.from_config(
        checkpoint_path=checkpoint_path_sres1p5,
        config_path=config_path_sres1p5,
        timestep_respacing=timestep_respacing_sres1p5
    )

sampling_model_sres2 = improved_diffusion.pipeline.SamplingModel.from_config(
    checkpoint_path=checkpoint_path_sres2,
    config_path=config_path_sres2,
    timestep_respacing=timestep_respacing_sres2
)
sampling_model_sres2.model.image_size = 256

sampling_model_sres3 = None

if using_sres3:
    sampling_model_sres3 = improved_diffusion.pipeline.SamplingModel.from_config(
        checkpoint_path=checkpoint_path_sres3,
        config_path=config_path_sres3,
        timestep_respacing=timestep_respacing_sres3
    )
    sampling_model_sres3.model.image_size = 512


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

        text = data['prompt'][:TRUNCATE_LENGTH]

        t1 = time.time()

        result = sampling_model_sres1.sample(
            text=text,
            batch_size=1,
            n_samples=1,
            to_visible=False,
            clf_free_guidance=True,
            guidance_scale=data.get('guidance_scale', 1),
            dynamic_threshold_p=data.get('dynamic_threshold_p', 0.995)
        )

        if using_sres1p5:
            result = sampling_model_sres1p5.sample(
                text=text,
                batch_size=1,
                n_samples=1,
                to_visible=False,
                from_visible=False,
                low_res=result,
                guidance_scale=data.get('guidance_scale', 1),
                noise_cond_ts=300,
            )

        result = sampling_model_sres2.sample(
            text=text if sampling_model_sres2.model.txt else None,
            batch_size=1,
            n_samples=1,
            to_visible=not using_sres3,
            from_visible=False,
            low_res=result,
        )

        if using_sres3:
            result = sampling_model_sres3.sample(
                text=None,
                batch_size=1,
                n_samples=1,
                to_visible=True,
                from_visible=False,
                low_res=result,
                noise_cond_ts=125,
            )
        im = Image.fromarray(result[0])

        delta_t = time.time() - t1
        print(f"pipeline took {delta_t:.1f}s")

        with BytesIO() as output:
            im.save(output, "png")
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
