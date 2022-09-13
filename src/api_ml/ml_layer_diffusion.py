import gc
import sys
import time
from io import BytesIO
from pprint import pprint

import torch
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
# HF_REPO_NAME_DIFFUSION = 'nostalgebraist/nostalgebraist-autoresponder-diffusion'
HF_REPO_NAME_DIFFUSION = 'nostalgebraist/nostalgebraist-autoresponder-diffusion-captions'
model_path_diffusion = 'nostalgebraist-autoresponder-diffusion'

USE_PLMS = True

if USE_PLMS:
    timestep_respacing_sres1 = 'ddim50'
    timestep_respacing_sres1p5 = 'ddim50'
    timestep_respacing_sres2 = '90,60,60,20,20'
    timestep_respacing_sres3 = '90,60,60,20,20'
else:
    timestep_respacing_sres1 = '250'
    timestep_respacing_sres1p5 = '90,60,60,20,20'
    timestep_respacing_sres2 = '90,60,60,20,20'
    timestep_respacing_sres3 = '90,60,60,20,20'
FORCE_CAPTS = True

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
    timestep_respacing=timestep_respacing_sres1,
    # silu_impl='torch',
)

sampling_model_sres1p5 = None

if using_sres1p5:
    sampling_model_sres1p5 = improved_diffusion.pipeline.SamplingModel.from_config(
        checkpoint_path=checkpoint_path_sres1p5,
        config_path=config_path_sres1p5,
        timestep_respacing=timestep_respacing_sres1p5,
        clipmod=sampling_model_sres1.model.clipmod,
        # silu_impl='torch',
    )

sampling_model_sres2 = improved_diffusion.pipeline.SamplingModel.from_config(
    checkpoint_path=checkpoint_path_sres2,
    config_path=config_path_sres2,
    timestep_respacing=timestep_respacing_sres2,
    # silu_impl='torch',
)
sampling_model_sres2.model.image_size = 256

sampling_model_sres3 = None

if using_sres3:
    sampling_model_sres3 = improved_diffusion.pipeline.SamplingModel.from_config(
        checkpoint_path=checkpoint_path_sres3,
        config_path=config_path_sres3,
        timestep_respacing=timestep_respacing_sres3,
        # silu_impl='torch',
    )
    sampling_model_sres3.model.image_size = 512

# sampling_model_sres3.model.convert_to_fp16()

# for sm in [sampling_model_sres1, sampling_model_sres1p5, sampling_model_sres2, sampling_model_sres3]:
#     sm.model = sm.model.cpu()

if USE_PLMS:
    sampling_model_sres1.set_timestep_respacing(timestep_respacing_sres1, double_mesh_first_n=3)
    sampling_model_sres1p5.set_timestep_respacing(timestep_respacing_sres1p5, double_mesh_first_n=3)


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
    did_generation = False

    for port, route in zip(ports, routes):
        r = requests.get(
            f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
        )

        data = r.json()
        if data is None or len(data) == 0:
            continue

        pprint(data)

        did_generation = True

        text = data['prompt'][:TRUNCATE_LENGTH]

        capt = data.get('capt')
        if FORCE_CAPTS and capt is None:
            print('using fallback capt')
            capt = 'unknown'

        t1 = time.time()

        # sampling_model_sres1.model.cuda();

        guidance_scale = data['guidance_scale']
        guidance_scale_txt = data.get('guidance_scale_txt')
        if guidance_scale_txt is None:
            guidance_scale_txt = guidance_scale
        pprint(dict(guidance_scale=guidance_scale, guidance_scale_txt=guidance_scale_txt))

        result = sampling_model_sres1.sample(
            text=text,
            batch_size=1,
            n_samples=1,
            to_visible=True,
            clf_free_guidance=True,
            guidance_scale=guidance_scale,
            guidance_scale_txt=guidance_scale_txt,
            dynamic_threshold_p=data.get('dynamic_threshold_p', 0.995),
            use_plms=USE_PLMS,
            capt=capt,
        )

        print('step1 done')
        collect_and_show()
        if show_memory:
            show_gpu()

        # sampling_model_sres1.model.cpu();

        if using_sres1p5:
            # sampling_model_sres1p5.model.cuda();

            result = sampling_model_sres1p5.sample(
                text=text,
                batch_size=1,
                n_samples=1,
                to_visible=True,
                from_visible=True,
                low_res=result,
                clf_free_guidance=True,
                guidance_scale=guidance_scale,
                guidance_scale_txt=guidance_scale_txt,
                dynamic_threshold_p=data.get('dynamic_threshold_p', 0.995),
                noise_cond_ts=225,
                use_plms=USE_PLMS,
                capt=capt,
            )

            print('step1p5 done')
            collect_and_show()
            if show_memory:
                show_gpu()

            # sampling_model_sres1p5.model.cpu();

        # sampling_model_sres2.model.cuda();

        # guidance_scale_step2 = 0 if text == "" else guidance_scale_txt
        guidance_scale_step2 = 0

        result = sampling_model_sres2.sample(
            text=text if sampling_model_sres2.model.txt else None,
            batch_size=1,
            n_samples=1,
            to_visible=True,
            from_visible=True,
            low_res=result,
            clf_free_guidance=True,
            guidance_scale=guidance_scale_step2,
            noise_cond_ts=150,
            # dynamic_threshold_p=data.get('dynamic_threshold_p', 0.995),
            # use_plms=USE_PLMS,
        )

        print('step2 done')
        collect_and_show()
        if show_memory:
            show_gpu()

        # sampling_model_sres2.model.cpu();

        if using_sres3:
            # sampling_model_sres3.model.cuda();

            result = sampling_model_sres3.sample(
                text=None,
                batch_size=1,
                n_samples=1,
                to_visible=True,
                from_visible=True,
                low_res=result,
                noise_cond_ts=75,
                # dynamic_threshold_p=data.get('dynamic_threshold_p', 0.995),
                # use_plms=USE_PLMS,
            )

            # sampling_model_sres3.model.cpu();

        im = Image.fromarray(result[0])

        print('step3 done')
        collect_and_show()
        if show_memory:
            show_gpu()

        delta_t = time.time() - t1
        print('--------------------')
        print(f"pipeline took {delta_t:.1f}s")
        print('--------------------\n')

        with BytesIO() as output:
            im.save(output, "png")
            b = output.getvalue()

        if not dummy:
            requests.post(
                f"{BRIDGE_SERVICE_REMOTE_HOST}:{port}/{route}",
                data=b
            )
    return did_generation


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
        did_generation = poll(dummy=dummy, ports=ports, routes=routes, show_memory=show_memory)
        if did_generation:
            collect_and_show()
        time.sleep(period)
        loop_counter += 1

if __name__ == "__main__":
    sys.exit(loop_poll())
