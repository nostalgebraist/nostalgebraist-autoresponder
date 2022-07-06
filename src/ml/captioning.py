from functools import lru_cache

import torch as th

from PIL import Image
from magma import Magma
from magma.image_input import ImageInput
from magma.sampling import generate_cfg

from util.error_handling import LogExceptionAndSkip


@lru_cache(1)
def get_caption_prompt_tensor(prompt: str, magma_wrapper: Magma):
    return magma_wrapper.word_embedding(magma_wrapper.tokenizer.encode(prompt, return_tensors="pt").cuda())


def caption_image_from_url(url: str, magma_wrapper: Magma):
    # TODO: delete this?
    frames = url_to_frame_bytes(url)

    if len(frames) != 1:
        return

    im = Image.open(BytesIO(frames[0]))

    return caption_image(im)


def caption_image(
    path_or_url: str,
    magma_wrapper: Magma,
    temperature=1.0,
    top_p=0.5,
    top_k=0,
    max_steps=30,
    guidance_scale=0,
    prompt='[Image description:',
    longest_of=3,
):
    for k in magma_wrapper.adapter_map:
        magma_wrapper.adapter_map[k] = magma_wrapper.adapter_map[k].cuda()

    magma_wrapper.add_adapters()

    caption = None
    caption_options = []

    with LogExceptionAndSkip('trying to caption image'):
        with th.no_grad():
            for _ in range(longest_of):
                text_t = get_caption_prompt_tensor(prompt, magma_wrapper)

                image_t = magma_wrapper.preprocess_inputs([ImageInput(path_or_url)])

                image_end = image_t.shape[1]

                embeddings = th.cat([image_t, text_t], dim=1)

                output = generate_cfg(
                    model=magma_wrapper,
                    embeddings=embeddings,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    max_steps=max_steps,
                    gs=guidance_scale,
                    image_end=image_end,
                )

                caption_options.append(output[0])

            caption = sorted(caption_options, key=len)[-1]

    magma_wrapper.detach_adapters()

    for k in magma_wrapper.adapter_map:
        magma_wrapper.adapter_map[k] = magma_wrapper.adapter_map[k].cpu()

    return caption
