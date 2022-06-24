from PIL import Image
from magma import Magma
from magma.image_input import ImageInput
from magma.sampling import generate_cfg

from util.error_handling import LogExceptionAndSkip


def caption_image_from_url(url: str, magma_wrapper: Magma):
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
):
    # todo: control sampling params

    inputs =[
        ## supports urls and path/to/image
        ImageInput(path_or_url),
        '[Image description:',  # todo: precompute embeds for this
    ]

    magma_wrapper.add_adapters()

    caption = None

    with LogExceptionAndSkip('trying to caption image'):
        embeddings = model.preprocess_inputs(inputs)

        output = generate_cfg(
            model=magma_wrapper,
            embeddings=embeddings,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_steps=max_steps,
            gs=guidance_scale,
        )

        caption = output[0]

    magma_wrapper.detach_adapters()

    return caption
