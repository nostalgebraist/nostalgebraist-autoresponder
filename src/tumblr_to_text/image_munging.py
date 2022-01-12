import re, random

from multimodal.image_analysis_static import (
    V9_IMAGE_FORMATTER,
    IMAGE_DIR,
    IMAGE_DELIMITER,
    IMAGE_DELIMITER_WHITESPACED
)

from multimodal import image_analysis_singleton

image_analysis_cache = image_analysis_singleton.IMAGE_ANALYSIS_CACHE

from multimodal.text_segmentation import make_image_simple

from api_ml.diffusion_connector import make_image_with_diffusion

# image stuff


def find_images_and_sub_text(
    text: str,
    image_formatter=V9_IMAGE_FORMATTER,
    verbose=False,
):
    text_subbed = text

    for match in re.finditer(r"(<img src=\")([^\"]+)(\"[^>]*>)", text):
        imtext = image_analysis_cache.extract_and_format_text_from_url(
            match.group(2)
        )

        text_subbed = text_subbed.replace(
            match.group(0),
            imtext,
        )

    text_subbed = re.sub(r"<figure[^>]*>", "", text_subbed)
    text_subbed = text_subbed.replace("</figure>", "")

    return text_subbed


def upload_images_to_tumblr_urls(images, keys, client, blogname):
    if len(images) != len(keys):
        print(
            f"!! warning: in upload_images_to_tumblr_urls, got {len(images)} images but {len(keys)} keys"
        )
        return {}
    if len(images) == 0:
        return {}
    paths = [f"{IMAGE_DIR}/temp{i}.png" for i, im in enumerate(images)]
    for p, im in zip(paths, images):
        im.save(p, format="png")

    # TODO: figure out how to do this in NPF consumption world
    orig_npf_flag = client.using_npf_consumption
    client.npf_consumption_off()

    r = client.create_photo(blogname, state="draft", data=paths)

    r2 = client.posts(blogname, id=r["id"])["posts"][0]
    urls = [ph["original_size"] for ph in r2["photos"]]

    client.delete_post(blogname, id=r["id"])

    if orig_npf_flag:
        client.npf_consumption_on()

    return {k: url for k, url in zip(keys, urls)}


def find_text_images_and_sub_real_images(
    text,
    client,
    blogname,
    verbose=False,
    dryrun=False,
    use_diffusion=False,
    guidance_scale=0.,
    guidance_scale_sres=0.,
):
    print(f'using diffusion?: {use_diffusion}')
    if use_diffusion:
        image_maker = make_image_with_diffusion
        image_maker_kwargs = {"guidance_scale": guidance_scale, "guidance_scale_sres": guidance_scale_sres}
    else:
        image_maker = make_image_simple
        image_maker_kwargs = {}
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    orig_text = text

    text = text.strip(" \n")

    if text.startswith(IMAGE_DELIMITER):
        text = "\n\n" + text

    if text.endswith(IMAGE_DELIMITER):
        text = text + "\n"

    # imtext_regex = r"(\n=======\n)(.+?)(=======\n)"
    escaped_delim = IMAGE_DELIMITER.encode('unicode_escape').decode()
    escaped_delim_ws = IMAGE_DELIMITER_WHITESPACED.encode('unicode_escape').decode()
    imtext_regex = rf"({escaped_delim_ws})(.+?)({escaped_delim}\n)"
    figure_format = """<figure data-orig-height="{h}" data-orig-width="{w}"><img src="{url}" data-orig-height="{h}" data-orig-width="{w}" alt="{alt}"/></figure>"""
    imtexts = []
    imtext_positions = []
    ims_checksum = 0

    for match in re.finditer(
        imtext_regex,
        text,
        flags=re.DOTALL,
    ):
        imtext = match.group(2).rstrip("\n")
        pos = match.start(2)

        imtexts.append(imtext)
        imtext_positions.append(pos)

        ims_checksum += 1

    vprint(f"find_text_images_and_sub_real_images: found {len(imtexts)} imtexts")

    if dryrun:
        return text, ims_checksum

    images = []
    keys = []
    for imtext, pos in zip(imtexts, imtext_positions):
        if len(imtext) > 0:
            images.append(image_maker(imtext, **image_maker_kwargs))
            keys.append((imtext, pos))

    imtexts_to_tumblr_images = upload_images_to_tumblr_urls(
        images, keys, client, blogname
    )

    vprint(f"find_text_images_and_sub_real_images: uploaded {len(imtexts)} images")

    def _replace_with_figure(match):
        imtext = match.group(2).rstrip("\n")
        pos = match.start(2)
        key = (imtext, pos)
        if key in imtexts_to_tumblr_images:
            tumblr_image = imtexts_to_tumblr_images[key]
            vprint(
                f"find_text_images_and_sub_real_images: subbing {repr(tumblr_image)} for {repr(imtext)} at {pos}"
            )
            return figure_format.format(
                url=tumblr_image["url"],
                h=tumblr_image["height"],
                w=tumblr_image["width"],
                alt=imtext,
            )
        else:
            vprint(
                f"find_text_images_and_sub_real_images: nothing to sub for {repr(imtext)} at {pos}"
            )
            return ""

    text_subbed = re.sub(
        imtext_regex,
        _replace_with_figure,
        text,
        flags=re.DOTALL,
    )

    happened = len(imtexts_to_tumblr_images) > 0
    if not happened:
        text_subbed = orig_text  # ensure no munging for matching went through
    return text_subbed, happened
