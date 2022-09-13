import re, random, html

from multimodal.image_analysis_static import (
    V9_IMAGE_FORMATTER,
    URL_PRESERVING_IMAGE_FORMATTER,
    IMAGE_DIR,
    IMAGE_DELIMITER,
    IMAGE_DELIMITER_WHITESPACED,
    remove_image_urls_and_captions_from_post_text,
    imurl_imtext_regex
)

from multimodal.text_segmentation import make_image_simple

from api_ml.diffusion_connector import make_image_with_diffusion

# image stuff


def find_images_and_sub_text(
    text: str,
    include_urls=False,
    # image_formatter=V9_IMAGE_FORMATTER,
    verbose=False,
    skip=False
):
    from multimodal import image_analysis_singleton

    image_analysis_cache = image_analysis_singleton.IMAGE_ANALYSIS_CACHE

    image_formatter = V9_IMAGE_FORMATTER
    if include_urls:
        image_formatter = URL_PRESERVING_IMAGE_FORMATTER

    text_subbed = text

    for match in re.finditer(r"(<img src=\")([^\"]+)(\"[^>]*>)", text):
        imtext = image_analysis_cache.extract_and_format_text_from_url(
            match.group(2), image_formatter=image_formatter, skip=skip
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


def prep_caption_for_model(caption):
    return " " + caption.lstrip(" ")


def find_text_images_and_sub_real_images(
    text,
    client,
    blogname,
    verbose=False,
    dryrun=False,
    use_diffusion=False,
    guidance_scale=2,
    textless_guidance_scale=4,
    textful_guidance_scale=1,
    text_guidance_scale=1,
    dynamic_threshold_p=0.995,
):
    print(f'using diffusion?: {use_diffusion}')
    if use_diffusion:
        image_maker = make_image_with_diffusion
        image_maker_kwargs = {
            "guidance_scale": guidance_scale,
            "guidance_scale_txt": text_guidance_scale,
            "dynamic_threshold_p": dynamic_threshold_p
        }
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
    figure_format = """{prefix_newline}<figure data-orig-height="{h}" data-orig-width="{w}"><img src="{url}" data-orig-height="{h}" data-orig-width="{w}" alt="{alt}"/></figure>"""
    imtexts = []
    imtext_positions = []
    captions = []
    ims_checksum = 0

    for match in re.finditer(
        imurl_imtext_regex,
        text,
        flags=re.DOTALL,
    ):
        caption = match.group(2)
        if caption is not None:
            caption = caption.strip(" ")
        imtext = match.group(4).rstrip("\n")
        imtext_pos = match.start(4)

        imtexts.append(imtext)
        imtext_positions.append(imtext_pos)
        captions.append(caption)

        ims_checksum += 1

    ncapt = sum(c is not None for c in captions)
    vprint(f"find_text_images_and_sub_real_images: found {len(imtexts)} imtexts, {ncapt} captions")

    if dryrun:
        return text, ims_checksum

    images = []
    keys = []
    regular_guidance_used = False
    textless_guidance_used = False
    textful_guidance_used = False
    for imtext, pos, caption in zip(imtexts, imtext_positions, captions):
        prompt = imtext
        per_image_kwargs = {}
        per_image_kwargs.update(image_maker_kwargs)
        per_image_kwargs['capt'] = prep_caption_for_model(caption)

        textless_guidance_substrings = ['[image]', '[animated gif]']
        textless_guidance_trigger = (len(imtext) == 0) or any(s == imtext.strip().lower() for s in textless_guidance_substrings)

        textful_guidance_trigger = (text_guidance_scale is None) and (max(len(line) for line in imtext.split("\n")) >= 30)

        if textless_guidance_trigger:
            print(f"using textless guidance scale={textless_guidance_scale} for {repr(imtext)}, {repr(caption)}")
            textless_guidance_used = True
            prompt = ''
            per_image_kwargs['guidance_scale'] = textless_guidance_scale
        elif textful_guidance_trigger:
            print(f"using textful guidance scale={textful_guidance_scale} for {repr(imtext)}, {repr(caption)}")
            textful_guidance_used = True
            per_image_kwargs['guidance_scale'] = textful_guidance_scale
        else:
            print(f"using regular guidance scale={guidance_scale} for {repr(imtext)}, {repr(caption)}")
            regular_guidance_used = True

        print(f"Using text_guidance_scale={text_guidance_scale}")

        images.append(image_maker(prompt, **per_image_kwargs))
        keys.append((imtext, pos, caption))

    imtexts_to_tumblr_images = upload_images_to_tumblr_urls(
        images, keys, client, blogname
    )

    vprint(f"find_text_images_and_sub_real_images: uploaded {len(imtexts)} images")

    def _replace_with_figure(match):
        caption = match.group(2)
        if caption is not None:
            caption = caption.strip(" ")
        imtext = match.group(4).rstrip("\n")
        pos = match.start(4)
        if caption is None:
            needs_prefix_newline = True
        else:
            needs_prefix_newline = match.group(1).startswith('\n\n')

        key = (imtext, pos, caption)
        if key in imtexts_to_tumblr_images:
            tumblr_image = imtexts_to_tumblr_images[key]
            vprint(
                f"find_text_images_and_sub_real_images: subbing {repr(tumblr_image)} for {repr(imtext)}, {repr(caption)} at {pos}"
            )
            alt_text = imtext
            if caption is not None:
                alt_text = "[Description] " + caption + " " + "[Text]" + imtext
                alt_text = alt_text.replace("<", "").replace(">", "")  # w/o this, "<PERSON>" entirely vanishes
            return figure_format.format(
                prefix_newline='\n' if needs_prefix_newline else '',
                url=tumblr_image["url"],
                h=tumblr_image["height"],
                w=tumblr_image["width"],
                alt=html.escape(alt_text).replace("\n", " [newline] "),
            )
        else:
            vprint(
                f"find_text_images_and_sub_real_images: nothing to sub for {repr(imtext)}, {repr(caption)} at {pos}"
            )
            return ""

    text_subbed = re.sub(
        imurl_imtext_regex,
        _replace_with_figure,
        text,
        flags=re.DOTALL,
    )

    happened = len(imtexts_to_tumblr_images) > 0
    if not happened:
        text_subbed = orig_text  # ensure no munging for matching went through
    return text_subbed, happened, regular_guidance_used, textless_guidance_used, textful_guidance_used


def mock_up_image_generation_tags_for_heads(continuation: str, guidance_scale: float, debug=False) -> str:
    criterion = IMAGE_DELIMITER_WHITESPACED in continuation
    if not criterion:
        return continuation

    tagstr, newl, suffix = continuation.partition('\n')
    extra = f' #computer generated image, #guidance scale {guidance_scale}'
    if '#' in tagstr:
        tagstr += ','
    mocked_up = tagstr + extra + newl + suffix

    if debug:
        print(f"from\n{repr(continuation)}\nmocked up\n{repr(mocked_up)}\n")

    return mocked_up
