from functools import partial

from io import BytesIO

import numpy as np
from PIL import Image
from ngram import NGram

import multimodal.image_analysis as ima


def read_generated_images(image_array, threshold=80):
    bs = []

    for a in image_array:
        with BytesIO() as output:
            with Image.fromarray(a) as im:
                im.save(output, "png")
            b = output.getvalue()
        bs.append(b)

    postprocessor_kwargs = [{"threshold": threshold}]
    results = [
        ima.execute_callspecs(
            [ima.detect_text_spec], b, postprocessor_kwargs=postprocessor_kwargs
        )
        for b in bs
    ]
    final_text = [ima.collect_text([r]) for r in results]
    return final_text


def get_ngram_similarity(gold, candidates, N=3, strip_space=True):
    def _strip_space(s):
        if not strip_space:
            return s
        return "\n".join([part.strip(" ") for part in s.split("\n")])

    ng = NGram([_strip_space(gold)], N=N)

    sims = []

    for c in candidates:
        ng_out = ng.search(_strip_space(c))
        if len(ng_out) == 0:
            sims.append(0.0)
        else:
            sims.append(ng_out[0][1])

    return sims


def read_and_get_ngram_similarity(image_array, gold, N=3, verbose=True, threshold=80):
    texts = read_generated_images(image_array, threshold=threshold)
    if verbose:
        print(f"got texts:\n{texts}")
    sims = get_ngram_similarity(gold, texts, N=N)
    if verbose:
        print(f"got sims:\n{sims}")
    return texts, sims


def read_and_prune(
    image_array,
    gold,
    N=3,
    verbose=True,
    threshold=80,
    delete_under=0.1,
    keep_only_if_above=0.95,
):
    if verbose:
        print(f"using delete_under={delete_under} | keep_only_if_above={keep_only_if_above}")
    texts, sims = read_and_get_ngram_similarity(
        image_array, gold, N=N, verbose=verbose, threshold=threshold
    )

    ok_indices = set(range(len(sims)))

    if max(sims) >= keep_only_if_above:
        if verbose:
            print("keep_only_if_above triggered")
        ok_indices = {i for i, s in enumerate(sims) if s >= keep_only_if_above}
    else:
        ok_indices = {i for i, s in enumerate(sims) if s >= delete_under}

    ok_indices = sorted(ok_indices)

    if len(ok_indices) == 0:
        retained = []
        retained_texts = []
    else:
        retained = np.stack([image_array[i] for i in ok_indices], axis=0)
        retained_texts = [texts[i] for i in ok_indices]

    if verbose:
        print(
            f"after prune: {len(ok_indices)} of {len(sims)} left\n{[(i, sims[i]) for i in ok_indices]}"
        )

    # return retained, retained_texts
    return retained, [gold] * len(retained)


def select_best(image_array, gold, N=3, verbose=True):
    texts, sims = read_and_get_ngram_similarity(image_array, gold, N=N, verbose=verbose)

    amax = np.argmax(sims)
    if verbose:
        print(f"picked {amax} with sim {sims[amax]}, text {texts[amax]}")

    return image_array[amax], amax


def run_pipeline(
    pipeline,
    prompt,
    batch_size,
    n_samples,
    clf_free_guidance=False,
    guidance_scale=0.,
    txt_drop_string='<mask><mask><mask><mask>',
    n_samples_sres=None,
    batch_size_sres=None,
    clf_free_guidance_sres=False,
    guidance_scale_sres=0.,
    seed=None,
    N=3,
    verbose=True,
    threshold=80,
    delete_under=0.1,
    keep_only_if_above=0.95,
    to_pil_image=True,
    truncate_length=None,
    return_both_resolutions=False,
):
    if truncate_length:
        prompt = prompt[:truncate_length]

    prune_fn = partial(
        read_and_prune,
        N=N,
        verbose=verbose,
        threshold=threshold,
        delete_under=delete_under,
        keep_only_if_above=keep_only_if_above,
    )
    image_array = pipeline.sample_with_pruning(
        text=prompt,
        n_samples=n_samples,
        batch_size=batch_size,
        prune_fn=prune_fn,
        seed=seed,
        batch_size_sres=batch_size_sres,
        n_samples_sres=n_samples_sres,
        clf_free_guidance=clf_free_guidance,
        guidance_scale=guidance_scale,
        txt_drop_string=txt_drop_string,
        clf_free_guidance_sres=clf_free_guidance_sres,
        guidance_scale_sres=guidance_scale_sres,
        return_both_resolutions=return_both_resolutions
    )

    if return_both_resolutions:
        image_array, image_array_lowres = image_array

    best, amax = select_best(image_array, prompt)

    if to_pil_image:
        best = Image.fromarray(best)

    if return_both_resolutions:
        best_lowres = image_array_lowres[amax]

        if to_pil_image:
            best = Image.fromarray(best)

        return best, best_lowres

    return best
