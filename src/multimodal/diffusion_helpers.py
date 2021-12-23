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


def get_ngram_similarity(gold, candidates, N=3):
    ng = NGram([gold], N=N)

    sims = []

    for c in candidates:
        ng_out = ng.search(c)
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
        print(f"got sims:\n{[(s, t) for s, t in zip(sims, texts)]}")
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
    texts, sims = read_and_get_ngram_similarity(image_array, gold, N=N, verbose=verbose, threshold=threshold)

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
        print(f"after prune:\n{[(i, sims[i], texts[i]) for i in ok_indices]}")

    # return retained, retained_texts
    return retained,  [gold] * len(retained)


def select_best(image_array, gold, N=3, verbose=True):
    texts, sims = read_and_get_ngram_similarity(image_array, gold, N=N, verbose=verbose)

    amax = np.argmax(sims)
    if verbose:
        print(f"picked {amax} with sim {sims[amax]}, text {texts[amax]}")

    return image_array[amax]


def run_pipeline(
    pipeline,
    prompt,
    batch_size,
    n_samples,
    seed=None,
    N=3,
    verbose=True,
    threshold=80,
    delete_under=0.1,
    keep_only_if_above=0.95,
):
    prune_fn = partial(read_and_prune, N=N, verbose=verbose,
                       threshold=threshold,
                       delete_under=delete_under, keep_only_if_above=keep_only_if_above)
    image_array = pipeline.sample_with_pruning(text=prompt, n_samples=n_samples, batch_size=batch_size, prune_fn=prune_fn, seed=10)

    best = select_best(image_array, prompt)
    return best
