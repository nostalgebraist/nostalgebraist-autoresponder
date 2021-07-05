from string import whitespace
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.autonotebook import tqdm

from scipy import ndimage as ndi

from skimage.exposure import histogram, match_histograms
from skimage import segmentation
from skimage.filters import sobel
from skimage.color import label2rgb
from skimage.feature import match_template

from PIL import Image, ImageDraw, ImageFont
from PIL import ImageOps as O


def _validate_image_histogram(image, hist, nbins=None):
    """
    from skimage.filters.thresholding

    https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/thresholding.py#L239-L279
    @ c799c7d
    """
    if image is None and hist is None:
        raise Exception("Either image or hist must be provided.")

    if hist is not None:
        if isinstance(hist, (tuple, list)):
            counts, bin_centers = hist
        else:
            counts = hist
            bin_centers = np.arange(counts.size)
    else:
        counts, bin_centers = histogram(image.ravel(), nbins, source_range="image")
    return counts.astype(float), bin_centers


def hist_local_maxima(
    image=None,
    nbins=256,
    max_iter=10000,
    nozero=False,
    amax=False,
    minratio=None,
    debug=False,
    *,
    hist=None,
):
    """
    from skimage.filters.threshold_minimum internals

    https://github.com/scikit-image/scikit-image/blob/master/skimage/filters/thresholding.py#L763-L797
    @ c799c7d
    """

    def find_local_maxima_idx(hist):
        # We can't use scipy.signal.argrelmax
        # as it fails on plateaus
        maximum_idxs = list()
        if amax:
            maximum_idxs.append(np.argmax(hist))

        avg = np.mean(hist)
        direction = 1

        for i in range(hist.shape[0] - 1):
            if direction > 0:
                if hist[i + 1] < hist[i]:
                    direction = -1

                    criterion = i > 0 or (not nozero)
                    if minratio is not None:
                        criterion = criterion and (hist[i] / avg) >= minratio
                    if criterion:
                        maximum_idxs.append(i)
            else:
                if hist[i + 1] > hist[i]:
                    direction = 1

        if amax:
            maximum_idxs = sorted(set(maximum_idxs))
        return maximum_idxs

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    smooth_hist = counts.astype(np.float64, copy=False)
    smooth_hist_path = []

    for counter in range(max_iter):
        smooth_hist = ndi.uniform_filter1d(smooth_hist, 3)
        smooth_hist_path.append(smooth_hist)
        maximum_idxs = find_local_maxima_idx(smooth_hist)
        if debug:
            print(f"{counter}/{max_iter}: {maximum_idxs}...")
        if len(maximum_idxs) < 3:
            break

    return bin_centers[maximum_idxs], smooth_hist, smooth_hist_path


def hist_quantiles(
    image=None,
    nbins=256,
    q1=0.1,
    q2=0.75,
    nozero=False,
    amax=False,
    minratio=None,
    debug=False,
    *,
    hist=None,
):
    """bad idea"""
    counts, bin_centers = _validate_image_histogram(image, hist, nbins)

    relcounts = counts / counts.sum()
    csum = relcounts.cumsum()

    where_q1 = np.where(csum <= q1)[0]
    where_q2 = np.where(csum > q2)[0]

    if len(where_q1) > 0:
        cut1 = where_q1[-1]
    else:
        cut1 = image.min()

    if len(where_q2) > 0:
        cut2 = where_q2[0]
    else:
        cut2 = image.max()
    return cut1, cut2


def hist_flatregion(
    image=None,
    nbins=256,
    mass_target=0.8,
    verbose=False,
    nozero=False,
    amax=False,
    minratio=None,
    debug=False,
    *,
    hist=None,
):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    counts, bin_centers = _validate_image_histogram(image, hist, nbins)
    relcounts = counts / counts.sum()

    left, right = nbins // 2, nbins // 2
    mass = relcounts[:left].sum() + relcounts[right:].sum()

    step = 0
    while mass > mass_target:
        if (left <= 0) and (right >= nbins):
            break
        if step > nbins:
            # catch bug before inf loop
            break

        dmass_left = relcounts[left - 1] if left > 0 else 2
        dmass_right = relcounts[right] if right < nbins - 1 else 2

        if dmass_left < dmass_right:
            left -= 1
        else:
            right += 1

        mass = relcounts[:left].sum() + relcounts[right:].sum()
        step += 1
        if step % 5 == 0:
            vprint((left, right, f"{mass:.2f}"), end=" ")
    return left, right


def region_based_segment(
    array,
    nozero=False,
    q=None,
    expected_nchars=None,
    show_intermediate=False,
    method="flatregion",
):
    elevation_map = sobel(array)

    if show_intermediate:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(elevation_map, cmap=plt.cm.gray)
        ax.set_title("elevation map")
        ax.axis("off")
        plt.show()

    markers = np.zeros_like(array)

    all_mean = np.mean(array)
    # center_mean = np.mean(
    #     array[
    #         (array.shape[0] // 4) : (3 * array.shape[0] // 4),
    #         (array.shape[1] // 4) : (3 * array.shape[1] // 4),
    #     ]
    # )
    corners = (array[:5, :5], array[:5, -5:], array[-5:, :5], array[-5:, -5:])
    corner_mean = np.mean([np.mean(c) for c in corners])
    invert_for_markers = corner_mean > all_mean  # all_mean > center_mean

    if invert_for_markers:
        array_for_markers = 255 - array
    else:
        array_for_markers = array

    hist, hist_centers = histogram(array_for_markers, normalize=True)
    if q is None:
        if method == "maxima":
            maximum_idxs, smooth_hist, smooth_hist_path = hist_local_maxima(
                array_for_markers, nozero=nozero, debug=False
            )
        elif method == "flatregion":
            maximum_idxs = hist_flatregion(array_for_markers)
        else:
            raise ValueError(method)

        if show_intermediate:
            fig, axes = plt.subplots(1, 2, figsize=(8, 3))
            axes[0].imshow(array, cmap=plt.cm.gray)
            axes[0].axis("off")
            axes[1].plot(hist_centers, hist, lw=2)
            if method == "maxima":
                axes[1].plot(
                    hist_centers,
                    smooth_hist / smooth_hist.sum(),
                    lw=2,
                )
            axes[1].set_title("histogram of gray values")
            for t in maximum_idxs:
                axes[1].axvline(t, c="g")
            plt.show()

            if method == "maxima":
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                traces = smooth_hist_path[::20]
                for i, h in enumerate(traces):
                    ax.plot(
                        hist_centers,
                        h,
                        lw=2,
                        color=(i / len(traces), 0, 1 - (i / len(traces))),
                        alpha=0.5,
                    )
                    for t in maximum_idxs:
                        ax.axvline(t, c="g")

            plt.show()

        qlow, qhigh = maximum_idxs
    else:
        qlow, qhigh = q

    if show_intermediate:
        print((qlow, qhigh))
        print(hist[hist_centers <= qlow].sum() / hist.sum())
        print(hist[hist_centers >= qhigh].sum() / hist.sum())

    markers[array_for_markers <= qlow] = 1
    markers[array_for_markers >= qhigh] = 2

    if show_intermediate:
        fig, ax = plt.subplots(figsize=(4, 3))
        im_ = ax.imshow(
            markers,
        )
        ax.set_title("markers")
        ax.axis("off")
        plt.colorbar(im_)
        plt.show()

    segmentation_array = segmentation.watershed(elevation_map, markers)

    if show_intermediate:
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.imshow(segmentation_array, cmap=plt.cm.gray)
        ax.set_title("segmentation")
        ax.axis("off")
        plt.show()

    segmentation_array_holefilled = ndi.binary_fill_holes(segmentation_array - 1)
    labeled_array, _ = ndi.label(segmentation_array_holefilled)

    # NEW by rob
    labeled_array = np.where(
        segmentation_array == 2, labeled_array, np.zeros_like(labeled_array)
    )

    image_label_overlay = label2rgb(
        labeled_array,
        image=None,
        bg_label=0,
    )

    q = (qlow, qhigh)
    return segmentation_array, labeled_array, image_label_overlay, invert_for_markers, q


def find_ordered_characters(labeled):
    labels = np.unique(labeled[labeled != 0])

    coords = []
    for l in labels:
        a = np.argwhere(labeled == l)

        vmin = a[:, 1].min()
        hmin = a[a[:, 1] == vmin][:, 0].min()
        coords.append((vmin, hmin))

    scores = [c[0] + c[1] for c in coords]

    return [tup[0] for tup in sorted(list(zip(labels, scores)), key=lambda tup: tup[1])]


def label_crop_to_bbox(array, label_array, label, pad=1):
    a = np.argwhere(label_array == label)

    box = (
        a[:, 0].min() - pad,
        a[:, 0].max() + pad + 1,
        a[:, 1].min() - pad,
        a[:, 1].max() + pad + 1,
    )
    box = tuple(max(0, c) for c in box)

    return array[box[0] : box[1], box[2] : box[3]]


def label_to_masked_cropped(
    image_array, labeled, label, darker_foreground, masked_mask=False
):
    mask = np.where(labeled == label, labeled, np.zeros_like(labeled))

    if masked_mask:
        bg = 255 * np.ones_like(image_array)
        image_masked = np.where(mask, mask, bg)
    else:
        bg = (
            255 * np.ones_like(image_array)
            if darker_foreground
            else np.zeros_like(image_array)
        )
        image_masked = np.where(mask, image_array, bg)

    return label_crop_to_bbox(image_masked, mask, label)


def make_imitation_target(image_array, labeled, darker_foreground, char_ix):
    to_imitate = None
    for label in find_ordered_characters(labeled)[char_ix : char_ix + 1]:
        to_imitate = label_to_masked_cropped(
            image_array,
            labeled,
            label,
            darker_foreground,
        )
    return to_imitate


def pad_crop_to(array, other, left=1, top=1):
    result = array

    if other.shape[0] <= result.shape[0]:
        if top:
            result = result[-other.shape[0] :]
        else:
            result = result[: other.shape[0]]
    else:
        delta = other.shape[0] - result.shape[0]
        result = np.pad(result, ((top * delta, (1 - top) * delta), (0, 0)), mode="edge")

    if other.shape[1] <= result.shape[1]:
        if left:
            result = result[:, -other.shape[1] :]
        else:
            result = result[:, : other.shape[1]]
    else:
        delta = other.shape[1] - result.shape[1]
        result = np.pad(
            result,
            (
                (0, 0),
                (
                    left * delta,
                    (1 - left) * delta,
                ),
            ),
            mode="edge",
        )

    return result


def fontbox(fnt, text):
    box = fnt.getbbox(text)
    return (box[2] - box[0], box[3] - box[1])


def fontratio(fnt, text, ref):
    box = fontbox(fnt, text)
    return [box[i] / ref.shape[i] for i in [0, 1]]


def estimate_font_size(path, text, ref):
    fnt = ImageFont.truetype(path, 40)
    ratio = fontratio(fnt, text, ref)
    ratio_single = np.mean(ratio)

    return max(8, int(round(40 / ratio_single)))


def imitate(to_imitate, fontpath, fontsize, text, darker_foreground):
    fnt = ImageFont.truetype(fontpath, fontsize)
    base_offset = -1 * np.array(fnt.getoffset(text)) + 1

    padded_to_imitate = np.pad(
        to_imitate,
        (
            (
                0,
                5,
            ),
            (0, 5),
        ),
        mode="edge",
    )

    imitation = Image.new(
        "L",
        # (to_imitate.shape[1]+10, to_imitate.shape[0]+10),
        (padded_to_imitate.shape[1], padded_to_imitate.shape[0]),
        (255 if darker_foreground else 0,),
    )
    d = ImageDraw.Draw(imitation)

    d.text(base_offset, text, font=fnt, fill=(0 if darker_foreground else 255,))

    # imitation = O.autocontrast(imitation)
    # imitation = Image.fromarray(match_histograms(np.array(imitation), padded_to_imitate))
    imitation_np = np.array(imitation)
    if darker_foreground:
        imitation_np[np.where(imitation_np > to_imitate[to_imitate < 255].max())] = 255
    else:
        imitation_np[np.where(imitation_np < to_imitate[to_imitate > 0].min())] = 0
    imitation_np = match_histograms(imitation_np, padded_to_imitate)
    imitation = Image.fromarray(imitation_np)
    return imitation


def font_fit(to_imitate, fontpath, text, darker_foreground):
    try:
        corrs = []

        base_fontsize = estimate_font_size(fontpath, text, to_imitate)

        specs = []
        # for fontsize in [base_fontsize]:
        for fontsize in [max(8, base_fontsize - 1), base_fontsize, base_fontsize + 1]:
            imitation = imitate(to_imitate, fontpath, fontsize, text, darker_foreground)
            imitation_np = np.array(imitation)
            corr = match_template(
                np.pad(to_imitate, 10, mode="edge"), imitation_np
            ).max()

            specs.append((fontpath, fontsize))
            corrs.append(corr)

        return specs[np.argmax(corrs)], max(corrs)
    except Exception as e:
        print((fontpath, base_fontsize, text, e))
        return (fontpath, None), 0.0  # None


def fonts_fit(to_imitate, fontpaths, text, darker_foreground, progbar=True):
    fits = {}
    iter_ = tqdm(fontpaths) if progbar else fontpaths
    for fontpath in iter_:
        spec, corr = font_fit(to_imitate, fontpath, text, darker_foreground)
        fits[spec] = corr
    fits = pd.Series(fits)
    fits = fits.sort_values(ascending=False)
    return fits


def measure_text_color(
    image_array_gs, image_array_color, labeled, darker_foreground, char_ix
):
    labeled_color = np.tile(labeled[:, :, np.newaxis], reps=(1, 1, 3))

    to_imitate_gs = make_imitation_target(
        image_array_gs, labeled, darker_foreground, char_ix=char_ix
    )
    to_imitate_color = make_imitation_target(
        image_array_color, labeled_color, darker_foreground, char_ix=char_ix
    )

    fillcolors = to_imitate_color[np.where(to_imitate_gs == to_imitate_gs.min())]
    fillcolor = Counter([tuple(row) for row in fillcolors]).most_common()[0][0]
    return fillcolor


def measure_bg_color(image_array_color, labeled, palette_downscale=5):
    mask = np.where(labeled == 0)

    bgcolors = image_array_color[mask]

    bgcolors = palette_downscale * (bgcolors // palette_downscale)
    bgcolor_counts = Counter([tuple(row) for row in bgcolors])

    # npix = sum(bgcolor_counts.values())
    bgcolor = bgcolor_counts.most_common()[0][0]
    # bcolor_mean = np.mean(bgcolors, axis=0)
    # bcolor_median = np.median(bgcolors, axis=0)

    return bgcolor


def crop_grayscale_contrast_and_segment(
    image,
    box,
    show_intermediate=False,
):
    cropped = box.crop(image)

    image_color = cropped
    image_array_color = np.array(image_color)

    image = cropped.convert("L")
    image = O.autocontrast(image)
    image_array = np.array(image)

    seg, labeled, overlay, darker_foreground, q = region_based_segment(
        image_array, show_intermediate=show_intermediate
    )
    return (
        image_array_color,
        image_array,
        seg,
        labeled,
        overlay,
        darker_foreground,
        q,
        box,
    )


def crop_grayscale_contrast_and_segment_with_autopad(
    image,
    box,
    show_intermediate=False,
    max_tries=3,
    verbose=False,
):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    padrel_x = 1 / image.size[0]
    padrel_y = 1 / image.size[1]

    tries = 0
    done = False

    while not done:
        (
            image_array_color,
            image_array,
            seg,
            labeled,
            overlay,
            darker_foreground,
            q,
            _,
        ) = crop_grayscale_contrast_and_segment(
            image, box, show_intermediate=show_intermediate
        )

        needs_pad = {
            "left": (box.left > 0) and not (labeled[:, 0] == 0).all(),
            "right": (box.right < 1) and not (labeled[:, -1] == 0).all(),
            "top": (box.top > 0) and not (labeled[0, :] == 0).all(),
            "bottom": (box.bottom < 1) and not (labeled[-1, :] == 0).all(),
        }
        vprint(f"try {tries+1}/{max_tries}, needs_pad={repr(needs_pad)}")

        if not any(list(needs_pad.values())):
            done = True
            vprint(f"try {tries+1}/{max_tries}: DONE")

        newbox = BBox4Point(
            left=max(0, box.left - padrel_x * needs_pad["left"]),
            right=min(1, box.right + padrel_x * needs_pad["right"]),
            top=max(0, box.top - padrel_y * needs_pad["top"]),
            bottom=min(1, box.bottom + padrel_y * needs_pad["bottom"]),
        )

        # vprint((box, newbox))
        box = newbox

        tries += 1

        if tries >= max_tries:
            done = True

    vprint()

    return (
        image_array_color,
        image_array,
        seg,
        labeled,
        overlay,
        darker_foreground,
        q,
        box,
    )


def font_match_pipeline(
    image,
    box,
    text,
    fontpaths,
    nchars=1,
    average_over_chars=True,
    max_tries=3,
    show_intermediate=False,
    show_imitation=False,
    progbar=False,
    verbose=False,
):
    (
        image_array_color,
        image_array,
        seg,
        labeled,
        overlay,
        darker_foreground,
        q,
        box,
    ) = crop_grayscale_contrast_and_segment_with_autopad(
        image,
        box,
        max_tries=max_tries,
        show_intermediate=show_intermediate,
        verbose=verbose,
    )

    for c in whitespace:
        text = text.replace(c, "")
    chars = text[:nchars]
    results = []
    results_fontcolors = Counter()

    try:
        color = measure_bg_color(image_array_color, labeled)
    except:
        color = None
    results_bgcolor = color

    for char_ix, c in enumerate(chars):
        to_imitate = make_imitation_target(
            image_array, labeled, darker_foreground, char_ix=char_ix
        )
        if to_imitate is None:
            continue

        result = fonts_fit(to_imitate, fontpaths, c, darker_foreground, progbar=progbar)
        results.append(result)

        try:
            color = measure_text_color(
                image_array, image_array_color, labeled, darker_foreground, char_ix
            )
        except:
            color = None
        results_fontcolors[color] += 1

        if show_imitation:
            fontpath, fontsize = result.index[0]
            imitation = imitate(to_imitate, fontpath, fontsize, c, darker_foreground)
            imitation_np = np.array(imitation)

            corr = match_template(np.pad(to_imitate, 10, mode="edge"), imitation_np)
            amax_corr = np.unravel_index(np.argmax(corr.ravel()), corr.shape)
            print(f"corr: {corr.max():.3f}")

            to_imitate_show = np.pad(to_imitate, 10, mode="edge")[
                amax_corr[0] : amax_corr[0] + imitation_np.shape[0],
                amax_corr[1] : amax_corr[1] + imitation_np.shape[1],
            ]

            imitation_np_show = imitation_np

            to_imitate_show = to_imitate_show[
                : to_imitate.shape[0] + 3, : to_imitate.shape[1] + 3
            ]
            imitation_np_show = imitation_np_show[
                : to_imitate.shape[0] + 3, : to_imitate.shape[1] + 3
            ]

            fig, axes = plt.subplots(1, 2, figsize=(6, 3))
            axes[0].imshow(to_imitate_show, cmap=plt.cm.gray)
            axes[1].imshow(imitation_np_show, cmap=plt.cm.gray)
            plt.show()

    if len(results) == 0:
        return None, None, None, box
    result_fontspecs = pd.concat(results, axis=1)  # .fillna(0)

    if average_over_chars:
        result_fontspecs = result_fontspecs.mean(axis=1).sort_values(ascending=False)
        results_fontcolors = results_fontcolors.most_common()[0][0]

    return result_fontspecs, results_fontcolors, results_bgcolor, box


def font_pipeline_on_aws_item(
    image,
    item,
    fontpaths,
    nchars=1,
    show_intermediate=False,
    show_imitation=False,
    progbar=False,
    verbose=False,
    average_over_chars=True,
    max_tries=3,
):
    box = BBox4Point.from_aws(item["Geometry"])
    item_text = item["DetectedText"]
    # item_im = bbox.crop(image)

    return font_match_pipeline(
        image,
        box,
        item_text,
        fontpaths,
        nchars=nchars,
        show_intermediate=show_intermediate,
        show_imitation=show_imitation,
        progbar=progbar,
        average_over_chars=average_over_chars,
        max_tries=max_tries,
        verbose=verbose,
    )


def font_match_pipeline_full_aws(
    image,
    text_detections,
    fontpaths,
    nchars=2,
    nwords=2,
    show_intermediate=False,
    show_imitation=False,
    progbar=False,
    max_tries=3,
    min_height=10,
    verbose=False,
):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    def _validate_height(item):
        h = image.size[1] * item["Geometry"]["BoundingBox"]["Height"]
        return h > min_height

    line_results = {}

    aws_id_to_item = {v["Id"]: v for v in text_detections}
    aws_parent_to_child_id = defaultdict(list)

    for id_, item in aws_id_to_item.items():
        if "ParentId" in item:
            aws_parent_to_child_id[item["ParentId"]].append(item["Id"])

    aws_lines_ids = [
        id_ for id_, item in aws_id_to_item.items() if item["Type"] == "LINE"
    ]

    iter_ = tqdm(aws_lines_ids) if progbar else aws_lines_ids
    for id_ in iter_:
        item = aws_id_to_item[id_]

        if id_ in aws_parent_to_child_id:
            # word pipeline
            result_fontspecs, results_fontcolors, results_bgcolor = [], None, None
            word_boxes = []

            vprint(f"{id_}: {len(aws_parent_to_child_id[id_])} children")
            for child_id in aws_parent_to_child_id[id_][:nwords]:
                child = aws_id_to_item[child_id]

                if not _validate_height(child):
                    continue

                (
                    this_result_fontspecs,
                    this_results_fontcolors,
                    this_results_bgcolor,
                    this_box,
                ) = font_pipeline_on_aws_item(
                    image,
                    child,
                    fontpaths,
                    nchars=nchars,
                    show_intermediate=show_intermediate,
                    show_imitation=show_imitation,
                    progbar=False,
                    average_over_chars=False,
                    max_tries=max_tries,
                    verbose=verbose,
                )

                if this_result_fontspecs is not None:
                    result_fontspecs.append(this_result_fontspecs)

                if results_fontcolors is None:
                    results_fontcolors = this_results_fontcolors
                else:
                    results_fontcolors.update(this_results_fontcolors)

                # TODO: improve?
                results_bgcolor = this_results_bgcolor

                word_boxes.append(this_box)

            # TODO: dry
            if len(result_fontspecs) > 0:
                result_fontspecs = pd.concat(result_fontspecs, axis=1)  # .fillna(0)
                result_fontspecs = result_fontspecs.mean(axis=1).sort_values(
                    ascending=False
                )

                results_fontcolors = results_fontcolors.most_common()[0][0]
                box = combine_bboxes(word_boxes)
            else:
                result_fontspecs = None
                box = None

            line_results[id_] = {
                "fontspecs": result_fontspecs,
                "fontcolors": results_fontcolors,
                "bgcolor": results_bgcolor,
                "text": item["DetectedText"],
                "box": box,  # BBox4Point.from_aws(item['Geometry']),
                "raw": item,
            }
        else:
            if not _validate_height(item):
                continue
            (
                result_fontspecs,
                results_fontcolors,
                results_bgcolor,
                box,
            ) = font_pipeline_on_aws_item(
                image,
                item,
                fontpaths,
                nchars=nchars,
                show_intermediate=show_intermediate,
                show_imitation=show_imitation,
                progbar=progbar,
                average_over_chars=True,
                max_tries=max_tries,
                verbose=verbose,
            )
            line_results[id_] = {
                "fontspecs": result_fontspecs,
                "fontcolors": results_fontcolors,
                "bgcolor": results_bgcolor,
                "text": item["Text"],
                "box": box,
                "raw": item,
            }
    return line_results


# TODO: move


def bbox_abs(bbox, im):
    abs_box = (
        im.size[0] * bbox.left,
        im.size[1] * bbox.top,
        im.size[0] * bbox.right,
        im.size[1] * bbox.bottom,
    )
    return abs_box


class BBox4Point:
    def __init__(self, left, top, right, bottom):
        self.left = left
        self.top = top
        self.right = right
        self.bottom = bottom

    def _intersect_h(self, bbox):
        return (bbox.right >= self.left) & (bbox.right <= self.right) | (
            self.right >= bbox.left
        ) & (self.right <= bbox.right)

    def _intersect_v(self, bbox):
        return (bbox.bottom >= self.top) & (bbox.bottom <= self.bottom) | (
            self.bottom >= bbox.top
        ) & (self.bottom <= bbox.bottom)

    def intersect(self, bbox):
        return self._intersect_h(bbox) and self._intersect_v(bbox)

    def crop(self, im):
        abs_box = bbox_abs(self, im)
        return im.crop(abs_box)

    def __repr__(self):
        s = "BBox4Point("
        s += f"left={self.left:.3f}, right={self.right:.3f}, "
        s += f"top={self.top:.3f}, bottom={self.bottom:.3f})"
        return s

    @staticmethod
    def from_aws(bbox):
        if "Geometry" in bbox:
            bbox = bbox["Geometry"]
        if "BoundingBox" in bbox:
            bbox = bbox["BoundingBox"]
        return BBox4Point(
            bbox["Left"],
            bbox["Top"],
            bbox["Left"] + bbox["Width"],
            bbox["Top"] + bbox["Height"],
        )


def combine_bboxes(bboxes):
    def _cast(l, attr, method):
        return method([getattr(e, attr) for e in l])

    return BBox4Point(
        left=_cast(bboxes, "left", min),
        right=_cast(bboxes, "right", max),
        top=_cast(bboxes, "top", min),
        bottom=_cast(bboxes, "bottom", max),
    )


def make_image_simple(text, fontpath="Arial", size=20, hpad=0, vpad=0):
    fnt = ImageFont.truetype(fontpath, size)

    hsize, vsize = fnt.getsize_multiline(text)
    hsize += hpad
    vsize += vpad
    im = Image.new("L", (hsize, vsize), (255,))
    d = ImageDraw.Draw(im)

    base_offset = -1 * np.array(fnt.getoffset(text)) + 1
    offset = np.array([0, 0])
    d.multiline_text(base_offset + offset, text, font=fnt, fill=(0,))
    return im
