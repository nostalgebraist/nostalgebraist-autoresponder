"""aws rekognition for seeing images on tumblr"""
from typing import NamedTuple, Optional, Callable, List
import os
import time
import pickle
import hashlib
from io import BytesIO

from tqdm.autonotebook import tqdm
import boto3
import urllib3

from PIL import Image
from moviepy.editor import VideoFileClip

from util.error_handling import LogExceptionAndSkip

IMAGE_DIR = "data/analysis_images/"

rek = boto3.client("rekognition")

ACCEPTABLE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}

AR_DETECT_TEXT_CONFIDENCE_THRESHOLD = 95

IMAGE_DELIMITER = "======="
IMAGE_DELIMITER_WHITESPACED = "\n=======\n"


def xtn_from_headers(
    response: urllib3.response.HTTPResponse,  # requests.models.Response
):
    return "." + response.headers.get("Content-Type", "").partition("/")[-1]


def url_to_frame_bytes(
    url: str, fps: float = 1.0, max_frames: int = 10, http=None
) -> List[bytes]:
    try:
        if http is None:
            http = urllib3.PoolManager()

        name = url.rpartition("/")[-1]
        xtn_from_url = "." + name.rpartition(".")[-1]

        r = http.request("GET", url)

        if r.status != 200:
            print(f"encountered code {r.status} trying to get {url}")
            return []

        xtn = xtn_from_url
        if xtn not in ACCEPTABLE_IMAGE_EXTENSIONS:
            xtn = xtn_from_headers(r)

        if xtn == ".gif":
            return gif_bytes_to_frame_bytes(r.data, fps=fps, max_frames=max_frames)
        elif xtn in ACCEPTABLE_IMAGE_EXTENSIONS:
            return [r.data]

        # if we reach this, nothing worked
        return []
    except Exception as e:
        print(f"encountered {e} trying to get {url}")
        return []


def gif_bytes_to_frame_bytes(
    b: bytes, fps: float = 1.0, max_frames: int = 10
) -> List[bytes]:
    frame_bytes = []

    path = os.path.join(IMAGE_DIR, f"temp.gif")
    with open(path, "wb") as f:
        f.write(b)

    clip = VideoFileClip(path)
    n_frames = len(list(clip.iter_frames()))

    n_frames_to_save = clip.duration * fps
    if n_frames_to_save > max_frames:
        fps = int(max_frames / clip.duration)
        n_frames_to_save = clip.duration * fps

    frame_indices = list(range(0, int(n_frames_to_save) + 1))

    clip.write_images_sequence(
        os.path.join(IMAGE_DIR, "temp_%04d.png"), fps=fps, verbose=False, logger=None
    )

    for ix in frame_indices:
        fn = os.path.join(IMAGE_DIR, f"temp_{ix:04d}.png")
        if os.path.exists(fn):
            frame_bytes.append(path_to_bytes(fn))

    # cleanup
    os.remove(path)
    for ix in range(n_frames):
        fn = os.path.join(IMAGE_DIR, f"temp_{ix:04d}.png")
        if os.path.exists(fn):
            os.remove(fn)

    return frame_bytes


def path_to_bytes(path: str) -> bytes:
    bio = BytesIO()

    im = Image.open(path)

    is_jpeg = path.lower().endswith(".jpg") or path.lower().endswith(".jpeg")
    im.save(bio, format="jpeg" if is_jpeg else "png")

    b = bio.getvalue()
    bio.close()
    return b


class CallSpec(NamedTuple):
    method: Callable
    kwargs: dict
    response_keys: List[str]
    postprocessor: Optional[Callable] = None


def execute_callspec(spec: CallSpec, b: bytes, **postprocessor_kwargs) -> dict:
    try:
        response = spec.method(b, **spec.kwargs)

        if spec.response_keys is not None:
            raw_results = {k: response.get(k) for k in spec.response_keys}
        else:
            raw_results = response
        if spec.postprocessor is not None:
            results = spec.postprocessor(raw_results, **postprocessor_kwargs)
        else:
            results = raw_results
        return results
    except Exception as e:
        print(f"encountered {e}")
        return {}


def execute_callspecs(
    specs: List[CallSpec],
    b: bytes,
    sleep_time: float = 0.33,
    postprocessor_kwargs: List[dict] = None,
) -> dict:
    results = {}
    if postprocessor_kwargs is None:
        postprocessor_kwargs = [{} for _ in specs]

    iter = (
        tqdm(zip(specs, postprocessor_kwargs))
        if len(specs) > 1
        else zip(specs, postprocessor_kwargs)
    )
    for spec, kwargs in iter:
        results.update(execute_callspec(spec, b, **kwargs))

        time.sleep(sleep_time)

    return results


def batch_execute_callspecs(
    specs: List[CallSpec],
    byte_list: List[bytes],
    sleep_time: float = 0.33,
    postprocessor_kwargs: List[dict] = None,
    verbose=True,
) -> dict:
    results = []

    if postprocessor_kwargs is None:
        postprocessor_kwargs = [{} for _ in specs]

    iter = tqdm(byte_list) if (verbose and len(byte_list) > 1) else byte_list
    for b in iter:
        results_one = {}
        iter = (
            tqdm(zip(specs, postprocessor_kwargs))
            if (verbose and len(specs) > 1)
            else zip(specs, postprocessor_kwargs)
        )
        for spec, kwargs in iter:
            results_one.update(execute_callspec(spec, b, **kwargs))

            time.sleep(sleep_time)
        results.append(results_one)

    return results


# rekognition callspecs


def labels_found(entry, threshold=0):
    if "Labels" not in entry:
        return {}
    entry_labels = entry["Labels"]
    return {
        "label_" + item.get("Name"): item.get("Confidence", 100)
        for item in entry_labels
        if item.get("Confidence", 100) >= threshold
    }


def text_lines_found(entry, threshold=95, corner_ignore_thresh=0.05):
    def _not_corner(item):
        bb = item.get("Geometry", {}).get("BoundingBox", {})
        top, left, width, height = (
            bb.get("Top"),
            bb.get("Left"),
            bb.get("Width"),
            bb.get("Height"),
        )
        if top is None or left is None or width is None or height is None:
            return True  # default permissive
        right = left + width
        bottom = top + height
        corner_diagnostic_x = min(left, 1 - right)
        corner_diagnostic_y = min(top, 1 - bottom)
        return (corner_diagnostic_x > corner_ignore_thresh) or (
            corner_diagnostic_y > corner_ignore_thresh
        )

    if "TextDetections" not in entry:
        return {}
    entry_text_lines = [
        item for item in entry["TextDetections"] if item.get("Type") == "LINE"
    ]
    results = {
        "text_lines": [
            {
                "Confidence": item.get("Confidence", 100),
                "text": item.get("DetectedText"),
                "BoundingBox": item.get("Geometry", {}).get("BoundingBox", {}),
            }
            for ix, item in enumerate(entry_text_lines)
            if item.get("Confidence", 100) >= threshold and _not_corner(item)
        ]
    }
    # if len(results["text_lines"]) > 0:
    #     print(f"found\n\t{results['text_lines']}")
    #     print(f"raw:\n\t{entry_text_lines}")
    return results


def moderation_labels(entry, threshold=0):
    if "ModerationLabels" not in entry:
        return {}
    entry_mod = entry["ModerationLabels"]
    return {
        "moderation_" + item.get("Name"): item.get("Confidence", 100)
        for item in entry_mod
        if item.get("Confidence", 100) >= threshold
    }


def face_features(entry, threshold=0):
    def _handle_dict(d, prefix=""):
        confidence = None
        if "Value" in d and d.get("Value") in [True, False]:
            confidence = (
                d["Confidence"] if d.get("Value", True) else 100 - d["Confidence"]
            )
            if confidence < threshold:
                return {}
            return {prefix: confidence}
        elif "Confidence" in d and d["Confidence"] < threshold:
            return {}

        if "Type" in d:
            return {prefix + "_" + d["Type"]: d["Confidence"]}

        if "Value" in d:
            return {prefix + "_" + d["Value"]: d["Confidence"]}

        return {prefix + k: v for k, v in d.items()}

    if "FaceDetails" not in entry:
        return {}
    entry_face = entry["FaceDetails"]

    results = {}

    for ix, face in enumerate(entry_face):
        for attribute_key, attribute in face.items():
            if attribute_key in {
                "Landmarks",
                "BoundingBox",
            }:
                continue
            if isinstance(attribute, dict):
                results.update(
                    _handle_dict(attribute, prefix=f"face{ix}_" + attribute_key)
                )
            elif isinstance(attribute, list):
                for sub_attribute in attribute:
                    results.update(
                        _handle_dict(sub_attribute, prefix=f"face{ix}_" + attribute_key)
                    )
            else:
                results.update({f"face{ix}_" + attribute_key: attribute})

    return results


detect_labels_spec = CallSpec(
    method=lambda b, **kwargs: rek.detect_labels(Image={"Bytes": b}, **kwargs),
    kwargs={"MaxLabels": 123},
    response_keys=["Labels"],
    postprocessor=labels_found,
)

detect_faces_spec = CallSpec(
    method=lambda b, **kwargs: rek.detect_faces(Image={"Bytes": b}, **kwargs),
    kwargs={"Attributes": ["ALL"]},
    response_keys=["FaceDetails"],
    postprocessor=face_features,
)

detect_moderation_labels_spec = CallSpec(
    method=lambda b, **kwargs: rek.detect_moderation_labels(
        Image={"Bytes": b}, **kwargs
    ),
    kwargs={},
    response_keys=["ModerationLabels"],
    postprocessor=moderation_labels,
)

detect_text_spec = CallSpec(
    method=lambda b, **kwargs: rek.detect_text(Image={"Bytes": b}, **kwargs),
    kwargs={},
    response_keys=["TextDetections"],
    postprocessor=text_lines_found,
)

detect_text_actually_raw_response_spec = CallSpec(
    method=lambda b, **kwargs: rek.detect_text(Image={"Bytes": b}, **kwargs),
    kwargs={},
    response_keys=["TextDetections"],  # other key is request http metadata
    postprocessor=None,
)

recognize_celebrities_spec = CallSpec(
    method=lambda b, **kwargs: rek.recognize_celebrities(Image={"Bytes": b}, **kwargs),
    kwargs={},
    response_keys=["CelebrityFaces"],
)

all_rek_specs = [
    detect_labels_spec,
    detect_faces_spec,
    detect_moderation_labels_spec,
    detect_text_spec,
    recognize_celebrities_spec,
]

autoreponder_rek_specs = [detect_labels_spec, detect_faces_spec, detect_text_spec]

only_text_rek_specs = [detect_text_spec]

autoreponder_rek_kwargs = [
    {"threshold": 90},  # labels
    {"threshold": 70},  # faces
    {"threshold": 80},  # text
]

only_text_rek_kwargs_original_flavor = [
    {"threshold": 80}
]  # used for corpora through v10, etc
only_text_rek_kwargs = [{"threshold": AR_DETECT_TEXT_CONFIDENCE_THRESHOLD}]


# utils / putting things together


def collect_text(results: List[dict], deduplicate=True, return_raw=False) -> str:
    lines = []

    usable_entries = []
    for entry in results:
        if "text_lines" in entry:
            usable_entries.append(entry)
            entry_lines = [text_line["text"] for text_line in entry["text_lines"]]
            lines.append("\n".join(entry_lines))

    lines_all = lines
    if deduplicate:
        lines_dedup = []
        for l in lines:
            if l not in lines_dedup:
                lines_dedup.append(l)
        lines = lines_dedup

    if return_raw:
        collected_text = "\n".join(lines)
        if len(usable_entries) != len(lines_all):
            print(
                f"warning: len(usable_entries)={len(usable_entries)} but len(lines_all)={len(lines_all)}"
            )
            print(f"usable_entries: {usable_entries}\nlines_all: {lines_all}")
        raw = []
        for e, l in zip(usable_entries, lines_all):
            d = {"line": l}
            d.update(e)
            raw.append(d)
        return collected_text, raw
    return "\n".join(lines)


def PRE_V9_IMAGE_FORMATTER(image_text):
    return "\n" + image_text + "\n"


def V9_IMAGE_FORMATTER(image_text):
    return "\n" + IMAGE_DELIMITER_WHITESPACED + image_text + IMAGE_DELIMITER_WHITESPACED


def format_extracted_text(image_text, image_formatter=V9_IMAGE_FORMATTER, verbose=False):
    if verbose and len(image_text) > 0:
        print(f"analysis text is\n{image_text}\n")
    if len(image_text) > 0:
        return image_formatter(image_text)
    return ""


class ImageAnalysisCache:
    def __init__(self, path="data/image_analysis_cache.pkl.gz", cache=None, hash_to_url=None):
        self.path = path
        self.cache = cache
        self.hash_to_url = hash_to_url

        if self.cache is None:
            self.cache = dict()

        if self.hash_to_url is None:
            self.hash_to_url = dict()

    @staticmethod
    def _get_text_from_cache_entry(entry, deduplicate=True):
        """deal with the various formats i've saved AWS responses in"""
        result = entry

        if isinstance(result, list):
            result = collect_text(result, deduplicate=deduplicate)

        if isinstance(result, dict):
            result = result['line']

        return result

    @staticmethod
    def _download_and_hash(
            url: str,
            http=None,
            verbose=True,
            downsize_to=[640, 540],
    ):
        if http is None:
            http = urllib3.PoolManager()

        url_ = url
        r_pre = http.request("GET", url_, preload_content=False)
        nbytes_ = int(r_pre.headers.get("Content-Length", -1))
        r_pre.release_conn()


        if downsize_to is not None and nbytes_ > 0:
            try:
                for downsize in downsize_to:
                    seg, _, xtn = url_.rpartition(".")
                    seg2, _, orig_size = seg.rpartition("_")
                    newurl = seg2 + "_" + str(downsize) + "." + xtn

                    r_pre = http.request("GET", newurl, preload_content=False)
                    nbytes_new = int(r_pre.headers["Content-Length"])
                    r_pre.release_conn()

                    if r_pre.status != 200:
                        raise ValueError

                    if nbytes_new < nbytes_:
                        if verbose:
                            print(f"{url_}\n\t--> {newurl}")
                        url_ = newurl
                        nbytes_ = nbytes_new
            except:
                if verbose:
                    pass  # print(f"couldn't downsize: {url}")

        frame_bytes = url_to_frame_bytes(url_, http=http)
        frame_hashes = [hashlib.md5(b).hexdigest() for b in frame_bytes]
        content_hash = "".join(frame_hashes)  # should be different at different sample rate for gif

        return frame_bytes, content_hash

    @staticmethod
    def _extract_text_from_bytes(
            frame_bytes: List[bytes],
            deduplicate=True,
            sleep_time: float = 0.1,
            verbose=True,
            return_raw=True,
            xtra_raw=False,
    ):
        return_raw = return_raw or xtra_raw
        specs = (
            [detect_text_actually_raw_response_spec] if xtra_raw else only_text_rek_specs
        )
        results = batch_execute_callspecs(
            specs,
            postprocessor_kwargs=only_text_rek_kwargs,
            byte_list=frame_bytes,
            sleep_time=sleep_time,
            verbose=verbose,
        )

        # TODO: (cleanup) make this return signature less bad
        # TODO: (cleanup) make this file less bad in general!
        if xtra_raw:
            return results

        _, raw = collect_text(results, return_raw=return_raw, deduplicate=deduplicate)
        return raw

    def extract_and_format_text_from_url(
        self,
        url: str,
        image_formatter=V9_IMAGE_FORMATTER,
        verbose=False,
    ):
        def vprint(*args, **kwargs):
            if verbose:
                print(*args, **kwargs)

        # TODO: integrate downsizing
        if url not in self.cache:
            vprint(f"url NOT in cache:\n\t{url}\n")
            entry = None

            try:
                frame_bytes, content_hash = self._download_and_hash(
                    url,
                    verbose=verbose
                )
            except urllib3.exceptions.RequestError:
                entry = ""
                content_hash = None

            if self.hash_to_url.get(content_hash):
                url_with_existing_hash = self.hash_to_url[content_hash]
                vprint(f"hash {content_hash} in hash_to_url, url is\n\t{url_with_existing_hash}\n")
                if url_with_existing_hash in self.cache:
                    vprint(f"\turl IN cache:\n\t\t{url}\n")
                    entry = self.cache[url_with_existing_hash]
                else:
                    vprint(f"\turl NOT in cache:\n\t\t{url}\n")

            if entry is None:
                vprint(f"calling rek for {url}")
                self.hash_to_url[content_hash] = url
                entry = self._extract_text_from_bytes(frame_bytes)
            self.cache[url] = entry
        else:
            vprint(f"url IN cache:\n\t{url}\n")

        cached_text = ""
        with LogExceptionAndSkip(f"retrieving {repr(url)} from cache"):
            cached_text = self._get_text_from_cache_entry(self.cache[url])

        formatted_text = format_extracted_text(cached_text, image_formatter=image_formatter, verbose=verbose)
        return formatted_text

    def save(self, verbose=True, do_backup=True):
        data = {"cache": self.cache, "hash_to_url": self.hash_to_url}
        with open(self.path, "wb") as f:
            pickle.dump(data, f)
        if do_backup:
            # TODO: better path handling
            with open(self.path[: -len(".pkl.gz")] + "_backup.pkl.gz", "wb") as f:
                pickle.dump(data, f)
        if verbose:
            print(
                f"saved image analysis cache with lengths cache={len(self.cache)}, hash_to_url={len(self.hash_to_url)}"
            )

    @staticmethod
    def load(
        path: str = "data/image_analysis_cache.pkl.gz", verbose=True
    ) -> "ImageAnalysisCache":
        cache = None
        if os.path.exists(path):
            with open(path, "rb") as f:
                data = pickle.load(f)

            try:
                cache = data["cache"]
                hash_to_url = data["hash_to_url"]
            except KeyError:
                # first time load
                cache = data
                hash_to_url = dict()

            if verbose:
                print(f"loaded image analysis cache with lengths cache={len(cache)}, hash_to_url={len(hash_to_url)}")
        else:
            print(f"initialized image analysis cache")
        loaded = ImageAnalysisCache(path, cache, hash_to_url)
        return loaded


# development helpers
def bbox_show(obj_dict, im):
    bbox = obj_dict["BoundingBox"]
    abs_box = (
        im.size[0] * bbox["Left"],
        im.size[1] * bbox["Top"],
        im.size[0] * (bbox["Left"] + bbox["Width"]),
        im.size[1] * (bbox["Top"] + bbox["Height"]),
    )
    im.crop(abs_box).show()


def labels_bbox_show(results, im):
    for l in results["Labels"]:
        for inst in l["Instances"]:
            print((l.get("Name"), inst), end="\n\n")
            bbox_show(inst, im)
            input()


def faces_bbox_show(results, im):
    for face in results["FaceDetails"]:
        print(face, end="\n\n")
        bbox_show(face, im)
        input()


raw_detect_faces_spec = CallSpec(
    method=lambda b, **kwargs: rek.detect_faces(Image={"Bytes": b}, **kwargs),
    kwargs={"Attributes": ["ALL"]},
    response_keys=["FaceDetails"],
    postprocessor=None,
)

raw_detect_labels_spec = CallSpec(
    method=lambda b, **kwargs: rek.detect_labels(Image={"Bytes": b}, **kwargs),
    kwargs={"MaxLabels": 123},
    response_keys=["Labels"],
    postprocessor=None,
)
