"""aws rekognition for seeing images on tumblr"""
from typing import NamedTuple, Optional, Callable, List, Tuple
import os
import time
from io import BytesIO

from tqdm import tqdm
import boto3
import requests

from PIL import Image
from moviepy.editor import VideoFileClip

IMAGE_DIR = "analysis_images/"

rek = boto3.client('rekognition')


def url_to_frame_bytes(url: str, fps: float=1., max_frames: int=10) -> List[bytes]:
    try:
        name = url.rpartition("/")[-1]
        xtn = "." + name.rpartition(".")[-1]

        r = requests.get(url)
        if r.status_code != 200:
            print(f"encountered code {r.status_code} trying to get {url}")
            return []

        if xtn in {".png", ".jpg", ".jpeg"}:
            return [r.content]
        elif xtn == ".gif":
            return gif_bytes_to_frame_bytes(r.content, fps=fps, max_frames=max_frames)
        else:
            print(f"encountered unknown extension {xtn} in {url}")
            return []
    except Exception as e:
        print(f"encountered {e} trying to get {url}")
        return []

def gif_bytes_to_frame_bytes(b: bytes, fps: float=1., max_frames: int=10) -> List[bytes]:
    frame_bytes = []

    path = os.path.join(IMAGE_DIR, f"temp.gif")
    with open(path, "wb") as f:
        f.write(b)

    clip = VideoFileClip(path)
    n_frames = len(list(clip.iter_frames()))

    n_frames_to_save = clip.duration * fps
    if n_frames_to_save > max_frames:
        fps = int(max_frames/clip.duration)
        n_frames_to_save = clip.duration * fps

    frame_indices = list(range(0, int(n_frames_to_save)+1))

    clip.write_images_sequence(os.path.join(IMAGE_DIR, "temp_%04d.png"), fps=fps)

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

def execute_callspec(spec: CallSpec, b: bytes,  **postprocessor_kwargs) -> dict:
    try:
        response = spec.method(b, **spec.kwargs)

        raw_results = {k: response.get(k) for k in spec.response_keys}
        if spec.postprocessor is not None:
            results = spec.postprocessor(raw_results, **postprocessor_kwargs)
        else:
            results = raw_results
        return results
    except Exception as e:
        print(f"encountered {e}")
        return {}

def execute_callspecs(specs: List[CallSpec], b: bytes, sleep_time: float=0.33,
                      postprocessor_kwargs: List[dict]=None) -> dict:
    results = {}
    if postprocessor_kwargs is None:
        postprocessor_kwargs = [{} for _ in specs]

    for spec, kwargs in tqdm(zip(specs, postprocessor_kwargs)):
        results.update(execute_callspec(spec, b,  **kwargs))

        time.sleep(sleep_time)

    return results

def batch_execute_callspecs(specs: List[CallSpec], byte_list: List[bytes], sleep_time: float=0.33,
                            postprocessor_kwargs: List[dict]=None) -> dict:
    results = []

    for b in tqdm(byte_list):
        results_one = {}
        for spec, kwargs in tqdm(zip(specs, postprocessor_kwargs)):
            results_one.update(execute_callspec(spec, b,  **kwargs))

            time.sleep(sleep_time)
        results.append(results_one)

    return results

# rekognition callspecs

def labels_found(entry, threshold=0):
    if 'Labels' not in entry:
        return {}
    entry_labels = entry['Labels']
    return {"label_" + item.get('Name'): item.get('Confidence', 100) for item in entry_labels
            if item.get('Confidence', 100) >= threshold}

def text_lines_found(entry, threshold=0, corner_ignore_thresh=0.05):
    def _not_corner(item):
        bb = item.get("Geometry", {}).get("BoundingBox", {})
        top, left, width, height = bb.get("Top"), bb.get("Left"), bb.get("Width"), bb.get("Height")
        if top is None or left is None or width is None or height is None:
            return True  # default permissive
        right = left + width
        bottom = top + height
        corner_diagnostic_x = min(left, 1-right)
        corner_diagnostic_y = min(top, 1-bottom)
        return (corner_diagnostic_x > corner_ignore_thresh) or (corner_diagnostic_y > corner_ignore_thresh)

    if 'TextDetections' not in entry:
        return {}
    entry_text_lines = [item for item in entry['TextDetections'] if item.get("Type") == "LINE"]
    results = {"text_lines":
        [{
            "Confidence": item.get('Confidence', 100),
            "text": item.get("DetectedText")
            }
         for ix, item in enumerate(entry_text_lines)
         if item.get('Confidence', 100) >= threshold and _not_corner(item)]
        }
    # if len(results["text_lines"]) > 0:
    #     print(f"found\n\t{results['text_lines']}")
    #     print(f"raw:\n\t{entry_text_lines}")
    return results

def moderation_labels(entry, threshold=0):
    if 'ModerationLabels' not in entry:
        return {}
    entry_mod = entry['ModerationLabels']
    return {"moderation_" + item.get('Name'): item.get('Confidence', 100) for item in entry_mod
            if item.get('Confidence', 100) >= threshold}

def face_features(entry, threshold=0):
    def _handle_dict(d, prefix=""):
        confidence = None
        if 'Value' in d and d.get('Value') in [True, False]:
            confidence = d['Confidence'] if d.get("Value", True) else 100-d['Confidence']
            if confidence < threshold:
                return {}
            return {prefix: confidence}
        elif 'Confidence' in d and d['Confidence'] < threshold:
            return {}

        if 'Type' in d:
            return {prefix + "_" + d['Type']: d['Confidence']}

        if 'Value' in d:
            return {prefix + "_" + d['Value']: d['Confidence']}

        return {prefix + k: v for k, v in d.items()}

    if 'FaceDetails' not in entry:
        return {}
    entry_face = entry['FaceDetails']

    results = {}

    for ix, face in enumerate(entry_face):
        for attribute_key, attribute in face.items():
            if attribute_key in {'Landmarks', 'BoundingBox',}:
                continue
            if isinstance(attribute, dict):
                results.update(_handle_dict(attribute, prefix=f"face{ix}_" + attribute_key))
            elif isinstance(attribute, list):
                for sub_attribute in attribute:
                    results.update(_handle_dict(sub_attribute, prefix=f"face{ix}_" + attribute_key))
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
    method=lambda b, **kwargs: rek.detect_moderation_labels(Image={"Bytes": b}, **kwargs),
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

recognize_celebrities_spec = CallSpec(
    method=lambda b, **kwargs: rek.recognize_celebrities(Image={"Bytes": b}, **kwargs),
    kwargs={},
    response_keys=["CelebrityFaces"]
)

all_rek_specs = [
    detect_labels_spec, detect_faces_spec, detect_moderation_labels_spec,
    detect_text_spec, recognize_celebrities_spec
]

autoreponder_rek_specs = [
    detect_labels_spec, detect_faces_spec, detect_text_spec
]

only_text_rek_specs = [detect_text_spec]

autoreponder_rek_kwargs = [
    {"threshold": 90},  # labels
    {"threshold": 70},  # faces
    {"threshold": 80},  # text
    ]

only_text_rek_kwargs = [{"threshold": 80}]


# utils / putting things together

def collect_text(results: List[dict]) -> str:
    lines = []

    for entry in results:
        if "text_lines" in entry:
            entry_lines = [text_line["text"] for text_line in entry["text_lines"]]
            lines.append("\n".join(entry_lines))

    return "\n".join(lines)

def extract_text_from_url(url: str) -> str:
    frame_bytes = url_to_frame_bytes(url)
    results = batch_execute_callspecs(only_text_rek_specs, postprocessor_kwargs=only_text_rek_kwargs, byte_list=frame_bytes)
    return collect_text(results)
