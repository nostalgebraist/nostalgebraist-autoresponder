import re

IMAGE_DIR = "data/analysis_images/"

ACCEPTABLE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}

AR_DETECT_TEXT_CONFIDENCE_THRESHOLD = 95

IMAGE_DELIMITER = "======="
IMAGE_DELIMITER_WHITESPACED = "\n=======\n"

extract_image_text_regex = re.compile(r"=======\n(.*?)\n=======", flags=re.DOTALL)


def PRE_V9_IMAGE_FORMATTER(image_text):
    return "\n" + image_text + "\n"


def V9_IMAGE_FORMATTER(image_text):
    return "\n" + IMAGE_DELIMITER_WHITESPACED + image_text + IMAGE_DELIMITER_WHITESPACED


def extract_image_texts_from_post_text(s):
    return extract_image_text_regex.findall(s)
