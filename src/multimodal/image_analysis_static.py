import re

IMAGE_DIR = "data/analysis_images/"

ACCEPTABLE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}

AR_DETECT_TEXT_CONFIDENCE_THRESHOLD = 95

IMAGE_DELIMITER = "======="
IMAGE_DELIMITER_WHITESPACED = "\n=======\n"
IMAGE_URL_DELIMITER = "\n=======\n=======\n"

extract_image_text_regex = re.compile(r"=======\n(.*?)\n=======", flags=re.DOTALL)


def PRE_V9_IMAGE_FORMATTER(image_text, *args, **kwargs):
    return "\n" + image_text + "\n"


def V9_IMAGE_FORMATTER(image_text, *args, **kwargs):
    return "\n" + IMAGE_DELIMITER_WHITESPACED + image_text + IMAGE_DELIMITER_WHITESPACED


def URL_PRESERVING_IMAGE_FORMATTER(image_text, url):
    return "\n" + IMAGE_URL_DELIMITER + url + IMAGE_DELIMITER_WHITESPACED + image_text + IMAGE_DELIMITER_WHITESPACED


def extract_image_texts_from_post_text(s):
    print('extract_image_texts_from_post_text called')  # is this used somewhere? in a notebook? let's find out
    return extract_image_text_regex.findall(s)


def extract_image_texts_and_urls_from_post_text(s):
    # TODO: DRY (vs image_munging.find_text_images_and_sub_real_images)
    escaped_delim = IMAGE_DELIMITER.encode('unicode_escape').decode()
    escaped_delim_ws = IMAGE_DELIMITER_WHITESPACED.encode('unicode_escape').decode()
    escaped_delim_url=IMAGE_URL_DELIMITER.encode('unicode_escape').decode()

    # 5 groups:
    #   1. IMAGE_URL_DELIMITER if present
    #   2. url if (1) present
    #   3. IMAGE_DELIMITER_WHITESPACED if present
    #   3. imtext if present
    #   3. IMAGE_DELIMITER if present
    imurl_imtext_regex=rf"({escaped_delim_url})?(?(1)(.+?))({escaped_delim_ws})(.+?)({escaped_delim}\n)"

    entries = []

    for match in re.finditer(
        imurl_imtext_regex,
        s,
        flags=re.DOTALL,
    ):
        url = match.group(2)
        if url is not None:
            url = url.strip(" ")
        imtext = match.group(4).rstrip("\n")
        imtext_pos = match.start(4)

        entries.append(
            {
                "imtext": imtext,
                "imtext_pos": imtext_pos,
                "url": url
            }
        )

    return entries


def remove_image_urls_from_post_text(s, return_urls=False):
    escaped_delim_ws = IMAGE_DELIMITER_WHITESPACED.encode('unicode_escape').decode()
    escaped_delim_url=IMAGE_URL_DELIMITER.encode('unicode_escape').decode()

    imurl_segment_regex=rf"({escaped_delim_url})(.+?)({escaped_delim_ws})"

    scrubbed = re.sub(imurl_segment_regex, IMAGE_DELIMITER_WHITESPACED, s)

    if return_urls:
        entries = extract_image_texts_and_urls_from_post_text(s)
        urls = [e['url'] for e in entries]
        return scrubbed, urls

    return scrubbed


def xtract_res(url, verbose=False):
    ps = url.split('/')
    fp = ps[-1]
    fn, _, xtn = fp.partition('.')
    _, _, rez = fn.partition('_')
    try:
        return int(rez)
    except Exception as e:
        if verbose:
            print((e, e.args))
        pass
    rezp = ps[-2]
    if rezp.startswith('s') and 'x' in rezp:
        try:
            x, y = rezp[1:].split('x')
            return (int(x), int(y))
        except Exception as e:
            if verbose:
                print((e, e.args))
            pass
