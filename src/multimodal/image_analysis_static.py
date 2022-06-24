import re

IMAGE_DIR = "data/analysis_images/"

ACCEPTABLE_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif"}

AR_DETECT_TEXT_CONFIDENCE_THRESHOLD = 95

IMAGE_DELIMITER = "======="
IMAGE_DELIMITER_WHITESPACED = "\n=======\n"
IMAGE_URL_DELIMITER = "\n=======\n=======\n"

extract_image_text_regex = re.compile(r"=======\n(.*?)\n=======", flags=re.DOTALL)

escaped_delim = IMAGE_DELIMITER.encode('unicode_escape').decode()
escaped_delim_ws = IMAGE_DELIMITER_WHITESPACED.encode('unicode_escape').decode()
escaped_delim_url = IMAGE_URL_DELIMITER.encode('unicode_escape').decode()

imurl_imtext_regex = rf"({escaped_delim_url})?(?(1)(.+?))({escaped_delim_ws})(.+?)({escaped_delim}(?:\n|$))"


def PRE_V9_IMAGE_FORMATTER(image_text, *args, **kwargs):
    return "\n" + image_text + "\n"


def V9_IMAGE_FORMATTER(image_text, *args, **kwargs):
    return "\n" + IMAGE_DELIMITER_WHITESPACED + image_text + IMAGE_DELIMITER_WHITESPACED


def URL_PRESERVING_IMAGE_FORMATTER(image_text, url):
    return "\n" + IMAGE_URL_DELIMITER + url + IMAGE_DELIMITER_WHITESPACED + image_text + IMAGE_DELIMITER_WHITESPACED


def extract_image_texts_from_post_text(s):
    # print('extract_image_texts_from_post_text called')  # is this used somewhere? in a notebook? let's find out
    return extract_image_text_regex.findall(s)


def extract_image_texts_and_urls_from_post_text(s):
    # TODO: DRY (vs image_munging.find_text_images_and_sub_real_images)

    # 5 groups:
    #   1. IMAGE_URL_DELIMITER if present
    #   2. url if (1) present
    #   3. IMAGE_DELIMITER_WHITESPACED if present
    #   4. imtext if present
    #   5. IMAGE_DELIMITER if present

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


def remove_image_urls_and_captions_from_post_text(s, return_urls=False):
    escaped_delim_ws = IMAGE_DELIMITER_WHITESPACED.encode('unicode_escape').decode()
    escaped_delim_url = IMAGE_URL_DELIMITER.encode('unicode_escape').decode()

    imurl_segment_regex=rf"({escaped_delim_url})(.+?)({escaped_delim_ws})"

    scrubbed = re.sub(imurl_segment_regex, IMAGE_DELIMITER_WHITESPACED, s)

    if return_urls:
        entries = extract_image_texts_and_urls_from_post_text(s)
        urls = [e['url'] for e in entries]
        return scrubbed, urls

    return scrubbed


def normalize_tumblr_image_url(k):
    return k.partition('media.tumblr.com')[2].replace('_','/').partition('.')[0]


def normalize_imtext_from_corpus(imtext):
    return imtext.lstrip(" ").replace("\n ", "\n")


def fill_url_based_captions(
    s,
    normed_url_to_replacement,
    normed_imtext_to_url,
    on_unreplaceable_url='unknown',
    on_unmappable_imtext='unknown',
    disable_url_norm=False,
    verbose=False,
):
    # TODO: DRY (vs image_munging.find_text_images_and_sub_real_images)
    escaped_delim = IMAGE_DELIMITER.encode('unicode_escape').decode()
    escaped_delim_ws = IMAGE_DELIMITER_WHITESPACED.encode('unicode_escape').decode()
    escaped_delim_url=IMAGE_URL_DELIMITER.encode('unicode_escape').decode()

    # 5 groups:
    #   1. IMAGE_URL_DELIMITER if present
    #   2. url if (1) present
    #   3. IMAGE_DELIMITER_WHITESPACED if present
    #   4. imtext if present
    #   5. IMAGE_DELIMITER if present

    entries = []

    matched = [0]
    imtext_mapped = [0]
    imtext_unmappable = [0]

    url_replaced = [0]
    url_unreplaceable = [0]

    def _replace_url(url):
        normed_url = url if disable_url_norm else normalize_tumblr_image_url(url)

        url_replacement = normed_url_to_replacement(normed_url)

        if url_replacement is None:
            if isinstance(on_unreplaceable_url, str):
                url_replacement = on_unreplaceable_url
                url_unreplaceable[0] += 1
            else:
                raise ValueError(f"unreplaceable {repr(url)}, {repr(normed_url)}")
        else:
            url_replaced[0] += 1

        return url_replacement

    def _map_imtext_to_url(imtext):
        normed_imtext = normalize_imtext_from_corpus(imtext)
        url = normed_imtext_to_url(normed_imtext, verbose=verbose)

        if url is None:
            if isinstance(on_unmappable_imtext, str):
                url = on_unmappable_imtext
                imtext_unmappable[0] += 1
            else:
                raise ValueError(f"unmappable {repr(imtext)}")
        else:
            imtext_mapped[0] += 1
        return url

    def _ismatch():
        matched[0] += 1

    def _replace(match):
        _ismatch()

        url = match.group(2)
        imtext = match.group(4).rstrip("\n")
        imtext_pos = match.start(4)

        if url is not None:
            """
            Case 1: new format.  Have url.
            """
            url = url.strip(" ")
            url_replacement = _replace_url(url)
        else:
            """
            Case 1: new format.  Only have imtext.
            """
            normed_imtext = normalize_imtext_from_corpus(imtext)

            url = _map_imtext_to_url(imtext)
            url_replacement = _replace_url(url)

        full_repl = ""

        full_repl += IMAGE_URL_DELIMITER             # group 1
        full_repl += url_replacement                 # group 2
        full_repl += match.group(3)                      # group 3
        full_repl += match.group(4)                   # group 4
        full_repl += match.group(5)                 # group 5


        return full_repl

    filled = re.sub(
        imurl_imtext_regex,
        _replace,
        s,
        flags=re.DOTALL,
    )

    if verbose:
        print(f"fill_url_based_captions: got original {repr(s)}")
        print(f"fill_url_based_captions: made replacement {repr(filled)}")

    return filled, matched[0], imtext_mapped[0], imtext_unmappable[0], url_replaced[0], url_unreplaceable[0]


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
