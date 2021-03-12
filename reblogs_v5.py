"""mysterious hacky html parsing that somehow (mostly?) successfully handles tumblr's weird pre-NPF markup format"""
Q_CHAR = "会"
A_CHAR = "域"
T_CHAR = "职"

UNAME_CHAR = "友"
ORIG_POST_CHAR = "翰"
START_DUMMY = "⭒"
V10_ASK_CHAR = "要"

V10_CHARS_TO_LEGACY_CHARS = {V10_ASK_CHAR: Q_CHAR}

ALWAYS_USE_A_CHAR_OPERATIONAL = True
TRY_LINKS_FOR_IMAGES = False
INCLUDE_HREF_FOR_A = True

EOT_FULL = "<|endoftext|>"

RECURSE_INTO = {
    "p",
    "blockquote",
    "div",
    "em",
    "i",
    "b",
    "u",
    "strong",
    "h2",
    "figure",
}
INCLUDE_TAGNAME = {
    "blockquote",
    "em",
    "i",
    "b",
    "u",
    "strong",
    "h2",
}
INCLUDE_VERBATIM = {"li", "ul", "ol"}
NEWLINE_AFTER = {
    "blockquote",
    "h2",
}
DOUBLE_NEWLINE_AFTER = {"p", "br", "img"}
AVOID = {
    "header",
}
USE_IMAGE_ANALYSIS = {"img"}

from string import whitespace
from itertools import product
from functools import partial

import bs4

from image_analysis import (
    extract_and_format_text_from_url,
    V9_IMAGE_FORMATTER,
    ImageAnalysisCache,
)


def IMAGE_ANALYSIS_FN(
    elem, image_formatter=V9_IMAGE_FORMATTER, image_analysis_cache=None, verbose=True
):
    url_attr = "href" if elem.name == "a" else "src"

    if elem.attrs.get(url_attr) is None:
        return None

    if image_analysis_cache is not None:
        return image_analysis_cache.extract_and_format_text_from_url(
            elem.attrs.get(url_attr), image_formatter=image_formatter
        )
    return extract_and_format_text_from_url(
        elem.attrs.get(url_attr), image_formatter=image_formatter
    )


def lprint(s, prefix=""):
    print(f"{prefix}{s}", end=f"\n\n{prefix}---------\n\n")


def map_uname(uname: str, uname_config: str = "frank"):
    uname_map = {}
    if uname_config == "frank":
        uname_map = {
            "nostalgebraist": "nostalgebraist-my-father",
            "nostalgebraist-autoresponder": "nostalgebraist",
        }
    elif uname_config == "frank_v5_train":
        uname_map = {
            "nostalgebraist": "nostalgebraist-autoresponder",
            "nostalgebraist-autoresponder": "nostalgebraist",
            "aprilwitching-deactivated201808": "aprilwitching",
        }
    elif uname_config == "frank_v10_train":
        uname_map = {
            "nostalgebraist": "Frank",
            "aprilwitching-deactivated201808": "aprilwitching",
        }
    elif uname_config == "frank_v5_operate":
        uname_map = {
            "nostalgebraist": "nostalgebraist-my-father",
            "nostalgebraist-autoresponder": "Frank",
        }
    elif uname_config == "frank_v10_operate":
        uname_map = {
            "nostalgebraist": "nostalgebraist",
            "nostalgebraist-autoresponder": "Frank",
        }
    elif uname_config == "frank_v10_1_operate":
        uname_map = {}

    return uname_map.get(uname, uname)


def make_text_processor_maps(uname_config: str = "frank"):
    if uname_config in ["frank_v5_train", "frank_v10_train"]:
        maps = [("nostalgebraist", "nostalgebraist-autoresponder")]
        maps = maps + [(m[0].capitalize(), m[1].capitalize()) for m in maps]

        punct_ws_toks = [":", ">", ".", " ", "\n", ",", "!", ";", "…", START_DUMMY]
        punct_ws_maps_base = [("rob", "frank"), ("robert", "francis")]
        punct_ws_maps_base = punct_ws_maps_base + [
            (m[0].capitalize(), m[1].capitalize()) for m in punct_ws_maps_base
        ]

        punct_ws_maps = []
        for m in punct_ws_maps_base:
            for t1, t2 in product(punct_ws_toks, punct_ws_toks):
                punct_ws_maps.append(
                    (
                        t1 + m[0] + t2,
                        t1 + m[1] + t2,
                    )
                )

        maps = maps + punct_ws_maps
        return maps
    else:
        return []


def text_processor(text: str, maps):
    for m in maps:
        orig = text
        text = text.replace(m[0], m[1])
        if m[0].startswith(START_DUMMY):
            text = (START_DUMMY + text).replace(m[0], m[1]).lstrip(START_DUMMY)
        if text != orig:
            print(f"text_processor: {orig} -> {text}")
    return text


def is_whitespace_string(elem):
    if not isinstance(elem, bs4.element.NavigableString):
        return False
    return all([(c in whitespace) and (c != " ") for c in str(elem)])


def show_bs4_elem(elem, prefix=""):
    if isinstance(elem, bs4.element.Tag):
        return f"{type(elem)}: {elem.name})\n\n{prefix}{elem}"
    elif isinstance(elem, bs4.element.NavigableString):
        if is_whitespace_string(elem):
            return "[whitespace string]"
        else:
            return f"{type(elem)})\n\n{prefix}'{str(elem)}'"
    else:
        raise ValueError(f"type {type(elem)}")


def _tags_from_footer(footer):
    true_tags = []
    note_count = None
    for elem in footer:
        if isinstance(elem, bs4.element.NavigableString):
            if " note" in str(elem):
                try:
                    note_count = int(str(elem).partition(" note")[0].split(" ")[-1])
                except ValueError as e:
                    print(elem)
                    raise e
        elif elem.text.startswith("#"):
            true_tags.append(elem.text + " ")
    return true_tags, note_count


def _format_asking_title(elem, uname_config):
    asker_name, _, question = elem.text.partition(" asked:")
    ask_char = V10_ASK_CHAR
    return [
        UNAME_CHAR,
        map_uname(asker_name, uname_config),
        ask_char,
        "\n",
        question.lstrip(" "),
    ]


def _get_unname_from_a(elem, in_h2, is_first, uname_config):
    uname = None
    href = elem.attrs.get("href", "")
    if (
        len(
            set(elem.attrs.get("class", set())).intersection(
                {"tumblr_blog", "username", "js-hover-trigger-TumblelogPopover"}
            )
        )
        > 0
    ):
        uname = elem.text
    elif (
        (href.endswith("tumblr.com/"))
        or ("tumblelog" in elem.attrs.get("class", set()))
    ) and is_first:
        uname = elem.text.lstrip("@")
        # print(f"via is_first=True, extracted uname {repr(uname)} from tumblelog tag {elem}")
    elif ".tumblr.com" in href and in_h2:
        uname = href.partition(".tumblr.com")[0]
        uname = uname.partition("://")[2]
    if (uname is not None) and len(uname) == 0:
        return None
    return map_uname(uname, uname_config)


def _process_elem(
    elem,
    uname_config,
    text_processor_maps,
    uname_levels=[""],
    quote_level=0,
    skip_colon=False,
    in_h2=False,
    is_first=False,
    reblog=False,
    debug=True,
    do_image_analysis=True,
    get_image_urls=False,
    reply_post_next_a=False,
    reply_post_url=None,
    user_defined_image_analysis=IMAGE_ANALYSIS_FN,
    user_defined_image_formatter=V9_IMAGE_FORMATTER,
):
    if debug:
        print(f"\t! for this {elem.name}, reblog={reblog}, is_first={is_first}")
    text_units = []
    meta = {
        "reblog": reblog,
        "tags": False,
        "is_quotes": False,
        "uname_levels": uname_levels,
        "quote_level": quote_level,
        "skip_colon": skip_colon,
        "ask_done": False,
        "in_h2": in_h2,
        "is_first": is_first,
        "image_urls": set(),
        "reply_post_next_a": reply_post_next_a,
        "reply_post_url": reply_post_url,
    }

    if is_whitespace_string(elem):
        return text_units, meta

    if meta["skip_colon"]:
        if isinstance(elem, bs4.element.NavigableString):
            if str(elem).strip(whitespace) == ":":
                meta["skip_colon"] = False
                return text_units, meta

    if isinstance(elem, bs4.element.NavigableString):
        if "replied to your" in str(elem):
            meta["reply_post_next_a"] = True
            return [], meta
        else:
            return [text_processor(str(elem), text_processor_maps)], meta

    if not isinstance(elem, bs4.element.Tag):
        print(f"warning: skipping element of type {type(elem)}")

    if elem.name in AVOID:
        return text_units, meta

    if elem.name == "h2" and " asked:" in elem.text:
        elem_text_units = _format_asking_title(elem, uname_config)
        text_units.extend(elem_text_units)
        meta["ask_done"] = True
    elif elem.name in RECURSE_INTO:
        if debug:
            print(f"recursing from {elem.name}, is_first={meta['is_first']}")
        reblog_for_blockquotes = False
        for ix2, elem2 in enumerate(elem):
            if debug:
                print(f"\trecursing into {elem2.name} ({ix2})")
            next_quote_level = (
                meta["quote_level"] + 1
                if elem.name == "blockquote"
                else meta["quote_level"]
            )
            # meta["reblog"] and ix2 == 1
            recur_text_units, recur_meta = _process_elem(
                elem2,
                uname_config,
                text_processor_maps,
                debug=debug,
                uname_levels=meta["uname_levels"],
                quote_level=next_quote_level,
                skip_colon=meta["skip_colon"],
                in_h2=meta["in_h2"] or elem.name == "h2",
                is_first=meta["is_first"] and len("".join(text_units)) == 0,
                reblog=reblog_for_blockquotes,
                do_image_analysis=do_image_analysis,
                get_image_urls=get_image_urls,
                reply_post_next_a=meta["reply_post_next_a"],
                reply_post_url=meta["reply_post_url"],
                user_defined_image_analysis=user_defined_image_analysis,
                user_defined_image_formatter=user_defined_image_formatter,
            )
            text_units.extend(recur_text_units)
            if recur_meta["reblog"]:
                reblog_for_blockquotes = True
            if not recur_meta["reblog"] and elem2.name is not None:
                reblog_for_blockquotes = False
            meta["reblog"] = reblog_for_blockquotes
            meta["skip_colon"] = recur_meta["skip_colon"]
            meta["image_urls"].update(recur_meta["image_urls"])
            meta["reply_post_next_a"] = recur_meta["reply_post_next_a"]
            if recur_meta["reply_post_url"] is not None:
                meta["reply_post_url"] = recur_meta["reply_post_url"]

            if debug:
                print(
                    f"\t(recur) {elem2.name} ({ix2}) got: reblog_for_blockquotes={reblog_for_blockquotes}, uname_levels={uname_levels}, quote_level={quote_level}"
                )
                print(f"\trecur_meta={recur_meta}")
                print(f"\ttext_units={text_units}")
                print(f"\t(recur) {elem2.name} ({ix2}) done\n")

    if elem.name == "footer":
        meta["tags"] = True
        tags, note_count = _tags_from_footer(elem)
        text_units.extend(tags)

        meta["is_quotes"] = any([t.rstrip(" ") == "#quotes" for t in tags])
        meta["note_count"] = note_count

    elif elem.name == "a":
        reblog_uname = _get_unname_from_a(
            elem, meta["in_h2"], meta["is_first"], uname_config
        )
        if reblog_uname is not None:
            name_unit = ""
            uname_char = UNAME_CHAR
            name_unit = name_unit + uname_char
            name_unit = name_unit + reblog_uname
            meta["reblog"] = True

            me_you_char = (
                A_CHAR
                if reblog_uname
                == map_uname("nostalgebraist-autoresponder", uname_config)
                else Q_CHAR
            )
            name_unit = name_unit + me_you_char
            if (
                ALWAYS_USE_A_CHAR_OPERATIONAL
                and map_uname(reblog_uname, uname_config) == "Frank"
            ):
                name_unit = A_CHAR
            text_units.append(name_unit)
            meta["uname_levels"].append(name_unit)
            meta["skip_colon"] = True
        elif meta["reply_post_next_a"]:
            meta["reply_post_next_a"] = False
            meta["reply_post_url"] = elem.attrs.get("href", None)
        else:
            image_units_for_a = []
            if TRY_LINKS_FOR_IMAGES and do_image_analysis:
                image_text = user_defined_image_analysis(
                    elem, image_formatter=user_defined_image_formatter
                )
                # TODO: DRY
                if image_text is not None:
                    image_units_for_a.append(image_text)
                    if get_image_urls:
                        meta["image_urls"].add(elem.attrs.get("href"))
            if len(image_units_for_a) > 0:
                text_units.extend(image_units_for_a)
            else:
                no_href_classes = {
                    "tmblr-truncated-link",
                    "tumblr_blog",
                    "notification_target",
                    "post_info_link",
                    "tumblelog",
                }
                if (
                    INCLUDE_HREF_FOR_A
                    and len(
                        set(elem.attrs.get("class", set())).intersection(
                            no_href_classes
                        )
                    )
                    == 0
                    and elem.attrs.get("href") is not None
                ):
                    # if elem.attrs.get('class') is not None:
                    #     print(f"a href feature: {repr(elem)}")
                    href = elem.attrs.get("href")
                    unit = f'<a href="{href}">{elem.text}</a>'
                    text_units.append(unit)
                else:
                    text_units.append(text_processor(elem.text, text_processor_maps))

    elif elem.name in INCLUDE_VERBATIM:
        text_units.append(elem.decode())
    elif elem.name in USE_IMAGE_ANALYSIS and do_image_analysis:
        image_text = user_defined_image_analysis(
            elem, image_formatter=user_defined_image_formatter
        )
        if image_text is not None:
            text_units.append(image_text)
        if get_image_urls:
            meta["image_urls"].add(elem.attrs.get("src"))
    elif elem.name not in RECURSE_INTO:
        text_units.append(text_processor(elem.text, text_processor_maps))

    if elem.name in INCLUDE_TAGNAME and not (reblog or meta["ask_done"]):
        text_units = [f"<{elem.name}>"] + text_units + [f"</{elem.name}>"]

    if elem.name in NEWLINE_AFTER:
        if len(text_units) > 0:
            text_units[-1] += "\n"
        else:
            text_units += "\n"
    elif elem.name in DOUBLE_NEWLINE_AFTER:
        if len(text_units) > 0:
            text_units[-1] += "\n\n"
        else:
            text_units += "\n\n"

    if elem.name == "blockquote" and reblog:
        try:
            text_units.append(meta["uname_levels"][meta["quote_level"]])
            if debug:
                print(
                    f"APPENDING {meta['uname_levels'][meta['quote_level']]} with quote_level {meta['quote_level']}"
                )
                print(f"meta['uname_levels']: {meta['uname_levels']}\n")
        except IndexError:
            print("indexerr")
            pass

    return text_units, meta


def process_post(
    soup,
    debug=False,
    use_article=True,
    uname_config="frank_v10_1_operate",
    do_image_analysis=True,
    get_image_urls=False,
    user_defined_image_analysis=IMAGE_ANALYSIS_FN,
    user_defined_image_formatter=V9_IMAGE_FORMATTER,
    image_analysis_cache: ImageAnalysisCache = None,
    V10=True,
):
    user_defined_image_analysis = partial(
        user_defined_image_analysis, image_analysis_cache=image_analysis_cache
    )
    text_processor_maps = make_text_processor_maps(uname_config)

    text_units = []
    ask_prefix_text_units = []

    uname_levels = [""]
    quote_level = 0
    skip_colon = False

    reblog_block_started = False
    main_post_marked = False

    reblog = False
    reply_post_next_a = False
    reply_post_url = None

    post_metadata = {
        "is_quotes": False,
        "is_ask": False,
        "is_reblog": False,
        "is_orig": False,
        "image_urls": set(),
        "reply_post_url": reply_post_url,
    }

    soup_iter = soup.article if use_article else soup.body
    for ix, elem in enumerate(soup_iter):
        if debug:
            print(f"({ix}) processing: {elem.name}")
            lprint("")

        is_first = len("".join(text_units)) == 0
        elem_text_units, elem_meta = _process_elem(
            elem,
            uname_config,
            text_processor_maps,
            uname_levels=uname_levels,
            quote_level=quote_level,
            skip_colon=skip_colon,
            is_first=is_first,
            debug=debug,
            reblog=reblog,
            do_image_analysis=do_image_analysis,
            get_image_urls=get_image_urls,
            reply_post_url=reply_post_url,
            reply_post_next_a=reply_post_next_a,
            user_defined_image_analysis=user_defined_image_analysis,
            user_defined_image_formatter=user_defined_image_formatter,
        )
        if debug:
            print(
                f"({ix} {elem.name}) got: uname_levels={uname_levels}, quote_level={quote_level}, text_units=\n"
            )
            for _ in elem_text_units:
                lprint(_.replace("\n", "\\n"), prefix="\t")
            print(elem_meta)
            print(f"({ix}) done\n")

        uname_levels = elem_meta["uname_levels"]
        quote_level = elem_meta["quote_level"]
        skip_colon = elem_meta["skip_colon"]
        reblog = elem_meta["reblog"]
        reply_post_next_a = elem_meta["reply_post_next_a"]
        reply_post_url = elem_meta["reply_post_url"]

        post_metadata["reply_post_url"] = reply_post_url

        # tags
        if elem_meta["tags"]:
            text_units.append(T_CHAR)
            post_metadata["is_quotes"] = elem_meta["is_quotes"]

        if "note_count" in elem_meta:
            post_metadata["note_count"] = elem_meta["note_count"]

        text_units.extend(elem_text_units)

        if get_image_urls:
            post_metadata["image_urls"].update(elem_meta["image_urls"])

        # handle bug processing reblogged asks when the ask text is in the html body
        # as in the scraper corpus (but not the API payloads)
        if (
            (elem_meta["reblog"] == True)
            and (post_metadata["is_ask"])
            and (main_post_marked)
        ):
            main_post_marked = False
            text_units = [u for u in text_units if u != A_CHAR]

        # reblog stuff
        if elem_meta["reblog"] == True:
            reblog_block_started = True
            post_metadata["is_reblog"] = True
        if reblog_block_started and elem.name == "blockquote" and not main_post_marked:
            text_units.append(A_CHAR)
            main_post_marked = True
        if elem_meta["ask_done"] == True and not main_post_marked:
            ask_prefix_text_units = [u for u in text_units]
            text_units = []

            text_units.append(A_CHAR)
            main_post_marked = True
            post_metadata["is_ask"] = True

        if debug:
            print(f"text units so far:\n\t{[ask_prefix_text_units, text_units]}")

    # clean up initial blockquote uname junk
    initial_uname_units = []
    initial_title_units = []
    in_title = False
    for unit in text_units:
        if not isinstance(unit, str):
            print(f"bad unit: {repr(unit)}")
        if unit.startswith("</h2>"):
            in_title = False
            initial_title_units.append(unit)
        elif unit.startswith("<h2>") and len(initial_uname_units) == 0:
            in_title = True
            initial_title_units.append(unit)
        elif in_title:
            initial_title_units.append(unit)
        elif unit.rstrip("\n").startswith(UNAME_CHAR) or (
            ALWAYS_USE_A_CHAR_OPERATIONAL and unit.rstrip("\n") == A_CHAR
        ):
            initial_uname_units.append(unit)
        elif not ((unit == "\n") or (unit == "\n\n")):
            # we continue past newlines bc there might be more usernames
            break
        else:
            break
    if len(initial_uname_units) > 1:
        if debug:
            print(f"got uname units: {initial_uname_units}")
        n_title = len(initial_title_units)
        if debug:
            print(
                f"stripping units: {text_units[n_title:(n_title+len(initial_uname_units)-1)]}"
            )
        text_units = (
            text_units[:n_title]
            + text_units[(n_title + len(initial_uname_units) - 1) :]
        )
        if debug:
            print(f"remaining units: {text_units}")
            print()

    text_units = ask_prefix_text_units + text_units
    # print(text_units)
    processed = "".join(text_units).rstrip(" ") + EOT_FULL

    # orig stuff
    if (
        not post_metadata["is_ask"]
        and not post_metadata["is_reblog"]
        and post_metadata["reply_post_url"] is None
    ):
        processed = ORIG_POST_CHAR + processed
        post_metadata["is_orig"] = True

    if not V10:
        for c in V10_CHARS_TO_LEGACY_CHARS.keys():
            processed = processed.replace(c, V10_CHARS_TO_LEGACY_CHARS[c])

    return processed, post_metadata
