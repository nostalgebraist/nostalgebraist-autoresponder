"""code for turning forumlike (v7) format into the v8 version"""
import os
import re
from functools import partial
from datetime import datetime
from munging.autoresponder_static import *
from experimental.nwo_deprecated import nwo_deprecated

TIME_SIDECHANNEL_CHAR = "\U0001f552"  # clock symbol 🕒

"""V8 format: time of post"""

# utilities for laptop side


def get_ts_from_fn(fn):
    if fn is None:
        return None
    try:
        return datetime.fromtimestamp(os.path.getmtime(fn))
    except FileNotFoundError:
        print(f"couldn't find {fn}")


def timestamp_to_v8_format(ts):
    if ts is None:
        return ""
    try:
        return ts.strftime("%-I:%M %p")
    except:
        return ""


def timestamp_to_v10_format(ts):
    if ts is None:
        return ""
    try:
        return ts.strftime("%-I %p %B %Y")
    except:
        return ""


@nwo_deprecated
def join_time_sidechannel(doc, time_text):
    return doc + TIME_SIDECHANNEL_CHAR + time_text


@nwo_deprecated
def post_and_time_to_v8_sidechannel_format(doc, ts):
    time_text = timestamp_to_v8_format(ts)
    return join_time_sidechannel(doc, time_text)


@nwo_deprecated
# utilities for compute side
def split_off_times_v8(doc):
    normal_text, _, time_text = doc.partition(TIME_SIDECHANNEL_CHAR)
    return normal_text, time_text


def format_segment_v8_time(time_text, control_seg_config=DEFAULT_CSC):
    if len(time_text) == 0:
        return ""
    return control_seg_config["posted_at"].format(time_text=time_text)
    # return f"Posted at {time_text}"


"""V8 format: interlocutor prefix"""


def get_ordered_interlocutors(doc, control_seg_config=DEFAULT_CSC):
    cchars = find_control_chars_forumlike(
        doc, incl_number=False, control_seg_config=control_seg_config
    )

    names = []
    for c in cchars:
        for ph in control_seg_config["numbered_phrases"]:
            if ph in c[0]:
                names.append(c[0].partition(" " + ph)[0])
    n_names = len(names)

    unique_names = []
    for n in names:
        if n not in unique_names:
            unique_names.append(n)
    return unique_names, n_names


def format_segment_v8_interlocutors(doc, control_seg_config=DEFAULT_CSC):
    names, n_names = get_ordered_interlocutors(
        doc, control_seg_config=control_seg_config
    )
    if len(names) == 0:
        return ""
    if len(names) == 1:
        name = names[0]
        if n_names == 1:
            return ""
        else:
            return control_seg_config["series_of_posts"].format(name=name)
    comma_names = ",".join(names[:-1]) + " and" + names[-1]
    return control_seg_config["conversation_between"].format(
        comma_names=comma_names, n_names=n_names
    )


"""V8 format: tags"""


@nwo_deprecated
def split_off_tags_v8(doc):
    doc_tagless, _, tag_string_raw = doc.partition(T_CHAR)
    tag_string_raw = tag_string_raw.partition(EOT)[0].partition("<|")[0]
    return doc_tagless, tag_string_raw


@nwo_deprecated
def format_segment_v8_tags(
    tag_string_raw, user_name="Frank", control_seg_config=DEFAULT_CSC
):
    if len(tag_string_raw) == 0:
        ftags = ""
        # return f"{user_name} tagged this post as: "
    else:
        ftags = ", ".join(["#" + t.rstrip(" ") for t in tag_string_raw.split("#")[1:]])
    # return f"{user_name} tagged this post as: {ftags}"
    return control_seg_config["user_tagged_post"].format(
        user_name=user_name, ftags=ftags
    )


"""V8 format: full thing"""


@nwo_deprecated
def globally_format_v8(
    doc_tagless,
    ts_string,
    interlocutor_string,
    tag_string,
    newline_postfix="\n",  # was "\n\n" in very first version of v8
    strip_final_newlines=True,  # was False in very first version of v8
    extra_names=[],
    control_seg_config=DEFAULT_CSC,
):
    lcc, lcc_loc = last_control_char(
        doc_tagless, extra_names=extra_names, control_seg_config=control_seg_config
    )

    insertion_ix = lcc_loc + len(lcc)
    insertion_string = " | ".join(
        [entry.strip(" ") for entry in [ts_string, tag_string] if len(entry) > 0]
    ).rstrip(" ")
    insertion_string = (
        " " + insertion_string + newline_postfix
        if len(insertion_string) > 0
        else insertion_string
    )
    doc_inserted = (
        doc_tagless[:insertion_ix] + insertion_string + doc_tagless[insertion_ix:]
    )

    prefix = (
        " " + interlocutor_string.rstrip(" ") + " |" + newline_postfix
        if len(interlocutor_string) > 0
        else ""
    )

    formatted = prefix + doc_inserted
    if strip_final_newlines:
        formatted = formatted.rstrip("\n")
    return formatted


@nwo_deprecated
def final_munge_before_neural_v8(
    doc,
    newline_postfix="\n",  # was "\n\n" in very first version of v8
    strip_final_newlines=True,  # was False in very first version of v8
    override_disable_forumlike=False,
    left_strip_newline=True,  # ignored!  for compatibility
    forced_tags_string=None,
    write_fic_override=False,
    control_seg_config=DEFAULT_CSC,
    user_name="Frank",
    mode="predict",
):
    normal_text, time_text = split_off_times_v8(doc)
    normal_text = final_munge_before_neural_v7(
        normal_text,
        override_disable_forumlike=override_disable_forumlike,
        left_strip_newline=True,
        control_seg_config=control_seg_config,
        user_name=user_name,
        mode=mode,
    )

    if override_disable_forumlike:
        return normal_text

    ts_string = format_segment_v8_time(time_text, control_seg_config=control_seg_config)

    interlocutor_string = format_segment_v8_interlocutors(
        normal_text, control_seg_config=control_seg_config
    )

    doc_tagless, tag_string_raw = split_off_tags_v8(normal_text)

    if forced_tags_string is not None and (forced_tags_string != ""):
        if control_seg_config["flags"]["add_control_prefix_to_forced_tag_strings"]:
            tag_string = format_segment_v8_tags(
                tag_string_raw + forced_tags_string,
                control_seg_config=control_seg_config,
                user_name=user_name,
            )
        else:
            tag_string = forced_tags_string
    else:
        tag_string = format_segment_v8_tags(
            tag_string_raw, control_seg_config=control_seg_config, user_name=user_name
        )

    if write_fic_override:
        interlocutor_string = ""

    formatted = globally_format_v8(
        doc_tagless,
        ts_string,
        interlocutor_string,
        tag_string,
        newline_postfix=newline_postfix,
        strip_final_newlines=strip_final_newlines,
        extra_names=[user_name] if mode == "train" else "",
        control_seg_config=control_seg_config,
    )
    if write_fic_override:
        print(f"applying write_fic_override...")
        print(f"starting with {repr(formatted)}")

        if control_seg_config["flags"].get("fic_override_v2", False):
            story_prompt = extract_core_from_forumlike_ask_prompt(formatted, control_seg_config=control_seg_config)

            formatted = construct_fic_override_v2(story_prompt, control_seg_config=control_seg_config)
        else:
            lcc = last_control_char(formatted, control_seg_config=control_seg_config)

            print(f"found lcc {lcc}")

            formatted_ = formatted[: lcc[1]]

            print(f"subsetted to {formatted_}")

            formatted_ = formatted_ + control_seg_config["ORIG_FICTION_CHAR_FORUMLIKE"]

            if control_seg_config["flags"]["fic_override_add_remainder"]:
                remainder = formatted[lcc[1] + len(lcc[0]) :]
                formatted_ = formatted_ + remainder
                print(f"added remainder {remainder}")

            formatted = formatted_
        print(f"using: {formatted}")
    if GLOBAL_DEBUG:
        print(
            f"v8: neural model will see exactly the following:\n\n{repr(formatted)}\n\n"
        )
    return formatted


@nwo_deprecated
def extract_core_from_forumlike_ask_prompt(text, control_seg_config=DEFAULT_CSC):
    ccs = find_control_chars(text, control_seg_config=control_seg_config)
    if len(ccs) >= 2:
        print(f"using these to segment: {repr(ccs[-2:])}")
        core_end = ccs[-1][1]
        core_start = ccs[-2][1] + len(ccs[-2][0])
        return text[core_start:core_end].strip("\n")
    else:
        print(f"could not deal with control chars, have {repr(ccs)}")
        return ""


def construct_fic_override_v2(story_prompt, control_seg_config=DEFAULT_CSC, verbose=True):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint(f"starting with {repr(story_prompt)}")

    title_triggers = [tt
                      for thing in ['story', 'fic']
                      for tt in [f'{thing} about', f'{thing} in which', f'{thing} of', ]]

    formatted = None

    for tt in title_triggers:
        if tt in story_prompt:
            title = story_prompt.partition(tt)[2].strip('.,!? ')
            if len(title) == 0:
                continue
            title = title[0].upper() + title[1:]

            # "A [noun]" --> "The [noun]"
            title = re.sub(r"\AA |\AAn ", "The ", title)

            vprint(f"on {tt} path")
            vprint(f"formed title {repr(title)}")
            formatted = control_seg_config['ORIG_FICTION_CHAR_FORUMLIKE'] + " #original fiction\n" + f"<h2>{title}</h2>"

    if formatted is None:
        formatted = control_seg_config['ORIG_FICTION_CHAR_FORUMLIKE']

    vprint(f"using {repr(formatted)}")

    return formatted


@nwo_deprecated
def final_munge_after_neural_v8(text, delete_title=False):
    # strip orig post starters
    for cchar in [
        ORIG_POST_CHAR_FORUMLIKE,
        REVIEW_CHAR_FORUMLIKE,
        ORIG_FICTION_CHAR_FORUMLIKE,
    ]:
        text = text.replace(cchar, "")

    # swap tags back into chinese format
    tag_text, _, post = text.partition("\n")
    tag_text = tag_text.rpartition("|")[2].rpartition("tagged this post as:")[2]

    post = post.replace(EOT, "")
    tag_text = tag_text.replace(EOT, "") + EOT

    if delete_title:
        post = re.sub(r"<h2>.+</h2>[\n]*", "", post)

    return post + T_CHAR + tag_text


@nwo_deprecated
def final_munge_before_neural_v10(doc, **kwargs):
    return final_munge_before_neural_v8(
        doc, control_seg_config=CONTROL_SEG_CONFIGS["V10"], **kwargs
    )


@nwo_deprecated
def final_munge_before_neural_v10_1(doc, **kwargs):
    if kwargs.get('mode') != 'train':
        kwargs["user_name"] = "nostalgebraist-autoresponder"
    return final_munge_before_neural_v8(
        doc,
        control_seg_config=CONTROL_SEG_CONFIGS["V10_1"],
        **kwargs
    )


@nwo_deprecated
def v10_1_to_v10_2(doc):
    bad = CONTROL_SEG_CONFIGS['V10_1']['ORIG_POST_CHAR_FORUMLIKE']
    good = CONTROL_SEG_CONFIGS['V10_2']['ORIG_POST_CHAR_FORUMLIKE']
    if doc.startswith(bad):
        return good + doc[len(bad):]
    return doc


def final_munge_before_neural_v10_2(doc, **kwargs):
    if kwargs.get('mode') != 'train':
        kwargs["user_name"] = "nostalgebraist-autoresponder"
    return v10_1_to_v10_2(
        final_munge_before_neural_v8(
            doc,
            control_seg_config=CONTROL_SEG_CONFIGS["V10_2"],
            **kwargs
        )
    )


@nwo_deprecated
def _final_munge_after_neural_v10(text, delete_title=False, control_seg_config=DEFAULT_CSC):
    # strip orig post starters

    for cchar in [
        control_seg_config["ORIG_POST_CHAR_FORUMLIKE"],
        control_seg_config["REVIEW_CHAR_FORUMLIKE"],
        control_seg_config["ORIG_FICTION_CHAR_FORUMLIKE"]
    ]:
        text = text.replace(cchar, "")

    # swap tags back into chinese format
    tag_text, _, post = text.partition("\n")

    if f" | {control_seg_config}'s tags:" in tag_text:
        tag_text = tag_text.rpartition("|")[2].rpartition("tags:")[2]

    post = post.replace(EOT, "")
    tag_text = tag_text.replace(EOT, "") + EOT

    if delete_title:
        post = re.sub(r"<h2>.+</h2>[\n]*", "", post)

    return post + T_CHAR + tag_text


final_munge_after_neural_v10 = partial(_final_munge_after_neural_v10, control_seg_config=CONTROL_SEG_CONFIGS['V10'])

final_munge_after_neural_v10_1 = partial(_final_munge_after_neural_v10, control_seg_config=CONTROL_SEG_CONFIGS['V10_1'])

final_munge_after_neural_v10_2 = partial(_final_munge_after_neural_v10, control_seg_config=CONTROL_SEG_CONFIGS['V10_2'])
