"""code for turning forumlike (v7) format into the v8 version"""
import os
from tumblr_to_text.classic.autoresponder_static import *
from util.times import fromtimestamp_pst


"""V8 format: time of post"""

# utilities for laptop side


def get_ts_from_fn(fn):
    if fn is None:
        return None
    try:
        return fromtimestamp_pst(os.path.getmtime(fn))
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


def construct_fic_override_v2(story_prompt, control_seg_config=DEFAULT_CSC, use_definite_article=True, verbose=True):
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

            if use_definite_article:
                # "A [noun]" --> "The [noun]"
                title = re.sub(r"\AA |\AAn ", "The ", title)

            vprint(f"on {tt} path")
            vprint(f"formed title {repr(title)}")
            formatted = control_seg_config['ORIG_FICTION_CHAR_FORUMLIKE'] + " #original fiction\n" + f"<h2>{title}</h2>"

    if formatted is None:
        formatted = control_seg_config['ORIG_FICTION_CHAR_FORUMLIKE']

    vprint(f"using {repr(formatted)}")

    return formatted
