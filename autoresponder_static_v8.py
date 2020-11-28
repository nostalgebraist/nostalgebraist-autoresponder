"""code for turning forumlike (v7) format into the v8 version"""
import os
from datetime import datetime
from autoresponder_static import *

TIME_SIDECHANNEL_CHAR = '\U0001f552'  # clock symbol 🕒

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

def join_time_sidechannel(doc, time_text):
    return doc + TIME_SIDECHANNEL_CHAR + time_text

def post_and_time_to_v8_sidechannel_format(doc, ts):
    time_text = timestamp_to_v8_format(ts)
    return join_time_sidechannel(doc, time_text)

# utilities for compute side
def split_off_times_v8(doc):
    normal_text, _, time_text = doc.partition(TIME_SIDECHANNEL_CHAR)
    return normal_text, time_text

def format_segment_v8_time(time_text):
    if len(time_text)==0:
        return ""
    return f"Posted at {time_text}"

"""V8 format: interlocutor prefix"""

def get_ordered_interlocutors(doc):
    chars = [c[0] for c in find_control_chars_forumlike(doc, incl_number=False) if " wrote" in c[0]]
    names = [c.partition(" wrote")[0] for c in chars]
    n_names = len(names)

    unique_names = []
    for n in names:
        if n not in unique_names:
            unique_names.append(n)
    return unique_names, n_names

def format_segment_v8_interlocutors(doc):
    names, n_names = get_ordered_interlocutors(doc)
    if len(names) == 0:
        return ""
    if len(names) == 1:
        name = names[0]
        return f"A series of blog posts by{name}"
    comma_names = ",".join(names[:-1]) + " and" + names[-1]
    return f"A blog conversation between{comma_names} | {n_names} posts"

"""V8 format: tags"""

def split_off_tags_v8(doc):
    doc_tagless, _, tag_string_raw = doc.partition(T_CHAR)
    tag_string_raw = tag_string_raw.partition(EOT_FULL)[0].partition("<|")[0]
    return doc_tagless, tag_string_raw

def format_segment_v8_tags(tag_string_raw, user_name="Frank"):
    if len(tag_string_raw)==0:
        return f"{user_name} tagged this post as: "
    ftags = ", ".join(["#"+t.rstrip(" ") for t in tag_string_raw.split("#")[1:]])
    return f"{user_name} tagged this post as: {ftags}"

"""V8 format: full thing"""

def globally_format_v8(doc_tagless, ts_string, interlocutor_string, tag_string,
                       newline_postfix="\n",  # was "\n\n" in very first version of v8
                       strip_final_newlines=True,  # was False in very first version of v8
                       extra_names=[],
                      ):
    lcc, lcc_loc = last_control_char(doc_tagless, extra_names=extra_names)

    insertion_ix = lcc_loc+len(lcc)
    insertion_string = " | ".join([entry.strip(" ") for entry in [ts_string, tag_string]
                         if len(entry)>0]).rstrip(" ")
    insertion_string = " " + insertion_string + newline_postfix if len(insertion_string) > 0 else insertion_string
    doc_inserted = doc_tagless[:insertion_ix] + insertion_string + doc_tagless[insertion_ix:]

    prefix = " " + interlocutor_string.rstrip(" ") + " |" + newline_postfix \
    if len(interlocutor_string) > 0 else ""

    formatted = prefix + doc_inserted
    if strip_final_newlines:
        formatted = formatted.rstrip("\n")
    return formatted

def final_munge_before_neural_v8(doc,
                                   newline_postfix="\n",  # was "\n\n" in very first version of v8
                                   strip_final_newlines=True,  # was False in very first version of v8
                                   override_disable_forumlike=False,
                                   left_strip_newline=True,  # ignored!  for compatibility
                                   forced_tags_string="",
                                   write_fic_override=False,
                                   ):
    normal_text, time_text = split_off_times_v8(doc)
    normal_text = final_munge_before_neural_v7(normal_text, override_disable_forumlike=override_disable_forumlike, left_strip_newline=True)
    if override_disable_forumlike:
        return normal_text

    ts_string = format_segment_v8_time(time_text)

    interlocutor_string = format_segment_v8_interlocutors(normal_text)

    doc_tagless, tag_string_raw = split_off_tags_v8(normal_text)
    tag_string = format_segment_v8_tags(tag_string_raw + forced_tags_string)

    if write_fic_override:
      interlocutor_string = ""

    formatted = globally_format_v8(doc_tagless, ts_string, interlocutor_string, tag_string,
                                   newline_postfix=newline_postfix,
                                   strip_final_newlines=strip_final_newlines)
    if write_fic_override:
        print(f'applying write_fic_override...')
        print(f'starting with {repr(formatted)}')
        lcc = last_control_char(formatted)

        print(f'found lcc {lcc}')

        formatted_ = formatted[:lcc[1]]

        print(f'subsetted to {formatted_}')

        formatted_ = formatted_ + ORIG_FICTION_CHAR_FORUMLIKE

        remainder = formatted[lcc[1]+len(lcc[0]):]
        formatted_ = formatted_ + remainder
        print(f'added remainder {remainder}')

        formatted = formatted_
        print(f'using: {formatted}')
    if GLOBAL_DEBUG:
      print(f"v8: neural model will see exactly the following:\n\n{repr(formatted)}\n\n")
    return formatted

def final_munge_after_neural_v8(text):
  # strip orig post starters
  for cchar in [ORIG_POST_CHAR_FORUMLIKE, REVIEW_CHAR_FORUMLIKE, ORIG_FICTION_CHAR_FORUMLIKE]:
    text = text.replace(cchar, "")

  # swap tags back into chinese format
  tag_text, _, post = text.partition("\n")
  tag_text = tag_text.rpartition("|")[2].rpartition("tagged this post as:")[2]

  post = post.replace(EOT_FULL, "")
  tag_text = tag_text.replace(EOT_FULL, "") + EOT_FULL

  return post + T_CHAR + tag_text
