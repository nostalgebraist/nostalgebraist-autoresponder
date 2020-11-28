import re
import numpy as np

FORUMLIKE = True
FORUMLIKE_V2 = True
NORMALIZE = True

GLOBAL_DEBUG = False

EOT_FULL = "<|endoftext|>"
T_CHAR = "职"
Q_CHAR = "会"
A_CHAR = "域"
UNAME_CHAR = "友"

ORIG_POST_CHAR_FORUMLIKE = " Blog post by Frank\n\n"
ORIG_POST_CHAR_CHINESE = "翰"

REVIEW_CHAR_FORUMLIKE = " Book review by Frank\n\n"
if FORUMLIKE_V2:
    ORIG_FICTION_CHAR_FORUMLIKE = " Original fiction by Frank\n\n"
else:
    ORIG_FICTION_CHAR_FORUMLIKE = ORIG_POST_CHAR_CHINESE

if FORUMLIKE:
    ORIG_POST_CHAR = ORIG_POST_CHAR_FORUMLIKE
else:
    ORIG_POST_CHAR = ORIG_POST_CHAR_CHINESE



def find_username_tags(text):
    return re.findall(r"#[0-9]+ [^ ]* wrote:[\n]{1,2}", text)

def find_control_chars_chinese(text,
                               incl_number=True  # ignored
                               ):
    results = []
    control_chars = [Q_CHAR, A_CHAR, UNAME_CHAR, ORIG_POST_CHAR_CHINESE] # no tchar
    for c in control_chars:
        if c in text:
            results.append((c, text.index(c)))
    return results

# TODO: deprecate find_control_chars_chinese for this one
def find_all_control_chars_chinese(text,
                                   incl_number=True  # ignored
                                   ):
    results = []
    control_chars = [Q_CHAR, A_CHAR, UNAME_CHAR, ORIG_POST_CHAR_CHINESE] # no tchar
    rx = f"({UNAME_CHAR}[^{Q_CHAR}]+{Q_CHAR}|{A_CHAR}|{ORIG_POST_CHAR_CHINESE})"
    for m in re.finditer(rx, text):
        results.append((m.group(1), m.span(1)[0]))
    return results

def find_control_chars_forumlike(text,
                                 incl_number=True,
                                 extra_names=[]
                                 ):
    results = []
    control_chars = [ORIG_POST_CHAR_FORUMLIKE, REVIEW_CHAR_FORUMLIKE, ORIG_FICTION_CHAR_FORUMLIKE] # no tchar
    control_chars.extend(list({c.replace("Frank", en) for c in control_chars for en in extra_names}))
    for c in control_chars:
        if c in text:
            results.append((c, text.index(c)))
    rx = r"(#[0-9]+ .*? wrote:[\n]{1,2})" if incl_number else r"#[0-9]+( .*? wrote:[\n]{1,2})"
    for m in re.finditer(rx, text):
        results.append((m.group(1), m.span(1)[0]))
    return results

if FORUMLIKE:
    find_control_chars = find_control_chars_forumlike
else:
    find_control_chars = find_control_chars_chinese

def contains_control_chars(text, incl_number=True, extra_names=[]):
    return len(find_control_chars(text, incl_number=incl_number, extra_names=extra_names))>0

def first_control_char(text, incl_number=True, extra_names=[]):
    return sorted(find_control_chars(text, incl_number=incl_number, extra_names=extra_names), key=lambda tup: tup[1])[0]

def last_control_char(text, incl_number=True, extra_names=[]):
    return sorted(find_control_chars(text, incl_number=incl_number, extra_names=extra_names),
                  key=lambda tup: tup[1])[-1]

def substitute_forumlike(text, shuffle=True, infer_first=True, mode="predict", left_strip_newline=True, user_name="Frank"):
    segments = [s for s in text.split(EOT_FULL)]

    segments_subbed = []
    accum = []
    for ixseg, seg_ in enumerate(segments):
        if len(seg_) == 0:
            segments_subbed.append(seg_)
            continue
        debug=False
        seg = "GPT2_EOT".join(accum + [seg_])
        control_ixs = {c: seg.find(c) for c in [ORIG_POST_CHAR_CHINESE, UNAME_CHAR, A_CHAR]}
        control_ixs = {k: v for k, v in control_ixs.items() if v >= 0}
        try:
            first_char = sorted(control_ixs.keys(), key=lambda c: control_ixs[c])[0]
            accum = []
        except IndexError:
            if len(accum)==0:
                accum = [segments[ixseg-1], seg_]
                segments_subbed = segments_subbed[:-1]
            else:
                accum.append(seg_)
            continue

        if first_char == ORIG_POST_CHAR_CHINESE:
            sub = f" Blog post by {user_name}\n\n"
            seg_subbed = seg[:control_ixs[first_char]] + sub + seg[control_ixs[first_char]+1:]
        else:
            seg_subbed = seg

        if infer_first and control_ixs[first_char] > 15:
            seg_subbed = "域" + seg_subbed

        seg_subbed = normalize_for_generator(seg_subbed)

        if "友 nostalgebraist域" in seg_subbed:
            if mode == "train":
                seg_subbed = seg_subbed.replace("友 nostalgebraist域", "友 Frank会")
                seg_subbed = seg_subbed.replace("域", "友 nostalgebraist-my-father会")
            elif mode == "predict":
                seg_subbed = seg_subbed.replace("友 nostalgebraist域", "友 nostalgebraist-my-father会")
                seg_subbed = seg_subbed.replace("域", "友 Frank会")
        else:
            seg_subbed = seg_subbed.replace("域", f"友 {user_name}会")

        seg_subbed = re.sub(r"<h2>友[^会^\/]*/[^会]*会 <\/h2>", "", seg_subbed)
        seg_subbed = re.sub(r"\n*友[^会^\/]*/[^会]*会\n*", "", seg_subbed)

        postix = 1
        next_ = None
        while True:
            next_ = re.sub(r"\n*友([^会^\/]*)会\n*", fr"\n\n#{postix}\1 wrote:\n\n", seg_subbed, count=1)
            if next_ == seg_subbed:
                break
            seg_subbed = next_
            postix += 1

        if mode == "train":
            seg_subbed = seg_subbed.strip("\n")
        elif left_strip_newline:
            seg_subbed = seg_subbed.lstrip("\n")
        if debug:
            print(f"got segment:\n\n{seg_subbed}")
            debug=False
        segments_subbed.append(seg_subbed)

    if shuffle:
        np.random.shuffle(segments_subbed)
    text_subbed = EOT_FULL.join(segments_subbed)
    return text_subbed

def collapse_multi_newline(s):
    return re.sub(r"([^\n])(\n{3,})([^\n])", r"\g<1>\n\n\g<3>", s)

newlines_then_wordlike_regex = re.compile(r"(\n+)([^\W\n])", flags=re.ASCII)
newlines_then_wordlike_repl = r"\g<1> \g<2>"

def add_space_after_newline_before_word(s):
    return re.sub(newlines_then_wordlike_regex, newlines_then_wordlike_repl, s)

def normalize_for_generator(s: str):
    normed_data_string = s

    char_maps = {
    '\xa0': ' ',
    "“": "\"",
    "”": "\"",
    "‘": "'",
    "’": "'",
    }

    for _ in range(2):
      for k, v in char_maps.items():
        normed_data_string = normed_data_string.replace(k, v)

    normed_data_string = collapse_multi_newline(normed_data_string)
    normed_data_string = add_space_after_newline_before_word(normed_data_string)

    control_chars = [Q_CHAR, A_CHAR, UNAME_CHAR, ORIG_POST_CHAR] # no tchar

    for c in control_chars:
        normed_data_string = re.sub(f"({c})([^ \n]+?)", r"\g<1> \g<2>", normed_data_string)

    normed_data_string = re.sub(
        r"([.!?]) {2,2}([\w])",
        r"\g<1> \g<2>",
        normed_data_string
    )

    return normed_data_string

def final_munge_before_neural_v7(text, override_disable_forumlike=False, left_strip_newline=True):
  text = re.sub(r"\\n", "\n", text)
  if NORMALIZE:
      text = normalize_for_generator(text)
  if FORUMLIKE and not override_disable_forumlike:
      text = substitute_forumlike(text, shuffle=False, infer_first=False, left_strip_newline=left_strip_newline)
  if GLOBAL_DEBUG:
      print(f"v7: neural model will see exactly the following:\n\n{repr(text)}\n\n")
  return text

def cut_to_final_exchange_chinese(to_cut):
    cchars = find_all_control_chars_chinese(to_cut)
    if len(cchars) < 2:
        print(f'not cutting: only found cchars {cchars}')
        return to_cut

    cut_ix = find_all_control_chars_chinese(to_cut)[-2][1]
    print(f'cutting at {cut_ix}')
    return to_cut[cut_ix:]
