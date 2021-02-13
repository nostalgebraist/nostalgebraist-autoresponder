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

V10_ASK_CHAR = "要"

ORIG_POST_CHAR_CHINESE = "翰"

CHINESE_CHAR_DELIMITERS = [UNAME_CHAR, Q_CHAR, A_CHAR, V10_ASK_CHAR, ORIG_POST_CHAR_CHINESE]

ORIG_POST_CHAR_FORUMLIKE = " Blog post by Frank\n\n"
ORIG_POST_CHAR_FORUMLIKE_V10 = " Frank posted:\n\n"

REVIEW_CHAR_FORUMLIKE = " Book review by Frank\n\n"
REVIEW_CHAR_FORUMLIKE_V10 = (
    " Book review by Frank\n\n Frank's tags:\n Frank tagged this post as:"  # my mistake
)

if FORUMLIKE_V2:
    ORIG_FICTION_CHAR_FORUMLIKE = " Original fiction by Frank\n\n"
else:
    ORIG_FICTION_CHAR_FORUMLIKE = ORIG_POST_CHAR_CHINESE

ORIG_FICTION_CHAR_FORUMLIKE_V10 = (
    " Original fiction by Frank\n\n Frank's tags:\n Frank tagged this post as:"
)

# if FORUMLIKE:
#     ORIG_POST_CHAR = ORIG_POST_CHAR_FORUMLIKE
# else:
#     ORIG_POST_CHAR = ORIG_POST_CHAR_CHINESE


CONTROL_SEG_CONFIGS = {
    "V9": {
        "ORIG_POST_CHAR_FORUMLIKE": ORIG_POST_CHAR_FORUMLIKE,
        "ORIG_FICTION_CHAR_FORUMLIKE": ORIG_FICTION_CHAR_FORUMLIKE,
        "REVIEW_CHAR_FORUMLIKE": REVIEW_CHAR_FORUMLIKE,
        "ORIG_POST_CHAR_NAMED": " Blog post by {user_name}\n\n",
        "asked_word": "wrote",
        "replied_word": "wrote",
        "op_word": "wrote",
        "reblogger_word": "wrote",
        "user_tagged_post": "{user_name} tagged this post as: {ftags}",
        "series_of_posts": "A series of blog posts by{name}",
        "conversation_between": "A blog conversation between{comma_names} | {n_names} posts",
        "posted_at": "Posted at {time_text}",
        "flags": {
            "fic_override_add_remainder": True,
            "add_control_prefix_to_forced_tag_strings": True,
        },
        "ASK_CHAR": Q_CHAR,
        "V10": False,
    },
    "V10": {
        "ORIG_POST_CHAR_FORUMLIKE": ORIG_POST_CHAR_FORUMLIKE_V10,
        "ORIG_FICTION_CHAR_FORUMLIKE": ORIG_FICTION_CHAR_FORUMLIKE_V10,
        "REVIEW_CHAR_FORUMLIKE": REVIEW_CHAR_FORUMLIKE_V10,
        "ORIG_POST_CHAR_NAMED": " {user_name} posted:\n\n",
        "asked_word": "asked",
        "replied_word": "responded",
        "op_word": "posted",
        "reblogger_word": "commented",
        "user_tagged_post": "{user_name}'s tags: {ftags}",
        "series_of_posts": "Posts by{name}",
        "conversation_between": "Conversation between{comma_names} | {n_names} posts",
        "posted_at": "Written {time_text}",
        "flags": {
            "fic_override_add_remainder": False,
            "add_control_prefix_to_forced_tag_strings": False,
        },
        "ASK_CHAR": V10_ASK_CHAR,
        "V10": True,
    },
}

for k in CONTROL_SEG_CONFIGS.keys():
    CONTROL_SEG_CONFIGS[k]["numbered_phrases"] = {
        CONTROL_SEG_CONFIGS[k][k2]
        for k2 in ["asked_word", "replied_word", "op_word", "reblogger_word"]
    }

# DEFAULT_CSC = CONTROL_SEG_CONFIGS["V9"]
DEFAULT_CSC = CONTROL_SEG_CONFIGS["V10"]


def find_username_tags(text):
    return re.findall(r"#[0-9]+ [^ ]* wrote:[\n]{1,2}", text)


def find_control_chars_chinese(
    text,
    incl_number=True,  # ignored
    control_seg_config=DEFAULT_CSC,  # ignored
):
    results = []
    control_chars = [Q_CHAR, A_CHAR, UNAME_CHAR, ORIG_POST_CHAR_CHINESE]  # no tchar
    for c in control_chars:
        if c in text:
            results.append((c, text.index(c)))
    return results


# TODO: deprecate find_control_chars_chinese for this one
def find_all_control_chars_chinese(
    text,
    incl_number=True,  # ignored
    control_seg_config=DEFAULT_CSC,  # ignored
):
    results = []
    sub_rxs = [
        f"{UNAME_CHAR}[^{Q_CHAR}{V10_ASK_CHAR}]+[{Q_CHAR}{V10_ASK_CHAR}]",
        A_CHAR,
        ORIG_POST_CHAR_CHINESE,
    ]
    rx = "(" + "|".join(sub_rxs) + ")"

    for m in re.finditer(rx, text):
        results.append((m.group(1), m.span(1)[0]))
    return results


def find_control_chars_forumlike(
    text,
    incl_number=True,
    extra_names=[],
    control_seg_config=DEFAULT_CSC,
):
    results = []
    control_chars = [
        ORIG_POST_CHAR_FORUMLIKE,
        ORIG_POST_CHAR_FORUMLIKE_V10,
        REVIEW_CHAR_FORUMLIKE,
        REVIEW_CHAR_FORUMLIKE_V10,
        ORIG_FICTION_CHAR_FORUMLIKE,
        ORIG_FICTION_CHAR_FORUMLIKE_V10,
    ]  # no tchar
    control_chars.extend(
        list({c.replace("Frank", en) for c in control_chars for en in extra_names})
    )
    for c in control_chars:
        if c in text:
            results.append((c, text.index(c)))

    for p in control_seg_config["numbered_phrases"]:
        rx = (
            fr"(#[0-9]+ .*? {p}:[\n]{{1,2}})"
            if incl_number
            else fr"#[0-9]+( .*? {p}:[\n]{{1,2}})"
        )
        # rx = r"(#[0-9]+ .*? wrote:[\n]{1,2})" if incl_number else r"#[0-9]+( .*? wrote:[\n]{1,2})"
        # print(rx)
        for m in re.finditer(rx, text):
            results.append((m.group(1), m.span(1)[0]))
    results = sorted(results, key=lambda tup: tup[1])
    return results


if FORUMLIKE:
    find_control_chars = find_control_chars_forumlike
else:
    find_control_chars = find_control_chars_chinese


def contains_control_chars(
    text,
    incl_number=True,
    extra_names=[],
    control_seg_config=DEFAULT_CSC,
):
    return (
        len(
            find_control_chars(
                text,
                incl_number=incl_number,
                extra_names=extra_names,
                control_seg_config=control_seg_config,
            )
        )
        > 0
    )


def first_control_char(
    text,
    incl_number=True,
    extra_names=[],
    control_seg_config=DEFAULT_CSC,
):
    return sorted(
        find_control_chars(
            text,
            incl_number=incl_number,
            extra_names=extra_names,
            control_seg_config=control_seg_config,
        ),
        key=lambda tup: tup[1],
    )[0]


def last_control_char(
    text,
    incl_number=True,
    extra_names=[],
    control_seg_config=DEFAULT_CSC,
):
    return sorted(
        find_control_chars(
            text,
            incl_number=incl_number,
            extra_names=extra_names,
            control_seg_config=control_seg_config,
        ),
        key=lambda tup: tup[1],
    )[-1]


def substitute_forumlike(
    text,
    shuffle=True,
    infer_first=True,
    mode="predict",
    left_strip_newline=True,
    user_name="Frank",
    control_seg_config=DEFAULT_CSC,
    debug=False,
):
    segments = [s for s in text.split(EOT_FULL)]

    segments_subbed = []
    accum = []
    for ixseg, seg_ in enumerate(segments):
        if len(seg_) == 0:
            segments_subbed.append(seg_)
            continue
        seg = "GPT2_EOT".join(accum + [seg_])

        control_ixs = {
            c: seg.find(c) for c in [ORIG_POST_CHAR_CHINESE, UNAME_CHAR, A_CHAR]
        }
        control_ixs = {k: v for k, v in control_ixs.items() if v >= 0}
        try:
            first_char = sorted(control_ixs.keys(), key=lambda c: control_ixs[c])[0]
            accum = []
        except IndexError:
            print(("indexerr seg", seg))
            if len(accum) == 0:
                accum = [segments[ixseg - 1], seg_]
                segments_subbed = segments_subbed[:-1]
            else:
                accum.append(seg_)
            continue

        if debug:
            print(f"control_ixs: {control_ixs}")
            print(f"first_char: {first_char}")

        if first_char == ORIG_POST_CHAR_CHINESE:
            # sub = f" Blog post by {user_name}\n\n"
            sub = control_seg_config["ORIG_POST_CHAR_NAMED"].format(user_name=user_name)
            seg_subbed = (
                seg[: control_ixs[first_char]]
                + sub
                + seg[control_ixs[first_char] + 1 :]
            )
        else:
            seg_subbed = seg

        if infer_first and control_ixs[first_char] > 15:
            seg_subbed = "域" + seg_subbed

        seg_subbed = normalize_for_generator(seg_subbed)

        # print(seg_subbed)
        # trig=False
        if "友 nostalgebraist-autoresponder域" in seg_subbed:
            # print(seg_subbed)
            seg_subbed = seg_subbed.replace(
                "友 nostalgebraist-autoresponder域", "友 Frank会"
            )
            if mode == "train":
                # seg_subbed = seg_subbed.replace("友 nostalgebraist-autoresponder", "友 Frank会")
                # seg_subbed = seg_subbed.replace("域", "友 nostalgebraist-my-father会")
                seg_subbed = seg_subbed.replace("域", "友 nostalgebraist会")
            # elif mode == "predict":
            #     seg_subbed = seg_subbed.replace("友 nostalgebraist会", "友 nostalgebraist-my-father会")
            #     seg_subbed = seg_subbed.replace("域", "友 Frank会")
            # print(seg_subbed)
            # trig=True
        seg_subbed = seg_subbed.replace("域", f"友 {user_name}会")

        seg_subbed = re.sub(r"<h2>友[^会^\/]*/[^会]*会 <\/h2>", "", seg_subbed)

        seg_subbed = re.sub(r"\n*友[^会^要^\/]*/[^会^要]*会\n*", "", seg_subbed)

        postix = 1
        next_ = None

        asking_name = None

        asked_word = control_seg_config["asked_word"]
        replied_word = control_seg_config["replied_word"]
        op_word = control_seg_config["op_word"]
        reblogger_word = control_seg_config["reblogger_word"]

        if debug:
            print(f"got raw segment:\n\n{seg_subbed}")

        while True:
            if asking_name is None:
                for candidate_name in re.findall(r"\n*友([^要^\/]*)要\n*", seg_subbed):
                    asking_name = candidate_name
                    if debug:
                        print(f"got asking_name {asking_name}")
            if postix == 1 and asking_name is None:
                next_ = re.sub(
                    r"\n*友([^会^\/]*)会\n*",
                    fr"\n\n#{postix}\1 {op_word}:\n\n",
                    seg_subbed,
                    count=1,
                )
                if debug:
                    print(f"updated subbed segment (op):\n\n{next_}")
            elif postix == 1 and asking_name is not None:
                next_ = re.sub(
                    r"\n*友([^要^\/]*)要\n*",
                    fr"\n\n#{postix}\1 {asked_word}:\n\n",
                    seg_subbed,
                    count=1,
                )
                if debug:
                    print(f"updated subbed segment (ask):\n\n{next_}")
            elif postix == 2 and asking_name is not None:
                next_ = re.sub(
                    r"\n*友([^会^\/]*)会\n*",
                    fr"\n\n#{postix}\1 {replied_word}:\n\n",
                    seg_subbed,
                    count=1,
                )
                if debug:
                    print(f"updated subbed segment (replied):\n\n{next_}")
            else:
                next_ = re.sub(
                    r"\n*友([^会^\/]*)会\n*",
                    fr"\n\n#{postix}\1 {reblogger_word}:\n\n",
                    seg_subbed,
                    count=1,
                )
                if debug:
                    print(f"updated subbed segment (reblogged):\n\n{next_}")
            if debug:
                print(f"got subbed segment:\n\n{next_}")
            if next_ == seg_subbed:
                break
            seg_subbed = next_
            postix += 1

        # if trig:
        #     print(seg_subbed+"\n---\n")

        if mode == "train":
            seg_subbed = seg_subbed.strip("\n")
        elif left_strip_newline:
            seg_subbed = seg_subbed.lstrip("\n")
        if debug:
            print(f"got segment:\n\n{seg_subbed}")
            # debug=False
        segments_subbed.append(seg_subbed)

    if shuffle:
        np.random.shuffle(segments_subbed)
    if debug:
        print(f"joining segs {segments_subbed}")
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
        "\xa0": " ",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }

    for _ in range(2):
        for k, v in char_maps.items():
            normed_data_string = normed_data_string.replace(k, v)

    normed_data_string = collapse_multi_newline(normed_data_string)
    normed_data_string = add_space_after_newline_before_word(normed_data_string)

    control_chars = [
        Q_CHAR,
        A_CHAR,
        UNAME_CHAR,
        ORIG_POST_CHAR_FORUMLIKE,
        ORIG_POST_CHAR_FORUMLIKE_V10,
    ]  # no tchar

    for c in control_chars:
        normed_data_string = re.sub(
            f"({c})([^ \n]+?)", r"\g<1> \g<2>", normed_data_string
        )

    normed_data_string = re.sub(
        r"([.!?]) {2,2}([\w])", r"\g<1> \g<2>", normed_data_string
    )

    return normed_data_string


def final_munge_before_neural_v7(
    text,
    override_disable_forumlike=False,
    left_strip_newline=True,
    control_seg_config=DEFAULT_CSC,
    user_name="Frank",
    mode="predict",
):
    text = re.sub(r"\\n", "\n", text)
    if NORMALIZE:
        text = normalize_for_generator(text)
    if FORUMLIKE and not override_disable_forumlike:
        text = substitute_forumlike(
            text,
            shuffle=False,
            infer_first=False,
            left_strip_newline=left_strip_newline,
            control_seg_config=control_seg_config,
            user_name=user_name,
            mode=mode,
        )
    if False:  # GLOBAL_DEBUG:
        print(f"v7: neural model will see exactly the following:\n\n{repr(text)}\n\n")
    return text


def cut_to_final_exchange_chinese(to_cut):
    cchars = find_all_control_chars_chinese(to_cut)
    if len(cchars) < 2:
        print(f"not cutting: only found cchars {cchars}")
        return to_cut

    cut_ix = find_all_control_chars_chinese(to_cut)[-2][1]
    print(f"cutting at {cut_ix}")
    return to_cut[cut_ix:]
