import re
from copy import deepcopy

NOSPACE = True

GLOBAL_DEBUG = False

EOT = "<|endoftext|>"
T_CHAR = "职"
Q_CHAR = "会"
A_CHAR = "域"
UNAME_CHAR = "友"

V10_ASK_CHAR = "要"

ORIG_POST_CHAR_CHINESE = "翰"

CHINESE_CHAR_DELIMITERS = [
    UNAME_CHAR,
    Q_CHAR,
    A_CHAR,
    V10_ASK_CHAR,
    ORIG_POST_CHAR_CHINESE,
]

ORIG_POST_CHAR_FORUMLIKE = " Blog post by Frank\n\n"
ORIG_POST_CHAR_FORUMLIKE_V10 = " Frank posted:\n\n"
ORIG_POST_CHAR_FORUMLIKE_V10_1 = " nostalgebraist-autoresponder posted:\n\n"
ORIG_POST_CHAR_FORUMLIKE_V10_2 = " Posts by nostalgebraist-autoresponder |\n nostalgebraist-autoresponder posted:\n\n"

REVIEW_CHAR_FORUMLIKE = " Book review by Frank\n\n"
REVIEW_CHAR_FORUMLIKE_V10 = (
    " Book review by Frank\n\n Frank's tags:\n Frank tagged this post as:"  # my mistake
)
REVIEW_CHAR_FORUMLIKE_V10_1 = (
    " Book review by nostalgebraist-autoresponder\n\n nostalgebraist-autoresponder's tags:\n nostalgebraist-autoresponder tagged this post as:"  # my mistake
)

ORIG_FICTION_CHAR_FORUMLIKE = " Original fiction by Frank\n\n"

ORIG_FICTION_CHAR_FORUMLIKE_V10 = (
    " Original fiction by Frank\n\n Frank's tags:\n Frank tagged this post as:"
)

ORIG_FICTION_CHAR_FORUMLIKE_V10_1 = (
    " Original fiction by nostalgebraist-autoresponder\n\n nostalgebraist-autoresponder's tags:\n nostalgebraist-autoresponder tagged this post as:"
)


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
        "user_name": "Frank",  # TODO: refactor to use this always
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
        "user_name": "Frank",  # TODO: refactor to use this always
    },
    "V10_1": {
        "ORIG_POST_CHAR_FORUMLIKE": ORIG_POST_CHAR_FORUMLIKE_V10_1,
        "ORIG_FICTION_CHAR_FORUMLIKE": ORIG_FICTION_CHAR_FORUMLIKE_V10_1,
        "REVIEW_CHAR_FORUMLIKE": REVIEW_CHAR_FORUMLIKE_V10_1,
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
            "fic_override_v2": True,
        },
        "ASK_CHAR": V10_ASK_CHAR,
        "V10": True,
        "user_name": "nostalgebraist-autoresponder",  # TODO: refactor to use this always
    },
}

CONTROL_SEG_CONFIGS["V10_2"] = deepcopy(CONTROL_SEG_CONFIGS["V10_1"])
CONTROL_SEG_CONFIGS["V10_2"]["ORIG_POST_CHAR_FORUMLIKE"] = ORIG_POST_CHAR_FORUMLIKE_V10_2

for k in CONTROL_SEG_CONFIGS.keys():
    CONTROL_SEG_CONFIGS[k]["numbered_phrases"] = {
        CONTROL_SEG_CONFIGS[k][k2]
        for k2 in ["asked_word", "replied_word", "op_word", "reblogger_word"]
    }

# DEFAULT_CSC = CONTROL_SEG_CONFIGS["V9"]
# DEFAULT_CSC = CONTROL_SEG_CONFIGS["V10"]

# works with v12_5 model
DEFAULT_CSC = CONTROL_SEG_CONFIGS["V10_1"]

# required by pre-v12_5 model
# DEFAULT_CSC = CONTROL_SEG_CONFIGS["V10_2"]


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


# TODO: (cleanup) (nwo) fix double-matching on "#1 xx posted" and "xx posted"
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
        ORIG_POST_CHAR_FORUMLIKE_V10_1,
        REVIEW_CHAR_FORUMLIKE,
        REVIEW_CHAR_FORUMLIKE_V10,
        REVIEW_CHAR_FORUMLIKE_V10_1,
        ORIG_FICTION_CHAR_FORUMLIKE,
        ORIG_FICTION_CHAR_FORUMLIKE_V10,
        ORIG_FICTION_CHAR_FORUMLIKE_V10_1,
    ]  # no tchar
    control_chars.extend(
        list({c.replace(control_seg_config["user_name"], en) for c in control_chars for en in extra_names})
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


find_control_chars = find_control_chars_forumlike


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
    if not NOSPACE:
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
