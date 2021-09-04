import re
from collections import UserDict

from tumblr_to_text.classic.autoresponder_static import CONTROL_SEG_CONFIGS
from experimental.corpus_text_hacks import split_forumlike_doc, get_ccs_with_fixes

DEFAULT_ASK_REPL = " ask"
DEFAULT_FIC_REPL = " fiction"
DEFAULT_REVIEW_REPL = " review"
DEFAULT_CONTROL_LINE_PREFIX = "|"

FIC_PREFIX = CONTROL_SEG_CONFIGS['V10_1']['ORIG_FICTION_CHAR_FORUMLIKE']
REVIEW_PREFIX = CONTROL_SEG_CONFIGS['V10_1']['REVIEW_CHAR_FORUMLIKE']

REPL_TIME_STRING = " 10 AM December 2017"
FIC_REVIEW_PREFIX_REPL = f""" nostalgebraist-autoresponder posted:\n\n Written{REPL_TIME_STRING} | nostalgebraist-autoresponder's tags:"""

FIC_REVIEW_CCHAR_REPLS = {FIC_PREFIX: " fiction", REVIEW_PREFIX: " review"}


class AttrDict(UserDict):
    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError(item)

    def __dir__(self):
        return dir(super(UserDict, self)) + list(self.keys())


class Toggles(AttrDict):
    def __getitem__(self, item):
        try:
            return super(AttrDict, self).__getitem__(item)
        except KeyError:
            return False


def make_clp_regex(control_line_prefix):
    return re.compile(r"(^\W)*" + re.escape(control_line_prefix))


DEFAULT_SLIMMER_OPTIONS = Toggles(
    omit_conversation_segment=True,
    nposts_use_in_orig=True,
    nposts_omit_word_posts=True,
    nposts_unspaced_num=False,
    remove_cchar_redundancies=True,
    tag_redundancy_use_hash=False,  # default: use comma
    tags_no_uname=True,
    omit_posted_words=True,
    asked_repl=" ask",
    control_line_prefix="|",
    remove_control_line_prefix_from_posts=True,
    collapse_final_post_line_and_last_cchar=True,
    remove_para_prefix_spaces=True,
)

DEFAULT_SLIMMER_OPTIONS.clp_regex = make_clp_regex(DEFAULT_SLIMMER_OPTIONS.control_line_prefix)


def _split_header(before):
    header, sep_header_end, prefinal_posts = before.rpartition(" |\n")
    conversation_segment, sep_conv_nposts, nposts_segment = header.partition(" |")

    return (
        conversation_segment,
        sep_conv_nposts,
        nposts_segment,
        sep_header_end,
        prefinal_posts,
    )


def segment_prefinal_posts(header):
    # needs `header` not `prefinal_posts` b/c it uses the conversation text
    cchars = get_ccs_with_fixes(header)

    SPECIAL_TEXTEND = "__INTERNAL__TEXTEND"

    cchars.append((SPECIAL_TEXTEND, len(header)))

    segments = []

    for cc1, cc2 in zip(cchars[:-1], cchars[1:]):
        segments.append({"kind": "cc", "text": cc1[0]})

        start_ix = cc1[1] + len(cc1[0])
        end_ix = cc2[1]

        segments.append({"kind": "content", "text": header[start_ix:end_ix]})

    return segments


def slim_forumlike_doc(doc: str, options: Toggles = DEFAULT_SLIMMER_OPTIONS, verbose=True):
    control_line_prefix = (
        options.control_line_prefix
        if options.control_line_prefix
        else DEFAULT_CONTROL_LINE_PREFIX
    )

    l_control_line_prefix = len(control_line_prefix)

    def _add_control_line_prefix(s):
        if not control_line_prefix:
            return s
        if " " not in control_line_prefix[:-1] + s[:1]:
            return control_line_prefix + " " + s
        return control_line_prefix + s

    def _fallback(s):
        return s

    if options.remove_control_line_prefix_from_posts:
        clp_regex = (
            options.clp_regex
            if options.clp_regex
            else make_clp_regex(control_line_prefix)
        )

        def _remove_control_line_prefix(s):
            return clp_regex.sub("\1", s)

    else:
        _remove_control_line_prefix = _fallback

    if options.remove_para_prefix_spaces:

        def _remove_para_prefix_spaces(s):
            return "\n".join(l.lstrip(" ") for l in s.split("\n"))

    else:
        _remove_para_prefix_spaces = _fallback

    def _munge_post_content(s):
        return _remove_para_prefix_spaces(_remove_control_line_prefix(s))

    # split

    special_doctype = None
    for p in (FIC_PREFIX, REVIEW_PREFIX):
        if doc.startswith(p):
            special_doctype = p
            doc = doc.replace(p, FIC_REVIEW_PREFIX_REPL)

    (
        header,
        sep,
        time_segment,
        sep2,
        _,
        _,
        _,
        _,
        tag_segment,
        sep3,
        final_content,
    ) = split_forumlike_doc(doc)

    (
        conversation_segment,
        sep_conv_nposts,
        nposts_segment,
        sep_header_end,
        prefinal_posts,
    ) = _split_header(header)

    prefinal_segments = segment_prefinal_posts(header)

    # debug
    prefinal_text = "".join([pfseg["text"] for pfseg in prefinal_segments])

    if prefinal_text != prefinal_posts:
        if verbose:
            print("doc: " + repr(doc))
            print("prefinal_text: " + repr(prefinal_text))
            print("prefinal_posts: " + repr(prefinal_posts))
        raise ValueError("prefinal_text != prefinal_posts")

    _tag_prefix_a, _tag_prefix_b, tags_proper = tag_segment.rpartition(":")
    tag_prefix = _tag_prefix_a + _tag_prefix_b
    tags_proper = tags_proper.lstrip(" ")

    # slim

    # slim: header line
    if options.omit_conversation_segment:
        conversation_segment = ""
        sep_conv_nposts = ""

    if options.nposts_use_in_orig and not nposts_segment:
        nposts_segment = " 1 posts"
        sep_header_end = " |\n"

    if options.nposts_omit_word_posts:
        nposts_segment, _, _ = nposts_segment.rpartition(" posts")

    if options.nposts_unspaced_num:
        nposts_segment = nposts_segment.lstrip(" ")

    # slim: redund
    if options.remove_cchar_redundancies:
        sep_header_end = "\n"  # skips space+pipe in " |\n"
        sep = ""  # skips "Written" before time
        sep2 = " "  # skips pipe between time and tags

        if options.tag_redundancy_use_hash:
            tags_proper = tags_proper.replace(", ", " ")  # no commas between tags
        else:
            tags_proper = tags_proper.replace("#", "")  # separate tags

    # slim: tags
    if options.tags_no_uname:
        tag_prefix = ""

    tag_segment = tag_prefix + tags_proper

    if len(tag_segment) == 0:
        sep2 = ""  # removes training space on final post control line

    # slim: cchars

    slimmed_prefinal_segments = []

    for pfseg in prefinal_segments:
        if pfseg["kind"] == "content":
            slimmed_prefinal_segments.append(
                {"kind": pfseg["kind"], "text": _munge_post_content(pfseg["text"])}
            )
        elif pfseg["kind"] == "cc":
            cctext = pfseg["text"]

            if options.nposts_use_in_orig and not cctext.startswith("#"):
                cctext = "#1" + cctext

            if options.omit_posted_words:
                ccwords = cctext.split(" ")

                word_to_slim = ccwords[-1]
                if word_to_slim.startswith("asked"):
                    asked_repl = (
                        options.asked_repl if options.asked_repl else DEFAULT_ASK_REPL
                    )
                    word_to_slim = word_to_slim.replace("asked", asked_repl)
                elif word_to_slim.startswith("posted") and special_doctype:
                    fic_review_repl = FIC_REVIEW_CCHAR_REPLS[special_doctype]
                    word_to_slim = word_to_slim.replace("posted", fic_review_repl)
                else:
                    word_to_slim = word_to_slim[word_to_slim.find(":") :]

                cctext = " ".join(ccwords[:-1]) + word_to_slim

            if options.remove_cchar_redundancies:
                cctext = "".join(cctext.rsplit(":", maxsplit=1)).lstrip("#")

            cctext = _add_control_line_prefix(cctext)

            slimmed_prefinal_segments.append({"kind": pfseg["kind"], "text": cctext})
        else:
            raise ValueError(pfseg["kind"])

    prefinal_text = "".join([pfseg["text"] for pfseg in slimmed_prefinal_segments])

    # join
    header_line = "".join(
        [conversation_segment, sep_conv_nposts, nposts_segment, sep_header_end]
    )

    final_post_line = "".join([sep, time_segment, sep2, tag_segment, sep3])

    # fix
    _prefinal_rchar = prefinal_text[-1]
    if _prefinal_rchar == " ":
        prefinal_text = prefinal_text[:-1]
        final_post_line = _prefinal_rchar + final_post_line
    # end of fix

    final_content = _munge_post_content(final_content)

    final_post_line = _add_control_line_prefix(final_post_line)

    if options.collapse_final_post_line_and_last_cchar:
        prefinal_text = prefinal_text.rstrip("\n ") + " "
        final_post_line = final_post_line.rstrip(" \n") + "\n\n"

    if special_doctype:
        final_post_line = final_post_line.replace(REPL_TIME_STRING, "")

    header_line = _add_control_line_prefix(header_line)

    joined = "".join([header_line, prefinal_text, final_post_line, final_content])
    return joined
