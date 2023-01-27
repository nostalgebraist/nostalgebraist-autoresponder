import re

from tumblr_to_text.classic.autoresponder_static import DEFAULT_CSC

CCS_SKIP_DEFAULT = tuple(DEFAULT_CSC[k] for k in ('ORIG_FICTION_CHAR_FORUMLIKE', 'REVIEW_CHAR_FORUMLIKE'))


pat = re.compile(
    r"(?P<prev>.*)(?P<pipe_meta> \| )(?P<tag_meta>[^ ]+'s tags:)(?P<tags>[^\n]*)(?P<end_meta>\n)(?P<final_post>.*)$",
    flags=re.DOTALL
)


def move_tags_to_end(s, ccs_skip=CCS_SKIP_DEFAULT):
    if any(s.startswith(cc) for cc in CCS_SKIP_DEFAULT):
        return s
    return pat.sub(r"\g<prev>\t\n\g<final_post>\n\n\t\g<tags>", s.replace("\t", 8*" "))
