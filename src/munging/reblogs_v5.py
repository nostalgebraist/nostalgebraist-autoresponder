"""mysterious hacky html parsing that somehow (mostly?) successfully handles tumblr's weird pre-NPF markup format"""
from munging.autoresponder_static import Q_CHAR, V10_ASK_CHAR

from multimodal import image_analysis_singleton

image_analysis_cache = image_analysis_singleton.IMAGE_ANALYSIS_CACHE

START_DUMMY = "⭒"

V10_CHARS_TO_LEGACY_CHARS = {V10_ASK_CHAR: Q_CHAR}

ALWAYS_USE_A_CHAR_OPERATIONAL = True
TRY_LINKS_FOR_IMAGES = False
INCLUDE_HREF_FOR_A = True

EOT = "<|endoftext|>"

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
