import random

from tumblr_to_text.classic.munging_shared import get_body
from persistence.response_cache import PostIdentifier


# TODO: DRY
def is_scrape_worthy_when_archiving_blog(
    post_payload, slow_scraping_ok=True
):
    post_identifier = PostIdentifier(post_payload["blog_name"], str(post_payload["id"]))

    has_comment = True
    if "reblog" in post_payload:
        comment_ = post_payload["reblog"].get("comment", "")
        has_comment = len(comment_) > 0

    if not has_comment:
        msg = "no comment"
        return False, msg, post_identifier

    if post_payload.get("type") in {
        "video",
    }:
        msg = "is video"
        return False, msg, post_identifier

    blocks = post_payload['content'] + [bl
                                        for entry in post_payload.get("trail", [])
                                        for bl in entry.get('content', [])]
    block_types = {bl['type'] for bl in blocks}
    if "text" not in block_types:
        msg = f"\trejecting {post_identifier}: no text blocks\n{block_types}"
        return False, msg, post_identifier

    try:
        p_body = get_body(post_payload)
    except Exception as e:
        msg = "parse error"
        return False, msg, post_identifier

    n_img = len(p_body.split("<img")) - 1
    if n_img > 2:
        msg = f"too many images ({n_img})"
        return False, msg, post_identifier

    if '.gif' in p_body:
        msg = "has gif"
        return False, msg, post_identifier

    if (not slow_scraping_ok) and (n_img > 0):
        msg = "has imgs and slow_scraping_ok=False"
        return False, msg, post_identifier

    if n_img > 0 and random.random() > 0.333:
        msg = "has imgs and roll failed"
        return False, msg, post_identifier

    return True, "", post_identifier
