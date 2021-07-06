import datetime

from api_tumblr.tumblr_parsing import *

# TODO: (cleanup) break dependency on old munging code files
from munging.autoresponder_static_v8 import *
from munging.munging_shared import find_images_and_sub_text
import munging.reblogs_v5


def post_payload_to_formatted_text(post_payload: dict, control_seg_config: dict = DEFAULT_CSC):
    return npf_thread_to_formatted_text(
        TumblrThread.from_payload(post_payload),
        control_seg_config=control_seg_config
    )


def npf_thread_to_formatted_text(thread: TumblrThread, control_seg_config: dict = DEFAULT_CSC):
    is_ask = [False for _ in thread.posts]

    has_ask = thread.ask_content is not None
    posts_with_ask = thread.posts

    if has_ask:
        posts_with_ask = [thread.ask_content] + posts_with_ask
        is_ask = [True] + is_ask

    is_single_original_post = len(posts_with_ask) == 1

    formatted_posts = [
        _npf_post_to_formatted_text(
            post,
            thread_index,
            timestamp=thread.timestamp,
            is_ask=is_ask,
            is_ask_reply=has_ask and thread_index == 1,
            is_single_original_post=is_single_original_post,
            is_final_post_in_thread=thread_index == len(posts_with_ask) - 1,
            control_seg_config=control_seg_config,
        )
        for thread_index, (post, is_ask) in enumerate(zip(posts_with_ask, is_ask))
    ]

    formatted_text = "\n\n".join(formatted_posts)

    conversational_prefix = format_segment_v8_interlocutors(formatted_text)

    # TODO: [cleanup] move inside the function when safe
    conversational_prefix = (
        " " + conversational_prefix.rstrip(" ") + " |\n"
        if len(conversational_prefix) > 0
        else ""
    )

    formatted_text = conversational_prefix + formatted_text
    formatted_text = normalize_for_generator(formatted_text)
    formatted_text = formatted_text.rstrip(" ")

    return formatted_text


def _npf_post_to_formatted_text(post: TumblrPost,
                                thread_index: int,
                                timestamp: int,
                                is_ask: bool,
                                is_ask_reply: bool,
                                is_single_original_post: bool,
                                is_final_post_in_thread: bool,
                                control_seg_config: dict = DEFAULT_CSC,
                                ):
    user_name = post.blog_name

    content = post.to_html()

    # strips or modifies certain html tags, adds whitespace after certain html tags
    content = _format_and_normalize_post_html(content)

    tags = getattr(post, 'tags', [])

    return _post_structural_elements_to_text(
        user_name=user_name,
        content=content,
        thread_index=thread_index,
        tags=tags,
        timestamp=timestamp,
        is_ask=is_ask,
        is_ask_reply=is_ask_reply,
        is_single_original_post=is_single_original_post,
        is_final_post_in_thread=is_final_post_in_thread,
        control_seg_config=control_seg_config
    )


def _post_structural_elements_to_text(
        user_name: str,
        content: str,
        tags: list,
        timestamp: int,
        thread_index: int,
        is_ask: bool,
        is_ask_reply: bool,
        is_single_original_post: bool,
        is_final_post_in_thread: bool,
        control_seg_config: dict = DEFAULT_CSC,
):
    if is_single_original_post:
        name_formatted = control_seg_config['ORIG_POST_CHAR_NAMED'].format(user_name=user_name)
    else:
        if is_ask:
            verb = control_seg_config['asked_word']
        elif is_ask_reply:
            verb = control_seg_config['replied_word']
        elif thread_index == 0:
            verb = control_seg_config['op_word']
        else:
            verb = control_seg_config['reblogger_word']

        # TODO: [cleanup] include in csc
        name_formatted = f"#{thread_index + 1} {user_name} {verb}:\n\n"

    if is_final_post_in_thread:
        v10_timestamp = timestamp_to_v10_format(datetime.fromtimestamp(timestamp))
        timestamp_formatted = control_seg_config['posted_at'].format(time_text=v10_timestamp)

        tag_list_formatted = ", ".join(["#" + t.rstrip(" ") for t in tags])

        tags_formatted = control_seg_config['user_tagged_post'].format(
            user_name=user_name, ftags=tag_list_formatted
        )
        # TODO: [cleanup] include implicit rstrip in csc
        # we need it because "tags: " has a training space in the 0 tags case
        tags_formatted = tags_formatted.rstrip(" ")

        final_post_content_formatted = " " + timestamp_formatted + " | " + tags_formatted + "\n"
    else:
        final_post_content_formatted = ""

    formatted_text = name_formatted + final_post_content_formatted + content

    return formatted_text


def _format_and_normalize_post_html(content):
    no_href_classes = {
        "tmblr-truncated-link",
        "tumblr_blog",
        "notification_target",
        "post_info_link",
        "tumblelog",
    }

    def _strip_no_href_classes(m):
        if any([c in m.group(1) for c in no_href_classes]):
            return m.group(2)
        return m.group(0)

    # remove certain classes of <a> tags
    content = re.sub(r'(<a [^>]+)>(.*?)</a>', _strip_no_href_classes, content)

    # strip classes other than "href" from remaining <a> tags
    content = re.sub(r'<a href=("[^\"]*")[^>]*>', r'<a href=\g<1>>', content)

    # add newline after certain tags
    for tag in munging.reblogs_v5.NEWLINE_AFTER:
        content = content.replace(f"</{tag}>", f"</{tag}>\n")

    # add two newlines after certain tags
    for tag in munging.reblogs_v5.DOUBLE_NEWLINE_AFTER:
        content = content.replace(f"</{tag}>", f"</{tag}>\n\n")

    # strip classes from some tags
    for tag in munging.reblogs_v5.INCLUDE_TAGNAME:
        content = re.sub(fr"<{tag} [^>]*>", tag, content)

    # TODO [cleanup]: just make a new set for this
    approved_tags = munging.reblogs_v5.INCLUDE_VERBATIM.union(
        munging.reblogs_v5.INCLUDE_TAGNAME
    ).union(
        {"img", "figure", "a"}
    )

    def _strip_non_approved_tags(m):
        if m.group(1) in approved_tags:
            return m.group(0)
        return ""

    # removes tags not specified in `approved_tags`
    content = re.sub(r"<[/]*([a-z0-9]*)[^><]*>",
                     _strip_non_approved_tags,
                     content)

    # apply OCR and replaces <img> and <figure> tags with text from the image
    content = find_images_and_sub_text(content)

    # the following lines make whitespace tweaks to match quirks of the previous implementation

    content = content.rstrip("\n")

    if not any(content.startswith(pre) for pre in {"[", "<", "\n"}):
        content = " " + content

    return content
