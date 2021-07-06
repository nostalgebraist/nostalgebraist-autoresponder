import datetime

from api_tumblr.tumblr_parsing import *
from munging.autoresponder_static_v8 import *
from munging.munging_shared import find_images_and_sub_text
import munging.reblogs_v5


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
            is_final_post_in_thread=thread_index == len(posts_with_ask)-1,
            control_seg_config=control_seg_config)
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

    return formatted_text


def _npf_post_to_formatted_text(post: TumblrPost,
                               thread_index: int,
                               timestamp: int,
                               is_ask: bool,
                               is_ask_reply: bool,
                               is_single_original_post: bool,
                               is_final_post_in_thread: bool,
                               control_seg_config: dict = DEFAULT_CSC):
    user_name = post.blog_name
    print(repr(post.to_html()))

    content = post.to_html()

    for tag in munging.reblogs_v5.NEWLINE_AFTER:
        content = content.replace(f"</{tag}>", f"</{tag}>\n")
    for tag in munging.reblogs_v5.DOUBLE_NEWLINE_AFTER:
        content = content.replace(f"</{tag}>", f"</{tag}>\n\n")
    for tag in munging.reblogs_v5.INCLUDE_TAGNAME:
        content = re.sub(fr"<{tag} [^>]*>", tag, content)

    approved_tags = munging.reblogs_v5.INCLUDE_VERBATIM.union(munging.reblogs_v5.INCLUDE_TAGNAME).union({"img", "figure"})
    def strip_non_approved_tags(m):
        # print((m.group(0), m.group(1)))
        if m.group(1) in approved_tags:
            return m.group(0)
        return ""

    content = re.sub(r"<[/]*([a-z]*)[^>]*>",
                     strip_non_approved_tags,
                     content)

    # content = content.replace("</p><p>", "\n")
    # content = content.replace("<p>", "")
    # content = content.replace("</p>", "")
    # content = content.replace("<br>", "\n")
    content = find_images_and_sub_text(content)
    content = content.rstrip("\n")
    content = " " + content
    print(repr(content))

    # TODO: [cleanup] cleanup
    # if content.startswith(" \n"):
    #     content = content[1:]
    # content = content.replace("\n \n", "\n\n")
    # print(repr(content))

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
        name_formatted = f"#{thread_index+1} {user_name} {verb}:\n\n"

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
