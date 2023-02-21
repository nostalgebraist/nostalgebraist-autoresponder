import re
import html as html_lib
from typing import Tuple, List, Union

from api_tumblr.tumblr_parsing import NPFAsk, TumblrPost, TumblrThread

# TODO: (cleanup) break dependency on old munging code files
from tumblr_to_text.classic.autoresponder_static import DEFAULT_CSC, normalize_for_generator, NOSPACE
from tumblr_to_text.classic.autoresponder_static_v8 import format_segment_v8_interlocutors, timestamp_to_v10_format
from tumblr_to_text.classic.munging_shared import sanitize_user_input_outer_shell
from tumblr_to_text.image_munging import find_images_and_sub_text
import tumblr_to_text.nwo_html_config

from util.times import fromtimestamp_pst

PostOrAsk = Union[TumblrPost, NPFAsk]


def expand_asks(thread: TumblrThread) -> Tuple[bool, List[PostOrAsk]]:
    has_ask = thread.ask_content is not None
    posts_with_ask = thread.posts

    if has_ask:
        posts_with_ask = [thread.ask_content] + posts_with_ask

    return has_ask, posts_with_ask


def post_payload_to_formatted_text(post_payload: dict,
                                   ml_prompt_format: bool = False,
                                   prob_delta_format: bool = False,
                                   include_image_urls: bool = False,
                                   include_post_identifier: bool = False,
                                   control_seg_config: dict = DEFAULT_CSC,
                                   endtags=False):
    return npf_thread_to_formatted_text(
        TumblrThread.from_payload(post_payload),
        ml_prompt_format=ml_prompt_format,
        prob_delta_format=prob_delta_format,
        include_image_urls=include_image_urls,
        include_post_identifier=include_post_identifier,
        control_seg_config=control_seg_config,
        endtags=endtags,
    )


def npf_thread_to_formatted_text(thread: TumblrThread,
                                 ml_prompt_format: bool = False,
                                 prob_delta_format: bool = False,
                                 include_image_urls: bool = False,
                                 include_post_identifier: bool = False,
                                 skip_image_analysis: bool = False,
                                 control_seg_config: dict = DEFAULT_CSC,
                                 endtags=False,):
    is_ask = [False for _ in thread.posts]

    has_ask, posts_with_ask = expand_asks(thread)

    if has_ask:
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
            ml_prompt_format=ml_prompt_format,
            prob_delta_format=prob_delta_format,
            include_image_urls=include_image_urls,
            skip_image_analysis=skip_image_analysis,
            control_seg_config=control_seg_config,
            endtags=endtags,
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

    if prob_delta_format:
        conversational_prefix = ""

    formatted_text = conversational_prefix + formatted_text
    formatted_text = normalize_for_generator(formatted_text)
    formatted_text = formatted_text.rstrip(" ")

    if include_post_identifier:
        last_post = posts_with_ask[-1]
        formatted_text = formatted_text + f"\nbn {last_post.blog_name} id {last_post.id}"

    return formatted_text


def _npf_post_to_formatted_text(post: TumblrPost,
                                thread_index: int,
                                timestamp: int,
                                is_ask: bool,
                                is_ask_reply: bool,
                                is_single_original_post: bool,
                                is_final_post_in_thread: bool,
                                ml_prompt_format: bool,
                                prob_delta_format: bool,
                                include_image_urls: bool,
                                skip_image_analysis: bool,
                                control_seg_config: dict = DEFAULT_CSC,
                                endtags=False,
                                ):
    user_name = post.blog_name

    content = post.to_html()

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
        ml_prompt_format=ml_prompt_format,
        prob_delta_format=prob_delta_format,
        include_image_urls=include_image_urls,
        skip_image_analysis=skip_image_analysis,
        control_seg_config=control_seg_config,
        endtags=endtags,
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
        ml_prompt_format: bool,
        prob_delta_format: bool,
        include_image_urls: bool,
        skip_image_analysis: bool,
        control_seg_config: dict = DEFAULT_CSC,
        endtags=False,
):
    if ml_prompt_format and is_final_post_in_thread:
        # if prompting, we want the model to write the content of the final post
        content = ""
        # if prompting, we want the model to write the tags
        tags = []
    else:
        # strips or modifies certain html tags, adds whitespace after certain html tags
        content = format_and_normalize_post_html(content,
                                                 include_image_urls=include_image_urls,
                                                 skip_image_analysis=skip_image_analysis)

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

        if prob_delta_format and is_final_post_in_thread:
            name_formatted = f"#{thread_index + 1}"

    if is_final_post_in_thread:
        v10_timestamp = timestamp_to_v10_format(fromtimestamp_pst(timestamp))
        timestamp_formatted = control_seg_config['posted_at'].format(time_text=v10_timestamp)

        final_post_content_formatted = ""
        if not NOSPACE:
            final_post_content_formatted += " "

        final_post_content_formatted += timestamp_formatted

        tag_list_formatted = ", ".join(["#" + t.rstrip(" ") for t in tags])

        if endtags:
            final_post_content_formatted = final_post_content_formatted + "\t\n"
        else:
            tags_formatted = control_seg_config['user_tagged_post'].format(
                user_name=user_name, ftags=tag_list_formatted
            )
            # TODO: [cleanup] include implicit rstrip in csc
            # we need it because "tags: " has a training space in the 0 tags case
            tags_formatted = tags_formatted.rstrip(" ")

            final_post_content_formatted = final_post_content_formatted + " | " + tags_formatted

            # a newline indicates the end of the tags -- if prompting the model, we want it to (optionally) write tags
            tag_suffix = "" if ml_prompt_format else "\n"
            final_post_content_formatted = final_post_content_formatted + tag_suffix
    else:
        final_post_content_formatted = ""

    if prob_delta_format:
        final_post_content_formatted = ""

    if is_final_post_in_thread and endtags:
        tag_prefix = "" if ml_prompt_format or prob_delta_format else "\n\n\t "
        tags_formatted = tag_prefix + tag_list_formatted
        formatted_text = name_formatted + final_post_content_formatted + content + tags_formatted
    else:
        formatted_text = name_formatted + final_post_content_formatted + content

    return formatted_text


def format_and_normalize_post_html(content, include_image_urls=False, skip_image_analysis=False):
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
    for tag in tumblr_to_text.nwo_html_config.NEWLINE_AFTER:
        content = content.replace(f"</{tag}>", f"</{tag}>\n")

    # add two newlines after certain tags
    for tag in tumblr_to_text.nwo_html_config.DOUBLE_NEWLINE_AFTER:
        content = content.replace(f"</{tag}>", f"</{tag}>\n\n")

    # strip classes from some tags
    for tag in tumblr_to_text.nwo_html_config.INCLUDE_TAGNAME:
        content = re.sub(fr"<{tag} [^>]*>", tag, content)

    # TODO [cleanup]: just make a new set for this
    approved_tags = tumblr_to_text.nwo_html_config.INCLUDE_VERBATIM.union(
        tumblr_to_text.nwo_html_config.INCLUDE_TAGNAME
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

    # removes problem strings like zero-width joiners and internally used separators
    content = sanitize_user_input_outer_shell(content)

    # apply OCR and replaces <img> and <figure> tags with text from the image
    content = find_images_and_sub_text(content, include_urls=include_image_urls, skip=skip_image_analysis)

    # undo html escapes now that tag munging is complete
    content = html_lib.unescape(content)

    # the following lines make whitespace tweaks to match quirks of the previous implementation

    content = content.rstrip("\n")

    if not NOSPACE:
        if not any(content.startswith(pre) for pre in {"[", "<", "\n"}):
            content = " " + content

    return content
