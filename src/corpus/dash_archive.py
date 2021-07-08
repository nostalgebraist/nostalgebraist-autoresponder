from typing import Optional
from pytumblr import TumblrRestClient

from tumblr_to_text.classic.autoresponder_static import EOT

from api_tumblr.tumblr_parsing import TumblrThread
from tumblr_to_text.nwo import npf_thread_to_formatted_text
from tumblr_to_text.nwo_munging import pop_reblog_without_commentary, set_tags

from util.error_handling import LogExceptionAndSkip

from config.bot_config import BotSpecificConstants
NO_SCRAPE_USERS = BotSpecificConstants.load().NO_SCRAPE_USERS


def handle_no_commentary_and_populate_tags(thread: TumblrThread, client: Optional[TumblrRestClient] = None):
    skip = False

    if not client:
        skip = True
        return thread, skip

    final_post = thread.posts[-1]
    if len(thread.posts) > 1 and len(final_post.content.blocks) == 0:
        # reblog w/o comment
        thread = pop_reblog_without_commentary(thread)
        final_post = thread.posts[-1]

        if final_post.blog_name in NO_SCRAPE_USERS or final_post.blog_name.startswith("artist"):
            print(f"archive: skipping, name={final_post.blog_name}", end=" ")
            skip = True
            return thread, skip
        try:
            tags = client.posts(final_post.blog_name, id=final_post.id)['posts'][0]['tags']
            thread = set_tags(thread, tags)
        except (KeyError, IndexError):
            print("archive: skipping, OP deleted?", end=" ")
            skip = True

    return thread, skip


def archive_to_corpus(post_payload, path, separator=EOT, client: Optional[TumblrRestClient] = None):
    with LogExceptionAndSkip("archive post to corpus"):
        thread = TumblrThread.from_payload(post_payload)

        thread, skip = handle_no_commentary_and_populate_tags(thread, client)
        if skip:
            return

        doc = npf_thread_to_formatted_text(thread)

        if separator in doc:
            raise ValueError(f"separator in doc: {repr(doc)}")

        with open(path, "a", encoding="utf-8") as f:
            line = doc + EOT
            f.write(line)
