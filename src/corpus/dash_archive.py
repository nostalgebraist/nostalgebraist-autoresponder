from typing import Optional

from tumblr_to_text.classic.autoresponder_static import EOT

from api_tumblr.tumblr_parsing import TumblrThread
from api_tumblr.client_pool import ClientPool

from util.error_handling import LogExceptionAndSkip

from config.bot_config import BotSpecificConstants
NO_SCRAPE_USERS = BotSpecificConstants.load().NO_SCRAPE_USERS


def handle_no_commentary_and_populate_tags(thread: TumblrThread,
                                           client_pool: Optional[ClientPool] = None,
                                           allow_posts_with_unrecoverable_tags=True):
    # import inside b/c it loads image cache
    from tumblr_to_text.nwo_munging import pop_reblog_without_commentary, set_tags

    skip = False

    final_post = thread.posts[-1]
    if len(thread.posts) > 1 and len(final_post.content.blocks) == 0:
        # reblog w/o comment
        thread = pop_reblog_without_commentary(thread)
        final_post = thread.posts[-1]

        if final_post.blog_name in NO_SCRAPE_USERS or final_post.blog_name.startswith("artist"):
            print(f"archive: skipping, name={final_post.blog_name}", end=" ")
            skip = True
            return thread, skip

        if not client_pool and allow_posts_with_unrecoverable_tags:
            return thread, skip

        try:
            tags = client_pool.get_client().posts(final_post.blog_name, id=final_post.id)['posts'][0]['tags']
            thread = set_tags(thread, tags)
        except (KeyError, IndexError):
            print("archive: OP deleted?", end=" ")
            if not allow_posts_with_unrecoverable_tags:
                skip = True

    return thread, skip


def archive_to_corpus(post_payload, path, separator=EOT, client_pool: Optional[ClientPool] = None,
                      allow_posts_with_unrecoverable_tags=True):
    # import inside b/c it loads image cache
    from tumblr_to_text.nwo import npf_thread_to_formatted_text

    with LogExceptionAndSkip("archive post to corpus"):
        thread = TumblrThread.from_payload(post_payload)

        thread, skip = handle_no_commentary_and_populate_tags(
            thread, client_pool, allow_posts_with_unrecoverable_tags
        )
        if skip:
            return

        doc = npf_thread_to_formatted_text(thread)

        if separator in doc:
            raise ValueError(f"separator in doc: {repr(doc)}")

        with open(path, "a", encoding="utf-8") as f:
            line = doc + EOT
            f.write(line)


def dedup_join_save():
    with open("data/dash_post_dump_nost.txt", "r", encoding="utf-8") as f:
        ds1 = f.read()

    with open("data/dash_post_dump_frank.txt", "r", encoding="utf-8") as f:
        ds2 = f.read()

    docs = {d for d in ds1.split(EOT) if len(d) > 0}
    docs.update({d for d in ds2.split(EOT) if len(d) > 0})

    ds_out = EOT.join(docs)

    with open("data/dedup_join_dash_scrape.txt", "w", encoding="utf-8") as f:
        f.write(ds_out)


if __name__ == "__main__":
    dedup_join_save()
