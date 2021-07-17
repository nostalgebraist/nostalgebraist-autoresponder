from typing import Optional
import random
import argparse
import os

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


def _train_val_split(docs, val_frac=0.03):
    charlen = sum(map(len, docs))
    val_charlen = val_frac * charlen

    train_docs = list(iter(docs))  # deep copy
    val_docs = []

    while sum(map(len, val_docs)) < val_charlen:
        ix = random.randint(0, len(train_docs))
        val_docs.append(train_docs.pop(ix))

    return train_docs, val_docs


def dedup_join_save(include_corpus_extensions=False, val_frac=0.03):
    with open("data/dash_post_dump_nost.txt", "r", encoding="utf-8") as f:
        ds1 = f.read()

    with open("data/dash_post_dump_frank.txt", "r", encoding="utf-8") as f:
        ds2 = f.read()

    docs = {d for d in ds1.split(EOT) if len(d) > 0}
    docs.update({d for d in ds2.split(EOT) if len(d) > 0})

    if include_corpus_extensions:
        for fn in os.listdir("data/corpus_nwo_extensions/"):
            if not fn.endswith(".txt"):
                continue
            fp = "data/corpus_nwo_extensions/" + fn
            with open(fp, "r", encoding="utf-8") as f:
                ds_xtn = f.read()
            docs.update({d for d in ds_xtn.split(EOT) if len(d) > 0})

    ds_out = EOT.join(docs)

    base_name = "dedup_join_dash_scrape_plus_xtn" if include_corpus_extensions else "dedup_join_dash_scrape"

    with open(f"data/{base_name}.txt", "w", encoding="utf-8") as f:
        f.write(ds_out)

    train_docs, val_docs = _train_val_split(list(docs), val_frac=val_frac)

    ds_out_train = EOT.join(train_docs)
    ds_out_val = EOT.join(val_docs)

    with open(f"data/{base_name}__train.txt", "w", encoding="utf-8") as f:
        f.write(ds_out_train)

    with open(f"data/{base_name}__val.txt", "w", encoding="utf-8") as f:
        f.write(ds_out_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-corpus-extensions", action="store_true")
    args = parser.parse_args()

    dedup_join_save(include_corpus_extensions=args.include_corpus_extensions)
