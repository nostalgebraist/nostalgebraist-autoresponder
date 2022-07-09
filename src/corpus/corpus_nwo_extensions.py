import json
import argparse
import os
import pickle
from typing import Optional
from datetime import datetime
from functools import partial

from tqdm.autonotebook import tqdm

from util.times import fromtimestamp_pst

from api_tumblr.client_pool import ClientPool
from api_tumblr.paging import fetch_posts

from corpus.scrape_worthiness import is_scrape_worthy_when_archiving_blog

import config.bot_config_singleton
bot_specific_constants = config.bot_config_singleton.bot_specific_constants
bot_name = bot_specific_constants.blogName


def fetch(
    blog_name: str,
    n: Optional[int] = None,
    offset: int = 0,
    slow_scraping_ok=True,
    require_scrape_worthiness=True,
    needs_private_client=False,
    no_end_date=False,
    refresh=False,
):
    with open("data/corpus_winter_2020_max_scraped_ids.json", "r") as f:
        max_ids = json.load(f)
    if blog_name not in max_ids:
        print(f"blog_name {blog_name} not found!")

    stop_at_id = max_ids.get(blog_name, 0)
    if refresh:
        stop_at_id = 0
    print(f"using stop_at_id={stop_at_id}")

    os.makedirs("data/corpus_nwo_extensions/", exist_ok=True)

    path_ident = f"{blog_name}__refresh" if refresh else blog_name
    raw_posts_path = os.path.join("data/corpus_nwo_extensions/", f"{path_ident}_raw_posts.pkl.gz")

    posts = []
    before = None

    if os.path.exists(raw_posts_path):
        with open(raw_posts_path, "rb") as f:
            posts = pickle.load(f)

        min_ts_posix = min(pp["timestamp"] for pp in posts)
        min_ts = fromtimestamp_pst(min_ts_posix).isoformat()
        max_id = max(pp['id'] for pp in posts)
        print(f"loaded {len(posts)} raw posts, min ts {min_ts}, max id {max_id}")

        if no_end_date:
            stop_at_id = max_id
        else:
            before = min_ts_posix

    print(f"using before={before}, offset={offset}")

    pool = ClientPool()

    screener = None
    if require_scrape_worthiness:
        screener = partial(is_scrape_worthy_when_archiving_blog,
                           slow_scraping_ok=slow_scraping_ok)

    new_posts = fetch_posts(
        pool,
        blog_name,
        n,
        offset,
        needs_private_client=needs_private_client,
        stop_at_id=stop_at_id,
        before=before,
        screener=screener
    )

    posts.extend(new_posts)

    min_ts_posix = min(pp["timestamp"] for pp in posts)
    min_ts = fromtimestamp_pst(min_ts_posix).isoformat()

    print(f"saving {len(posts)} raw posts, min ts {min_ts}")

    with open(raw_posts_path, "wb") as f:
        pickle.dump(posts, f)


def process(blog_name, scrape_format_v2=False, no_end_date=False, refresh=False, n=None,
            save_image_cache=False):
    from corpus.dash_archive import archive_to_corpus

    import multimodal.image_analysis_singleton

    path_ident = f"{blog_name}__refresh" if refresh else blog_name
    processed_after_path = os.path.join("data/corpus_nwo_extensions/", f"processed_after_{path_ident}.json")

    processed_after = None
    processed_before = None

    if os.path.exists(processed_after_path):
        with open(processed_after_path, "r") as f:
            meta = json.load(f)
        processed_after = meta["processed_after"]
        processed_after_ts = fromtimestamp_pst(processed_after)

        processed_before = meta.get("processed_before")
        if processed_before:
            processed_before_ts = fromtimestamp_pst(processed_before)

    raw_posts_path = os.path.join("data/corpus_nwo_extensions/", f"{path_ident}_raw_posts.pkl.gz")

    with open(raw_posts_path, "rb") as f:
        posts = pickle.load(f)

    min_ts_posix = min(pp["timestamp"] for pp in posts)
    min_ts = fromtimestamp_pst(min_ts_posix).isoformat()
    max_ts_posix = max(pp["timestamp"] for pp in posts)
    max_ts = fromtimestamp_pst(max_ts_posix).isoformat()
    print(f"loaded {len(posts)} raw posts, min ts {min_ts}, max ts {max_ts}")

    if processed_after and not no_end_date:
        posts = [pp for pp in posts if pp["timestamp"] < processed_after]
        print(f"subsetted to {len(posts)} posts before {processed_after_ts}")

    if processed_before and no_end_date:
        posts = [pp for pp in posts if pp["timestamp"] > processed_before]
        print(f"subsetted to {len(posts)} posts after {processed_before_ts}")

    archive_path = os.path.join("data/corpus_nwo_extensions/", f"{path_ident}.txt")

    if n is not None:
        posts = posts[:n]

    min_ts_posix = min(pp["timestamp"] for pp in posts)
    max_ts_posix = max(pp["timestamp"] for pp in posts)

    for pp in tqdm(posts, mininterval=0.3, smoothing=0):
        archive_to_corpus(pp, archive_path,
                          include_image_urls=scrape_format_v2,
                          include_post_identifier=scrape_format_v2,
                          )

    print()

    with open(processed_after_path, "w") as f:
        json.dump({"processed_after": min_ts_posix, "processed_before": max_ts_posix}, f)

    if save_image_cache:
        multimodal.image_analysis_singleton.IMAGE_ANALYSIS_CACHE.save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type=int, default=None)
    parser.add_argument("--blog-name", type=str, required=True)
    parser.add_argument("--save-image-cache", action="store_true")
    parser.add_argument("--fetch-only", action="store_true")
    parser.add_argument("--process-only", action="store_true")
    parser.add_argument("--needs-private-client", action="store_true")
    parser.add_argument("--slow-scraping-off", action="store_true")
    parser.add_argument("--scrape-all", action="store_true")
    parser.add_argument("--scrape-format-v2", action="store_true")
    parser.add_argument("--no-end-date", action="store_true")  # append future posts - sets before=None
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if not args.process_only:
        fetch(
            blog_name=args.blog_name,
            n=args.n,
            require_scrape_worthiness=not args.scrape_all,
            needs_private_client=args.needs_private_client,
            slow_scraping_ok=not args.slow_scraping_off,
            no_end_date=args.no_end_date,
            refresh=args.refresh,
        )

    if not args.fetch_only:
        process(
            blog_name=args.blog_name,
            n=args.n,
            scrape_format_v2=args.scrape_format_v2,
            no_end_date=args.no_end_date,
            refresh=args.refresh,
            save_image_cache=args.save_image_cache,
        )


if __name__ == "__main__":
    main()
