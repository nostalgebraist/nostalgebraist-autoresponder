from typing import Optional

from tqdm.autonotebook import tqdm

from api_tumblr.client_pool import ClientPool

from util.error_handling import LogExceptionAndSkip

from config.bot_config import BotSpecificConstants
bot_name = BotSpecificConstants.load().blogName


# TODO: DRY (centralize paging helpers)
def fetch_next_page(client, offset, limit=50, blog_name: str = bot_name):
    response = client.posts(blog_name, limit=limit, offset=offset)
    posts = response["posts"]
    total_posts = response["total_posts"]

    next_offset = None
    with LogExceptionAndSkip("get next offset for /posts"):
        print(response)
        next_offset = int(response["_links"]["next"]["query_params"]["offset"])
    if next_offset is None:
        print((next_offset, offset, len(posts)))
        next_offset = offset + len(posts)  # fallback
        print(
            f"falling back to: old offset {offset} + page size {len(posts)} = {next_offset}"
        )
    return posts, next_offset, total_posts


def fetch_posts(pool: ClientPool,
                blog_name: str = bot_name,
                n: Optional[int] = None,
                offset: int = 0,
                report_cadence=5000):
    posts = []
    ids = set()
    ndup = 0
    since_last_report = 0

    tqdm_bar = None

    while True:
        client = pool.get_client()
        page, next_offset, total_posts = fetch_next_page(client, offset=offset, blog_name=blog_name)

        if not tqdm_bar:
            tqdm_bar = tqdm(total=total_posts)
            tqdm_bar.set_postfix(offset=offset, ndup=ndup, cl=pool.client_name(client))

        if (len(page) == 0) or (next_offset == offset):
            print(f"stopping, empty page after {len(posts)} posts")
            return posts

        since_last_report += len(page)
        if since_last_report >= report_cadence:
            pool.report()
            since_last_report = 0

        nraw = len(page)
        page = [pp for pp in page if pp['id'] not in ids]
        ndup += len(page) - nraw

        page_ids = {pp['id'] for pp in page}

        ids.update(page_ids)
        posts.extend(page)
        offset = next_offset

        tqdm_bar.update(len(page))
        tqdm_bar.set_postfix(offset=offset, ndup=ndup, cl=pool.client_name(client))

        max_n = total_posts
        if n:
            max_n = min(n, max_n)

        if len(posts) >= max_n:
            print(f"stopping, reached maximum {max_n} with {len(posts)} posts")
            return posts
