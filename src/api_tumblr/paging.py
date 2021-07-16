from typing import Optional
from datetime import datetime

from tqdm.autonotebook import tqdm

from api_tumblr.client_pool import ClientPool

from util.error_handling import LogExceptionAndSkip

from config.bot_config import BotSpecificConstants
bot_name = BotSpecificConstants.load().blogName


# TODO: DRY (centralize paging helpers)
def fetch_next_page(client, offset, limit=50, blog_name: str = bot_name, before=None):
    kwargs = dict(limit=limit)
    if before:
        kwargs["before"] = before
    else:
        kwargs["offset"] = offset
    response = client.posts(blog_name, **kwargs)
    posts = response["posts"]
    total_posts = response["total_posts"]

    next_offset = None
    with LogExceptionAndSkip("get next offset for /posts"):
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
                report_cadence=5000,
                needs_private_client=False,
                needs_dash_client=False,
                stop_at_id=0,
                before=None):
    posts = []
    ids = set()
    since_last_report = 0

    tqdm_bar = None

    if needs_private_client and needs_dash_client:
        raise ValueError("fetch_posts: only one of needs_private_client and needs_dash_client can be true")

    client_getter = pool.get_client
    if needs_private_client:
        client_getter = pool.get_private_client
    if needs_dash_client:
        client_getter = pool.get_dashboard_client

    while True:
        client = client_getter()
        page, next_offset, total_posts = fetch_next_page(client, offset=offset, blog_name=blog_name, before=before)
        before = None

        if not tqdm_bar:
            tqdm_bar = tqdm(total=total_posts)
            tqdm_bar.update(offset)
            tqdm_bar.set_postfix(cl=pool.client_name(client))

        if (len(page) == 0) or (next_offset == offset):
            print(f"stopping, empty page after {len(posts)} posts")
            return posts

        since_last_report += len(page)
        if since_last_report >= report_cadence:
            pool.report()
            since_last_report = 0

        nraw = len(page)
        page = [pp for pp in page if pp['id'] not in ids]
        ndedup = len(page)

        page = [pp for pp in page
                if pp['id'] > stop_at_id
                or pp.get('is_pinned')  # pins make id non-monotonic
                ]
        nafter = len(page)
        nbefore = ndedup - nafter

        page_ids = {pp['id'] for pp in page}

        ids.update(page_ids)
        posts.extend(page)
        offset = next_offset

        if len(page) == 0:
            min_ts = None
        else:
            min_ts = datetime.fromtimestamp(min(pp['timestamp'] for pp in page)).isoformat()
        tqdm_bar.update(len(page))
        tqdm_bar.set_postfix(cl=pool.client_name(client), min_ts=min_ts)

        max_n = total_posts
        if n:
            max_n = min(n, max_n)

        if len(posts) >= max_n:
            print(f"stopping with {len(posts)} posts: reached maximum {max_n}")
            return posts

        if nbefore > 0:
            print(f"stopping with {len(posts)} posts: {nbefore}/{ndedup} in current page are before id {stop_at_id}")
            return posts
