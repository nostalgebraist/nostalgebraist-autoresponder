from api_tumblr.tumblr_parsing import TumblrThread, TumblrPost
from tumblr_to_text.nwo import expand_asks


def apply_nost_identity_ouroboros(thread: TumblrThread):
    has_ask, posts_with_ask = expand_asks(thread)

    names = {post.blog_name for post in posts_with_ask}

    if 'nostalgebraist' in names and 'nostalgebraist-autoresponder' not in names:
        new_posts = []
        for post in thread.posts:
            if post.blog_name == 'nostalgebraist':
                new_posts.append(
                    TumblrPost(blog_name='nostalgebraist-autoresponder',
                               content=post.content,
                               tags=post.tags
                               )
                )
            else:
                new_posts.append(post)
        thread = TumblrThread(posts=new_posts, timestamp=thread.timestamp)
    return thread
