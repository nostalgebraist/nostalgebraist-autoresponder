def mockup_xkit_reply(
    post_url: str,
    post_summary: str,
    reply_blog_name: str,
    reply_blog_url: str,
    reply_body: str,
):
    post_summary = post_summary.replace("\n", " ")
    reply_body = f'<a class="tumblelog" href="{reply_blog_url}">@{reply_blog_name}</a> replied to your post <a href="{post_url}">“{post_summary}”</a><p><blockquote>{reply_body}</blockquote></p>'

    return reply_body


def post_body_find_reply_data(post_body: str):
    seg1, _, seg2 = post_body.partition(' replied to your post <a href="')

    url = seg2.partition(">")[0]
    urlseg = url.partition("/post/")[2]
    pid = urlseg.rstrip("\"").partition('/')[0]

    replier_seg = seg1
    if replier_seg.endswith(">"):
        replier_seg = replier_seg.rpartition("<")[0]

    replier_name = replier_seg.rpartition("@")[2]

    try:
        return int(pid), replier_name
    except:
        msg = f"couldn't extract reply metadata from\n{post_body}\nwith"
        msg += f"\n\tseg={seg1}\n\tseg2={seg2}\n\turl={url}"
        msg += f"\n\turlseg={urlseg}\n\tpid={pid}"
        print(msg)
        return None, None
