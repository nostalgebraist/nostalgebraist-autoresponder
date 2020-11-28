from munging_shared import UNAME_CHAR, Q_CHAR, A_CHAR
from bs4 import BeautifulSoup


def mockup_xkit_reply(post_url: str,
                      post_summary: str,
                      reply_blog_name: str,
                      reply_blog_url: str,
                      reply_body: str):
    post_summary = post_summary.replace("\n", " ")
    reply_body = f"<a class=\"tumblelog\" href=\"{reply_blog_url}\">@{reply_blog_name}</a> replied to your post  <a href=\"{post_url}\">“{post_summary}”</a><p><blockquote>{reply_body}</blockquote></p>"

    return reply_body

def bootstrap_draft_inject_reply(processed_bootstrap_draft: str,
                                 reply_blog_name: str,
                                 reply_body: str):
    result = ""

    result += processed_bootstrap_draft.rpartition(A_CHAR)[0]  # strip prompting A_CHAR

    postpend = UNAME_CHAR + reply_blog_name + Q_CHAR + reply_body + "\n\n" + A_CHAR
    result += postpend

    return result
