from munging.autoresponder_static import EOT

from api_tumblr.tumblr_parsing import TumblrThread
from experimental.nwo import npf_thread_to_formatted_text
from experimental.nwo_munging import pop_reblog_without_commentary

from util.error_handling import LogExceptionAndSkip


def archive_to_corpus(post_payload, path, separator=EOT):
    with LogExceptionAndSkip("archive post to corpus"):
        thread = TumblrThread.from_payload(post_payload)

        thread = pop_reblog_without_commentary(thread)

        doc = npf_thread_to_formatted_text(thread)

        if separator in doc:
            raise ValueError(f"separator in doc: {repr(doc)}")

        with open(path, "a", encoding="utf-8") as f:
            line = doc + EOT
            f.write(line)
