from munging.autoresponder_static import EOT

from experimental.nwo import post_payload_to_formatted_text

from util.error_handling import LogExceptionAndSkip


def archive_to_corpus(post_payload, path, separator=EOT):
    with LogExceptionAndSkip("archive post to corpus"):
        doc = post_payload_to_formatted_text(post_payload)

        if separator in doc:
            raise ValueError(f"separator in doc: {repr(doc)}")

        with open(path, "a", encoding="utf-8") as f:
            line = doc + EOT
            f.write(line)
