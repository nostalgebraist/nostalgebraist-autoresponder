import jsonlines


CAPT_ARCHIVE_PATH = "data/capt_archive.jsonl"


def archive_caption(url, capt, path=CAPT_ARCHIVE_PATH):
    with jsonlines.open(path, mode='a') as f:
        f.write({url: capt})
