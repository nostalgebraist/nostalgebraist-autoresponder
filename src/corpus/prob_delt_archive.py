import jsonlines


PROB_DELT_ARCHIVE_PATH = "data/prob_delt_archive.jsonl"


def archive_prob_delt(kind, user, substring, prob_delt, path=PROB_DELT_ARCHIVE_PATH):
    with jsonlines.open(path, mode='a') as f:
        f.write(dict(kind=kind, user=user, substring=substring, prob_delt=float(prob_delt)))
