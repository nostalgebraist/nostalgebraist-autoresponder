import jsonlines


PROB_DELT_ARCHIVE_PATH = "data/prob_delt_archive.jsonl"


def archive_prob_delt(kind, user, substring, prob_delt, post_id=None, path=PROB_DELT_ARCHIVE_PATH):
    data = dict(kind=kind, user=user, substring=substring, prob_delt=float(prob_delt))
    if post_id is not None:
        data['post_id'] = str(post_id)
    with jsonlines.open(path, mode='a') as f:
        f.write(data)
