import os
import json
from functools import partial

from tqdm.autonotebook import tqdm

from experimental.decision_encoding import (
    prep_for_selector,
    prep_for_sentiment,
    unique_id_for_doc,
)


def run_model_on_docs(
    docs,
    prep_fn,
    predict_fn,
    save_path,
    save_dir="data/decision_encoding",
    batch_size=8,
):
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, save_path)

    results = {}
    if os.path.exists(save_path):
        results = json.load(open(save_path))

    uids = {d: unique_id_for_doc(d) for d in docs}
    docs = [d for d in docs if uids[d] not in results]

    batches = []
    for i in range(0, len(docs), batch_size):
        batches.append(docs[i : i + batch_size])

    if sum([len(b) for b in batches]) < len(docs):
        batches.append(docs[i:])

    for batch in tqdm(batches, mininterval=1, smoothing=0):
        batch_prepped = [prep_fn(d) for d in batch]
        probs = predict_fn(batch_prepped)
        for d, p in zip(batch, probs):
            results[uids[d]] = p

    with open(save_path, "w") as f:
        json.dump(results, f)

    return results


def run_selector_on_docs(docs, save_path="selector.json", batch_size=8):
    import experimental.ml_connector

    return run_model_on_docs(
        docs,
        prep_fn=prep_for_selector,
        predict_fn=partial(
            experimental.ml_connector.selection_proba_from_gpt2_service,
            already_forumlike=True,
        ),
        save_path=save_path,
        batch_size=batch_size,
    )


def run_sentiment_on_docs(docs, save_path="sentiment.json", batch_size=8):
    import experimental.ml_connector

    return run_model_on_docs(
        docs,
        prep_fn=prep_for_sentiment,
        predict_fn=experimental.ml_connector.sentiment_logit_diffs_from_gpt2_service,
        save_path=save_path,
        batch_size=batch_size,
    )


def run_selector_on_docs_local(
    docs, save_path="selector.json", batch_size=8, device="cuda:0", selector_est=None
):
    if not selector_est:
        import experimental.ml_layer_torch
        import experimental.ml_connector
        experimental.ml_layer_torch.selector_est.model_.to(device)
        experimental.ml_layer_torch.sentiment_est.model_.cpu()
        experimental.ml_layer_torch.autoreviewer_est.model_.cpu()
        selector_est = experimental.ml_layer_torch.selector_est
    else:
        import experimental.ml_connector

    def monkeypatched_selector_do(method, *args, repeat_until_done_signal=False, **kwargs):
        out = getattr(selector_est, method)(*args, **kwargs)
        return [{"result": out}]

    experimental.ml_connector.selector_est.do = monkeypatched_selector_do

    return run_selector_on_docs(docs, save_path=save_path, batch_size=batch_size)


def run_sentiment_on_docs_local(
    docs, save_path="sentiment.json", batch_size=8, device="cuda:0", sentiment_est=None
):
    if not sentiment_est:
        import experimental.ml_connector
        import experimental.ml_layer_torch

        experimental.ml_layer_torch.sentiment_est.model_.to(device)
        experimental.ml_layer_torch.selector_est.model_.cpu()
        experimental.ml_layer_torch.autoreviewer_est.model_.cpu()
        sentiment_est = experimental.ml_layer_torch.sentiment_est
    else:
        import experimental.ml_connector

    def monkeypatched_sentiment_do(method, *args, repeat_until_done_signal=False, **kwargs):
        out = getattr(sentiment_est, method)(*args, **kwargs)
        return [{"result": out}]

    experimental.ml_connector.sentiment_est.do = monkeypatched_sentiment_do

    return run_sentiment_on_docs(docs, save_path=save_path, batch_size=batch_size)
