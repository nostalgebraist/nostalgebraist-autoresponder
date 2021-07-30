import os
import json
from functools import partial

from tqdm.autonotebook import tqdm

from experimental.decision_encoding import unique_id_for_doc
from experimental.corpus_text_hacks import now, prep_for_selector, prep_for_sentiment, prep_for_autoreviewer


def run_model_on_docs(
    docs,
    prep_fn,
    predict_fn,
    save_path,
    save_dir="data/decision_encoding",
    batch_size=8,
    recompute_existing=False,
    suppress_tqdm=False
):
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, save_path)

    results = {}
    if os.path.exists(save_path):
        results = json.load(open(save_path))

    uids = {d: unique_id_for_doc(d) for d in docs}
    if not recompute_existing:
        docs = [d for d in docs if uids[d] not in results]

    batches = []
    for i in range(0, len(docs), batch_size):
        batches.append(docs[i : i + batch_size])

    if sum([len(b) for b in batches]) < len(docs):
        batches.append(docs[i:])

    batch_iter = batches if suppress_tqdm else tqdm(batches, mininterval=1, smoothing=0)
    for batch in batch_iter:
        batch_prepped = [prep_fn(d) for d in batch]
        probs = predict_fn(batch_prepped)
        for d, p in zip(batch, probs):
            results[uids[d]] = p

        with open(save_path, "w") as f:
            json.dump(results, f)

    return results


def run_selector_on_docs(docs, save_path="selector.json", batch_size=8, recompute_existing=False, ts=now):
    import api_ml.ml_connector

    return run_model_on_docs(
        docs,
        prep_fn=partial(prep_for_selector, ts=ts),
        predict_fn=api_ml.ml_connector.selection_proba_from_gpt,
        save_path=save_path,
        batch_size=batch_size,
        recompute_existing=recompute_existing
    )


def run_sentiment_on_docs(docs, save_path="sentiment.json", batch_size=8, recompute_existing=False, ts=now):
    import api_ml.ml_connector

    return run_model_on_docs(
        docs,
        prep_fn=partial(prep_for_sentiment, ts=ts),
        predict_fn=api_ml.ml_connector.sentiment_logit_diffs_from_gpt,
        save_path=save_path,
        batch_size=batch_size,
        recompute_existing=recompute_existing
    )


def run_autoreviewer_on_docs(docs, save_path="autoreviewer.json", batch_size=8, recompute_existing=False, ts=now):
    import api_ml.ml_connector

    return run_model_on_docs(
        docs,
        prep_fn=partial(prep_for_autoreviewer, ts=ts),
        predict_fn=api_ml.ml_connector.autoreview_proba_from_gpt,
        save_path=save_path,
        batch_size=batch_size,
        recompute_existing=recompute_existing
    )


def run_selector_on_docs_local(
    docs, save_path="selector.json", batch_size=8, recompute_existing=False, device="cuda:0", selector_est=None, ts=now
):
    if not selector_est:
        import api_ml.ml_layer_torch
        import api_ml.ml_connector
        api_ml.ml_layer_torch.selector_est.model_.to(device)
        api_ml.ml_layer_torch.sentiment_est.model_.cpu()
        api_ml.ml_layer_torch.autoreviewer_est.model_.cpu()
        selector_est = api_ml.ml_layer_torch.selector_est
    else:
        import api_ml.ml_connector

    def monkeypatched_selector_do(method, *args, repeat_until_done_signal=False, **kwargs):
        out = getattr(selector_est, method)(*args, **kwargs)
        return [{"result": out}]

    api_ml.ml_connector.selector_est.do = monkeypatched_selector_do

    return run_selector_on_docs(docs, save_path=save_path, batch_size=batch_size, recompute_existing=recompute_existing,
                                ts=ts)


def run_sentiment_on_docs_local(
    docs, save_path="sentiment.json", batch_size=8, recompute_existing=False, device="cuda:0", sentiment_est=None, ts=now
):
    if not sentiment_est:
        import api_ml.ml_connector
        import api_ml.ml_layer_torch

        api_ml.ml_layer_torch.sentiment_est.model_.to(device)
        api_ml.ml_layer_torch.selector_est.model_.cpu()
        api_ml.ml_layer_torch.autoreviewer_est.model_.cpu()
        sentiment_est = api_ml.ml_layer_torch.sentiment_est
    else:
        import api_ml.ml_connector

    def monkeypatched_sentiment_do(method, *args, repeat_until_done_signal=False, **kwargs):
        out = getattr(sentiment_est, method)(*args, **kwargs)
        return [{"result": out}]

    api_ml.ml_connector.sentiment_est.do = monkeypatched_sentiment_do

    return run_sentiment_on_docs(docs, save_path=save_path, batch_size=batch_size, recompute_existing=recompute_existing,
                                 ts=ts)


def run_autoreviewer_on_docs_local(
    docs, save_path="autoreviewer.json", batch_size=8, recompute_existing=False, device="cuda:0",
    autoreviewer_est=None, ts=now
 ):
     if not autoreviewer_est:
         import api_ml.ml_connector
         import api_ml.ml_layer_torch

         api_ml.ml_layer_torch.autoreviewer_est.model_.to(device)
         api_ml.ml_layer_torch.selector_est.model_.cpu()
         api_ml.ml_layer_torch.sentiment_est.model_.cpu()
         autoreviewer_est = api_ml.ml_layer_torch.autoreviewer_est
     else:
         import api_ml.ml_connector

     def monkeypatched_sentiment_do(method, *args, repeat_until_done_signal=False, **kwargs):
         out = getattr(autoreviewer_est, method)(*args, **kwargs)
         return [{"result": out}]

     api_ml.ml_connector.sentiment_est.do = monkeypatched_sentiment_do

     return run_autoreviewer_on_docs(docs, save_path=save_path, batch_size=batch_size,
                                     recompute_existing=recompute_existing,
                                     ts=ts)
