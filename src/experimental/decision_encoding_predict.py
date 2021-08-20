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
    suppress_tqdm=False,
    **kwargs
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
        probs = predict_fn(batch_prepped, **kwargs)
        for d, p in zip(batch, probs):
            results[uids[d]] = p

        with open(save_path, "w") as f:
            json.dump(results, f)

    return results


def run_selector_on_docs(docs, save_path="selector.json", ts=now, **kwargs):
    import api_ml.ml_connector

    return run_model_on_docs(
        docs,
        prep_fn=partial(prep_for_selector, ts=ts),
        predict_fn=api_ml.ml_connector.selection_proba_from_gpt,
        save_path=save_path,
        **kwargs
    )


def run_sentiment_on_docs(docs, save_path="sentiment.json", **kwargs):
    import api_ml.ml_connector

    return run_model_on_docs(
        docs,
        prep_fn=partial(prep_for_sentiment),
        predict_fn=api_ml.ml_connector.sentiment_logit_diffs_from_gpt,
        save_path=save_path,
        **kwargs,
    )


def run_autoreviewer_on_docs(docs, save_path="autoreviewer.json", ts=now, **kwargs):
    import api_ml.ml_connector

    return run_model_on_docs(
        docs,
        prep_fn=partial(prep_for_autoreviewer, ts=ts),
        predict_fn=api_ml.ml_connector.autoreview_proba_from_gpt,
        save_path=save_path,
        **kwargs
    )


def run_selector_on_docs_local(
    docs, save_path="selector.json", device="cuda:0", head_model=None, ts=now, **kwargs
):
    if not head_model:
        import api_ml.ml_layer_torch
        import api_ml.ml_connector
        api_ml.ml_layer_torch.selector_est.model_.to(device)
        api_ml.ml_layer_torch.sentiment_est.model_.cpu()
        api_ml.ml_layer_torch.autoreviewer_est.model_.cpu()
        head_model = api_ml.ml_layer_torch.selector_est
    else:
        import api_ml.ml_connector

    def monkeypatched_selector_do(method, *args, repeat_until_done_signal=False, **kwargs_):
        out = getattr(head_model, method)(*args, **kwargs_)
        return [{"result": out}]

    api_ml.ml_connector.selector_est.do = monkeypatched_selector_do

    return run_selector_on_docs(docs, save_path=save_path, ts=ts, **kwargs)


def run_sentiment_on_docs_local(
    docs, save_path="sentiment.json", device="cuda:0", head_model=None, **kwargs
):
    if not head_model:
        import api_ml.ml_connector
        import api_ml.ml_layer_torch

        api_ml.ml_layer_torch.sentiment_est.model_.to(device)
        api_ml.ml_layer_torch.selector_est.model_.cpu()
        api_ml.ml_layer_torch.autoreviewer_est.model_.cpu()
        head_model = api_ml.ml_layer_torch.sentiment_est
    else:
        import api_ml.ml_connector

    def monkeypatched_sentiment_do(method, *args, repeat_until_done_signal=False, **kwargs_):
        out = getattr(head_model, method)(*args, **kwargs_)
        return [{"result": out}]

    api_ml.ml_connector.sentiment_est.do = monkeypatched_sentiment_do

    return run_sentiment_on_docs(docs, save_path=save_path, **kwargs)


def run_autoreviewer_on_docs_local(
    docs, save_path="autoreviewer.json", device="cuda:0", head_model=None, ts=now, **kwargs
):
     if not head_model:
         import api_ml.ml_connector
         import api_ml.ml_layer_torch

         api_ml.ml_layer_torch.autoreviewer_est.model_.to(device)
         api_ml.ml_layer_torch.selector_est.model_.cpu()
         api_ml.ml_layer_torch.sentiment_est.model_.cpu()
         head_model = api_ml.ml_layer_torch.autoreviewer_est
     else:
         head_model.model_.to(device)
         import api_ml.ml_connector

     def monkeypatched_autoreview_do(method, *args, repeat_until_done_signal=False, **kwargs_):
         out = getattr(head_model, method)(*args, **kwargs_)
         return [{"result": out}]

     api_ml.ml_connector.autoreviewer_est.do = monkeypatched_autoreview_do

     out = run_autoreviewer_on_docs(docs, save_path=save_path, ts=ts, **kwargs)
     head_model.model_.cpu()
     return out


def _run_head_single_doc_local(doc, run_fn, head_model=None, **kwargs):
    raw = run_fn([doc], head_model=head_model, suppress_tqdm=True, batch_size=1, verbose=False, **kwargs)
    return list(raw.values())[0]


run_selector_local = partial(_run_head_single_doc_local, run_fn=run_selector_on_docs_local)
run_sentiment_local = partial(_run_head_single_doc_local, run_fn=run_sentiment_on_docs_local)
run_autoreviewer_local = partial(_run_head_single_doc_local, run_fn=run_autoreviewer_on_docs_local)
