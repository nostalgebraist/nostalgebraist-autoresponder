import os
import json
from functools import partial

from tqdm.autonotebook import tqdm

from experimental.decision_encoding import prep_for_selector, prep_for_sentiment, unique_id_for_doc

def run_model_on_docs(docs,
                      prep_fn,
                      predict_fn,
                      save_path,
                      save_dir="data/decision_encoding",
                      batch_size=8,
                      verbose=False):
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, save_path)

    results = {}
    if os.path.exists(save_path):
        results = json.load(open(save_path))

    uids = {d: unique_id_for_doc(d) for d in docs}
    docs = [d for d in docs if uids[d] not in results]

    batches = [docs[i:i+batch_size] for i in range(0, len(docs), batch_size)]
    if len(batches) * batch_size < len(docs):
        batches.append(docs[i:])

    for batch in tqdm(batches, mininterval=1, smoothing=0):
        batch_prepped = [prep_fn(d) for d in batch]
        probs = predict_fn(batch_prepped)
        for d, p in zip(batch, probs):
            results[uids[d]] = p

    with open(save_path, "w") as f:
        json.dump(results, f)

    return results


def run_selector_on_docs(docs, save_path="selector.json"):
    import experimental.ml_connector

    return run_model_on_docs(docs,
                             prep_fn=prep_for_selector,
                             predict_fn=partial(experimental.ml_connector.selection_proba_from_gpt2_service, already_forumlike=True),
                             save_path=save_path)


def run_sentiment_on_docs(docs, save_path="sentiment.json"):
    import experimental.ml_connector

    return run_model_on_docs(docs,
                             prep_fn=prep_for_sentiment,
                             predict_fn=experimental.ml_connector.sentiment_logit_diffs_from_gpt2_service,
                             save_path=save_path)


def run_selector_on_docs_local(docs, save_path="selector.json", device='cuda:0'):
    import experimental.ml_layer_torch
    import experimental.ml_connector

    experimental.ml_layer_torch.selector_est.to(device)

    experimental.ml_connector.selector_est = experimental.ml_layer_torch.selector_est
    return run_selector_on_docs(docs, save_path=save_path)


def run_sentiment_on_docs_local(docs, save_path="selector.json", device='cuda:0'):
    import experimental.ml_layer_torch
    import experimental.ml_connector

    experimental.ml_layer_torch.sentiment_est.to(device)

    experimental.ml_connector.sentiment_est = experimental.ml_layer_torch.sentiment_est
    return run_sentiment_on_docs(docs, save_path=save_path)
