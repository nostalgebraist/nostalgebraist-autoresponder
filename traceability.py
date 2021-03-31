import os
import pickle
import time
from copy import deepcopy
from datetime import datetime

import pandas as pd

TRACEABILITY_FN = os.path.join(os.path.dirname(__file__), "data/traceability_logs.pkl.gz")


def _add_field(logs, fieldname):
    new_logs = {"fields": logs["fields"] + [fieldname], "data": []}
    for entry in logs["data"]:
        new_entry = deepcopy(entry)
        if fieldname not in new_entry:
            new_entry[fieldname] = None
        new_logs["data"].append(new_entry)
    return new_logs


def on_post_creation_callback(api_response: dict, bridge_response: dict):
    t1 = time.time()

    if not os.path.exists(TRACEABILITY_FN):
        logs = {"fields": [], "data": []}
    else:
        with open(TRACEABILITY_FN, "rb") as f:  # TODO: make this less slow
            logs = pickle.load(f)
    _tload = time.time()
    print(f"on_post_creation_callback LOAD: {_tload-t1:.3f}s sec")

    entry = {"api__" + k: v for k, v in api_response.items()}
    entry.update(bridge_response)

    entry['timestamp_manual'] = datetime.now().timestamp()

    for k in sorted(entry.keys()):
        if k not in logs["fields"]:
            print(f"on_post_creation_callback: adding field named {repr(k)}")
            logs = _add_field(logs, k)

    logs["data"].append(entry)
    _tadd = time.time()
    print(f"on_post_creation_callback ADD: {_tadd-_tload:.3f}s sec")

    with open(TRACEABILITY_FN, "wb") as f:
        pickle.dump(logs, f)

    _tsave = time.time()
    print(f"on_post_creation_callback SAVE: {_tsave-_tadd:.3f}s sec")

    with open(TRACEABILITY_FN[: -len(".pkl.gz")] + "_backup.pkl.gz", "wb") as f:
        pickle.dump(logs, f)

    _tsave2 = time.time()
    print(f"on_post_creation_callback SAVE 2: {_tsave2-_tsave:.3f}s sec")

    t2 = time.time()
    print(f"on_post_creation_callback: took {t2-t1:.3f}s sec")


def traceability_logs_to_df(logs,
                            boring_fields=None,
                            make_input_blogname=True,
                            drop_malformed={"miro_traces",}
                            ):
    data = logs["data"]

    # drop_malformed
    if "miro_traces" in drop_malformed:
        for entry in data:
            mt = entry.get('miro_traces')
            if mt and len(mt['mu'][0]) != len(mt['k'][0]):
                entry['miro_traces'] = None

    df = pd.DataFrame(data)
    if boring_fields is None:
        boring_fields = ["base_id", "AB_fork"]
        boring_fields += [c for c in df.columns if c.startswith("all_alt_")]
    df = df.drop(boring_fields, axis=1)
    if make_input_blogname:
        df["input_blogname"] = df.input_ident.apply(
            lambda tup: None
            if not isinstance(tup, tuple)
            else [entry for entry in tup if isinstance(entry, str)][0]
        )
    return df


def load_traceability_logs_to_df(**kwargs):
    with open(TRACEABILITY_FN, "rb") as f:
        logs = pickle.load(f)
    return traceability_logs_to_df(logs, **kwargs)
