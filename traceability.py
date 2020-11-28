import os
import pickle
from copy import deepcopy

import pandas as pd

TRACEABILITY_FN = "traceability_logs.pkl.gz"

def _add_field(logs, fieldname):
    new_logs = {"fields": logs['fields'] + [fieldname], "data": []}
    for entry in logs['data']:
        new_entry = deepcopy(entry)
        if fieldname not in new_entry:
            new_entry[fieldname] = None
        new_logs["data"].append(new_entry)
    return new_logs

def on_post_creation_callback(api_response: dict, bridge_response: dict):
    if not os.path.exists(TRACEABILITY_FN):
        logs = {"fields": [], "data": []}
        fields = []
    else:
        with open(TRACEABILITY_FN, "rb") as f:
            logs = pickle.load(f)

    entry = {"api__"+k: v for k, v in api_response.items()}
    entry.update(bridge_response)

    for k in sorted(entry.keys()):
        if k not in logs['fields']:
            print(f"on_post_creation_callback: adding field named {repr(k)}")
            logs = _add_field(logs, k)

    logs['data'].append(entry)

    with open(TRACEABILITY_FN, "wb") as f:
        pickle.dump(logs, f)

    with open(TRACEABILITY_FN[:-len(".pkl.gz")] + "_backup.pkl.gz", "wb") as f:
        pickle.dump(logs, f)

def traceability_logs_to_df(logs, boring_fields=None, make_input_blogname=True):
    if boring_fields is None:
        boring_fields = ["api__id_string", "base_id", "AB_fork"]
    df = pd.DataFrame(logs['data'])
    df = df.drop(boring_fields, axis=1)
    if make_input_blogname:
        df['input_blogname'] = df.input_ident.apply(lambda tup: None if not isinstance(tup, tuple) else [entry for entry in tup if isinstance(entry, str)][0])
    return df

def load_traceability_logs_to_df(**kwargs):
    with open(TRACEABILITY_FN, "rb") as f:
        logs = pickle.load(f)
    return traceability_logs_to_df(logs, **kwargs)
