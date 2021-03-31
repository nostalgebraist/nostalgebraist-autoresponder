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


class TraceabilityLogs:
    def __init__(self, logs: dict, path: str):
        self.logs = logs
        self.path = path
        print(f"traceability logs init: lengths {self.lengths}")

    @property
    def lengths(self):
        return {k: len(v) for k, v in self.logs.items()}

    @staticmethod
    def load(path=TRACEABILITY_FN) -> 'TraceabilityLogs':
        if not os.path.exists(TRACEABILITY_FN):
            print('initializing fresh traceability logs')
            logs = {"fields": [], "data": []}
        else:
            print('loading traceability logs')
            with open(TRACEABILITY_FN, "rb") as f:
                logs = pickle.load(f)
            return TraceabilityLogs(logs=logs, path=path)

    def save(self):
        print(f'saving traceability logs: lengths {self.lengths}')
        with open(self.path, "wb") as f:
            pickle.dump(self.logs, f)

        with open(self.path[: -len(".pkl.gz")] + "_backup.pkl.gz", "wb") as f:
            pickle.dump(self.logs, f)

    def on_post_creation_callback(self, api_response: dict, bridge_response: dict):
        t1 = time.time()

        _tload = time.time()
        print(f"on_post_creation_callback LOAD: {_tload-t1:.3f}s sec")

        entry = {"api__" + k: v for k, v in api_response.items()}
        entry.update(bridge_response)

        entry['timestamp_manual'] = datetime.now().timestamp()

        for k in sorted(entry.keys()):
            if k not in self.logs["fields"]:
                print(f"on_post_creation_callback: adding field named {repr(k)}")
                self.logs = _add_field(self.logs, k)

        self.logs["data"].append(entry)
        _tadd = time.time()
        print(f"on_post_creation_callback ADD: {_tadd-_tload:.3f}s sec")

        self.save()

        t2 = time.time()
        print(f"on_post_creation_callback: took {t2-t1:.3f}s sec")


def traceability_logs_to_df(logs,
                            boring_fields=None,
                            make_input_blogname=True,
                            drop_malformed={"miro_traces", }
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
