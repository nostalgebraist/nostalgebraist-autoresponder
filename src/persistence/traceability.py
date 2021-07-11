import os
import pickle
import time
import subprocess
from copy import deepcopy
from datetime import datetime

import pandas as pd

TRACEABILITY_FN = "data/traceability_logs.pkl.gz"
TRACEABILITY_COLD_STORAGE_FN = "data/traceability_logs_cold_storage.pkl.gz"


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
        if not os.path.exists(path):
            print('initializing fresh traceability logs')
            logs = {"fields": [], "data": []}
        else:
            print('loading traceability logs')
            with open(path, "rb") as f:
                logs = pickle.load(f)
        return TraceabilityLogs(logs=logs, path=path)

    def save(self):
        print(f'saving traceability logs: lengths {self.lengths}')
        t1 = time.time()

        with open(self.path, "wb") as f:
            pickle.dump(self.logs, f)

        _tsave = time.time()
        print(f"trace save 1: {_tsave-t1:.3f}s sec")

        backup_path = self.path[: -len(".pkl.gz")] + "_backup.pkl.gz"
        subprocess.check_output(f"cp {self.path} {backup_path}", shell=True)

        _tsave2 = time.time()
        print(f"trace save 2: {_tsave2-_tsave:.3f}s sec")

    def on_post_creation_callback(self, api_response: dict, bridge_response: dict):
        t1 = time.time()

        entry = {"api__" + k: v for k, v in api_response.items()}
        entry.update(bridge_response)

        entry['timestamp_manual'] = datetime.now().timestamp()

        for k in sorted(entry.keys()):
            if k not in self.logs["fields"]:
                print(f"on_post_creation_callback: adding field named {repr(k)}")
                self.logs = _add_field(self.logs, k)

        self.logs["data"].append(entry)

        self.save()

        t2 = time.time()
        print(f"on_post_creation_callback: took {t2-t1:.3f}s sec")


def traceability_logs_to_df(logs,
                            boring_fields=None,
                            make_input_blogname=True,
                            drop_malformed={"miro_traces", },
                            unpack={"state_reasons"},
                            ):
    data = logs["data"]

    # drop_malformed
    if "miro_traces" in drop_malformed:
        for entry in data:
            mt = entry.get('miro_traces')
            if mt and len(mt['mu']) > 0 and len(mt['mu'][0]) != len(mt['k'][0]):
                entry['miro_traces'] = None

    df = pd.DataFrame(data)
    if boring_fields is None:
        boring_fields = ["base_id", "AB_fork"]
        boring_fields += [c for c in df.columns if c.startswith("all_alt_")]
    boring_fields = set(boring_fields).intersection(df.columns)
    df = df.drop(boring_fields, axis=1)
    if make_input_blogname:
        df["input_blogname"] = df.input_ident.apply(
            lambda tup: None
            if not isinstance(tup, tuple)
            else [entry for entry in tup if isinstance(entry, str)][0]
        )
    for col in unpack:
        filt = df[col].notnull()
        to_load = df[filt][col]
        loaded = pd.DataFrame.from_records(to_load.values, index=to_load.index)
        df = df.join(loaded)
    return df


def load_full_traceability_logs():
    import persistence.traceability_singleton
    trace_logs_hot = persistence.traceability_singleton.TRACE_LOGS.logs

    trace_logs_cold = TraceabilityLogs.load(path=TRACEABILITY_COLD_STORAGE_FN).logs

    full_trace_logs = {"fields": trace_logs_cold["fields"], "data": trace_logs_cold["data"]}

    for field in trace_logs_hot["fields"]:
        if field not in full_trace_logs["fields"]:
            print(f"adding hot field to cold: {repr(field)}")
            _add_field(full_trace_logs, field)

    full_trace_logs["data"] = full_trace_logs["data"] + trace_logs_hot["data"]

    return full_trace_logs


def load_traceability_logs_to_df(**kwargs):
    full = kwargs.pop("full", True)
    if full:
        logs = load_full_traceability_logs()
    else:
        import persistence.traceability_singleton
        logs = persistence.traceability_singleton.TRACE_LOGS.logs

    return traceability_logs_to_df(logs, **kwargs)
