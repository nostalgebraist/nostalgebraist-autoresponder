import time
import json

from persistence.base import SelfArchivingJsonlStore
from persistence.response_cache import PostIdentifier

TRACEABILITY_NAME = "traceability_logs"
TRACEABILITY_DIR = "data"
TRACEABILITY_MAX_ENTRIES_HOT = 5000
TRACEABILITY_MIN_BATCH_SIZE = 2000


def fill_pis(d):
    for k in list(d.keys()):
        _, prefix, newk = k.partition('_pi_')
        if prefix:
            d[newk] = PostIdentifier(*d.pop(k))
    return d

def custom_loads(s):
    return fill_pis(json.loads(s))

def custom_dumps(d):
    keymap = {}
    for k in list(d.keys()):
        if isinstance(d[k], PostIdentifier):
            keymap[k] = '_pi_' + k
    return json.dumps({keymap.get(k, k): d[k] for k in d})


class TraceabilityLogsJsonl(SelfArchivingJsonlStore):
    def __init__(self,
                 name=TRACEABILITY_NAME,
                 directory=TRACEABILITY_DIR,
                 max_entries_hot=TRACEABILITY_MAX_ENTRIES_HOT,
                 archival_min_batch_size=TRACEABILITY_MIN_BATCH_SIZE,):
        super().__init__(
            name=name,
            directory=directory,
            max_entries_hot=max_entries_hot,
            archival_min_batch_size=archival_min_batch_size,
            loads=custom_loads,
            dumps=custom_dumps,
        )
        print(f"traceability logs init: length {self.n_entries}")

    def save(self):
        pass

    def maybe_archive(self, do_backup=True):
        needs = self.needs_archive()
        if needs:
            print(f"traceability logs: length {self.n_entries}, archiving...")

        t1 = time.time()

        super().maybe_archive(do_backup=do_backup)

        if needs:
            print(f"traceability logs: archived in {time.time()-t1:.1f}s, length now {self.n_entries}")

    def on_post_creation_callback(self, api_response: dict, bridge_response: dict, do_backup=True):
        t1 = time.time()

        entry = {"api__" + k: v for k, v in api_response.items()}
        entry.update(bridge_response)

        entry['timestamp_manual'] = now_pst().timestamp()

        self.write_entry(entry, do_backup=do_backup)

        t2 = time.time()
        print(f"on_post_creation_callback: took {t2-t1:.3f}s sec")
