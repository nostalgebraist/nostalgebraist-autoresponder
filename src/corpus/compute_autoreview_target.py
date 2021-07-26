import json
import argparse
from collections import defaultdict, Counter
from datetime import datetime

from tqdm.auto import tqdm

from persistence import traceability
from corpus.blog_archive import roll_head_timestamp


def sub_prompt_timestamp(base_head_timestamp, actual_timestamp, prompt_autoreviewer):
    before, sep, seg = prompt_autoreviewer.rpartition("\n\n Written ")
    timeseg, sep2, after = seg.partition(" | ")

    head_ts = roll_head_timestamp(
        base_head_timestamp=base_head_timestamp, actual_timestamp=actual_timestamp
    )

    return before + sep + head_ts.strftime("%-I %p %B %Y") + sep2 + after


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dryrun", action="store_true")
    parser.add_argument("--hot-only", action="store_true")
    args = parser.parse_args()

    base_head_timestamp = datetime.now()

    # trace
    print("loading trace logs")
    if args.hot_only:
        import persistence.traceability_singleton

        trace_logs = persistence.traceability_singleton.TRACE_LOGS.logs["data"]
    else:
        trace_logs = traceability.load_full_traceability_logs()["data"]

    print(f"loaded trace logs: {len(trace_logs)} rows")

    trace_logs = [row for row in trace_logs if row.get("requested__state") == "draft"]

    print(f"subsetted trace logs to draft:  {len(trace_logs)} rows")

    required_keys = ["api__id", "prompt_autoreviewer", "choice_ix", "all_continuations", "timestamp_manual"]
    keycounts = Counter()
    key_nonnull_counts = Counter()

    for row in trace_logs:
        for k in required_keys:
            keycounts[k] += (k in row)
            key_nonnull_counts[k] += (row.get(k) is not None)

    print(f"keycounts: {keycounts}\nkey_nonnull_counts: {key_nonnull_counts}")

    trace_logs = [
        row
        for row in trace_logs
        if all(
            row.get(k) is not None
            for k in required_keys
        )
    ]

    print(f"subsetted trace logs to nwo / usable:  {len(trace_logs)} rows")

    trace_indices_to_texts = {}
    for i, row in enumerate(trace_logs):
        actual_timestamp = datetime.fromtimestamp(row["timestamp_manual"])

        subbed = sub_prompt_timestamp(base_head_timestamp, actual_timestamp, row["prompt_autoreviewer"])
        trace_indices_to_texts[i] = subbed + row["all_continuations"][row["choice_ix"]]

    trace_map = defaultdict(list)

    for i, row in enumerate(trace_logs):
        trace_map[row["api__id"]].append(i)

    # pub
    print("loading pub logs")
    with open("data/head_training_data.json", "r", encoding="utf-8") as f:
        pub_logs = json.load(f)

    print(f"loaded pub logs: {len(pub_logs)} rows")

    for row in pub_logs:
        gid = row["genesis_post_id"]
        row["genesis_or_published_id"] = gid if gid is not None else row["id"]

    pub_map = defaultdict(list)

    for i, row in enumerate(pub_logs):
        pub_map[row["genesis_or_published_id"]].append(i)

    # match
    print("matching...")

    trace_indices_to_targets = {}
    trace_indices_to_published_ids = {}

    n_accept = 0
    n_reject = 0
    n_multimatch = 0

    iter_ = tqdm(trace_map.items(), total=len(trace_map), mininterval=1, smoothing=0)

    for api__id, group_trace_indices in iter_:
        pub_gids_matching_trace_id = pub_map.get(api__id, [])

        if len(pub_gids_matching_trace_id) == 0:
            # never published
            for trace_index in group_trace_indices:
                trace_indices_to_targets[trace_index] = "reject"
                trace_indices_to_published_ids[trace_index] = None
            n_reject += len(group_trace_indices)
        else:
            if len(pub_gids_matching_trace_id) > 1:
                # ???
                n_multimatch += 1

            matching_pub_row = pub_logs[pub_gids_matching_trace_id[0]]

            # assumes trace is ordered by time -- i believe this is true
            pubd_ix = group_trace_indices[-1]
            trace_indices_to_targets[pubd_ix] = "accept"
            trace_indices_to_published_ids[pubd_ix] = matching_pub_row["id"]

            for trace_index in group_trace_indices[:-1]:
                trace_indices_to_targets[trace_index] = "reject"
                trace_indices_to_published_ids[trace_index] = None

            n_accept += 1
            n_reject += len(group_trace_indices) - 1

        iter_.set_postfix(
            n_accept=n_accept, n_reject=n_reject, zz_n_multimatch=n_multimatch
        )

    # verify
    n_accept_verify = sum(v == "accept" for v in trace_indices_to_targets.values())
    n_reject_verify = sum(v == "reject" for v in trace_indices_to_targets.values())

    print(f"\nn_accept: {n_accept_verify} vs {n_accept}")
    print(f"n_reject: {n_reject_verify} vs {n_reject}")

    autoreview_train_data = []
    for ix in sorted(trace_indices_to_targets.keys()):
        trace_indices_to_texts[ix]
        autoreview_train_data.append(
            {
                "text": trace_indices_to_texts[ix],
                "target": trace_indices_to_targets[ix],
                "trace_api__id": trace_logs[ix]["api__id"],
                "pub_api__id": trace_indices_to_published_ids[ix],
            }
        )

    if not args.dryrun:
        with open("data/autoreview_train_data.json", "w", encoding="utf-8") as f:
            json.dump(autoreview_train_data, f, indent=1)


if __name__ == "__main__":
    main()
