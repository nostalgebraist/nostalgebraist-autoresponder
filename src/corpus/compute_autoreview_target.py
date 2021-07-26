import json
import argparse
from collections import defaultdict

from persistence import traceability


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    # trace
    print("loading trace logs")
    trace_logs = traceability.load_full_traceability_logs()["logs"]

    print(f"loaded trace logs: {len(trace_logs)} rows")

    trace_logs = [row for row in trace_logs
                  if row.get("requested__state") == "draft"]

    print(f"subsetted trace logs to draft:  {len(trace_logs)} rows")

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
    print('matching...')

    trace_indices_to_targets = {}

    for api__id, group_trace_indices in trace_map.items():
        pass  # TODO


if __name__ == "__main__":
    main()
