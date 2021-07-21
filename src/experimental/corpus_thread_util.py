import hashlib
import random
from collections import defaultdict
from functools import partial

from tqdm.auto import tqdm as tqdm_base
tqdm = partial(tqdm_base, mininterval=1, smoothing=0)

from tumblr_to_text.classic.autoresponder_static import find_control_chars_forumlike, EOT


def remove_ignored_substrings(s):
    return s.replace("\n", "").replace(" ", "").replace("</h2>", "")


def post_after_cc_is_empty(doc, ccs, i):
    try:
        start_ix = ccs[i][1]
        start_ix += len(ccs[i][0])
        end_ix = ccs[i + 1][1]
        interior = doc[start_ix:end_ix]
        interior = remove_ignored_substrings(interior)
        return len(interior) == 0
    except IndexError:
        return False  # end of list


def extract_prefix(doc, verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    extra_names = doc.split(" ")[1:2]
    if extra_names[0] == 'nostalgebraist-autoresponder':
        extra_names = []

    ccs = find_control_chars_forumlike(doc, extra_names=extra_names)

    # edge case
    if ccs[0][0].startswith("#1 nostalgebraist-autoresponder posted"):
        if ccs[1][0].startswith(" nostalgebraist-autoresponder posted"):
            ccs.pop(1)

    vprint(ccs)

    pre_content = ""
    if len(ccs) > 1 and "asked:" in ccs[0][0]:
        pre_content = doc[ccs[0][1]:ccs[1][1]]
        ccs.pop(0)

    while post_after_cc_is_empty(doc, ccs, 0):
        pre_content += doc[ccs[0][1]:ccs[1][1]]
        ccs.pop(0)

    vprint(ccs)

    if len(ccs) > 1:
        prefix = doc[ccs[0][1]:ccs[1][1]]
    else:
        prefix = doc[ccs[0][1]:]
        lines = prefix.splitlines()
        lines.pop(2)
        prefix = "\n".join(lines)
    vprint(repr(prefix))

    if prefix.startswith("#"):
        prefix = prefix.partition(" ")[2]

    prefix = pre_content + prefix
    prefix = remove_ignored_substrings(prefix)
    return prefix


def map_docs(docs):
    trails = defaultdict(set)

    for i, doc in tqdm(enumerate(docs), total=len(docs)):
        prefix = extract_prefix(doc)
        prefix_hash = hashlib.md5(prefix.encode("utf-8")).hexdigest()
        trails[prefix_hash].add(i)

    return trails


def map_docs_multiple_groups(*doc_groups):
    docs = []
    doc_index_to_group_index = {}

    doc_index_offset = 0
    for group_index, g in enumerate(doc_groups):
        # indexing g[i] enforces orderedness requirement on inputs
        docs.extend([g[i] for i in range(len(g))])

        doc_index_to_group_index.update({i + doc_index_offset: group_index for i in range(len(g))})

        doc_index_offset += len(g)

    trails = map_docs(docs)
    return docs, trails, doc_index_to_group_index


def find_trails_crossing_groups(trails, doc_index_to_group_index):
    group_indices_of_trails = {
        k: {doc_index_to_group_index[ix] for ix in v} for k, v in trails.items()
    }
    crossing_trails = {k: trails[k] for k, v in group_indices_of_trails.items()
                       if len(v) > 1}
    return crossing_trails, group_indices_of_trails


def dedup_groups_trailwise(docs, trails, doc_index_to_group_index, random_seed=10,
                           decontaminate_only=False,
                           choose_longest=False,
                           duplication_frac=0  # TODO: implement
                           ):
    random.seed(random_seed)

    def _choose_one(docs, trail):
        if choose_longest:
            lens = [len(docs[doc_ix]) for doc_ix in trail]
            chosen_ix = sorted(range(lens), key=lambda i: lens[i])[-1]  # argmax
        else:
            chosen_ix = random.randrange(len(trail))
        return trail[chosen_ix]

    allowed_doc_indices = set()

    if decontaminate_only:
        crossing_trails, group_indices_of_trails = find_trails_crossing_groups(trails, doc_index_to_group_index)

        for k, v in tqdm(list(trails.items())):
            if k in crossing_trails:
                # group_indices = sorted(set(group_indices_of_trails[k]))
                # selected_group_ix = random.choice(group_indices)
                selected_doc_ix = _choose_one(v)
                selected_group_ix = doc_index_to_group_index[selected_doc_ix]
                allowed_doc_indices.update({doc_ix for doc_ix in v
                                            if doc_index_to_group_index[doc_ix] == selected_group_ix
                                            })
            else:
                allowed_doc_indices.update(v)
    else:
        for v in tqdm(list(trails.values())):
            # selected_doc_ix = random.choice(sorted(v))
            selected_doc_ix = _choose_one(v)
            allowed_doc_indices.add(selected_doc_ix)

    ngroup = len(set(doc_index_to_group_index.values()))

    deduped_groups = [[] for _ in range(ngroup)]

    for doc_index, doc in enumerate(tqdm(docs)):
        if doc_index in allowed_doc_indices:
            group_index = doc_index_to_group_index[doc_index]
            try:
                deduped_groups[group_index].append(doc)
            except IndexError:
                print((doc_index, group_index, len(deduped_groups), ngroup))
                raise IndexError

    return deduped_groups


def nontrivial_trails(trails):
    return {k: v for k, v in trails.items() if len(v) > 1}


def show_trail(docs, trails, key):
    tdocs = sorted([docs[i] for i in trails[key]], key=len)
    for td in tdocs:
        print(repr(td))
        print()
        print(repr(extract_prefix(td)))
        print('\n------\n')


def load_trails_from_docs(paths):
    if isinstance(paths, str):
        paths = [paths]

    doc_groups = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            ds = f.read()
        g = [d for d in ds.split(EOT) if len(d) > 0]
        doc_groups.append(g)
        print(f"read group from file: {len(g)} docs")
    print()

    docs, trails, doc_index_to_group_index = map_docs_multiple_groups(*doc_groups)

    nt = nontrivial_trails(trails)

    trail_stats(docs, trails, nt)

    return docs, trails, nt, doc_index_to_group_index


def trail_stats(docs, trails, nt):
    print(f"{len(docs)} docs")
    print(f"{len(trails)} trails")
    print(f"{len(nt)} trails with length > 1")
    print()

    ndoc = len(docs)
    ndoc_in_nt = sum(len(v) for v in nt.values())

    lendoc = sum(len(d) for d in docs)
    lendoc_in_nt = sum(len(docs[i]) for v in nt.values() for i in v)

    print(f"{ndoc_in_nt/ndoc:.2%} of docs are in trails with length > 1\n\t({ndoc_in_nt} / {ndoc})\n")
    print(f"{lendoc_in_nt/lendoc:.2%} of characters are in trails with length > 1\n\t({lendoc_in_nt} / {lendoc})\n")
