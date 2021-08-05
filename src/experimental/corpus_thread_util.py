import hashlib
import random
import re
from collections import defaultdict, Counter
from functools import partial

from tqdm.auto import tqdm as tqdm_base
tqdm = partial(tqdm_base, mininterval=1, smoothing=0)

from tumblr_to_text.classic.autoresponder_static import find_control_chars_forumlike, EOT

quote_hell_regex = re.compile(r"<blockquote><a href=\"http://[^\.]+\.tumblr\.com/post/[^\"]+\">[^<]+</a>:")


def remove_ignored_substrings(s):
    s = re.sub(r"<h2>.*</h2>", "", s)
    s = s.replace("\n", "").replace(" ", "")
    s = re.sub(r"\<.*?\>", "", s)
    return s


def post_after_cc_is_empty(doc, ccs, i, verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    try:
        start_ix = ccs[i][1]
        start_ix += len(ccs[i][0])
        end_ix = ccs[i + 1][1]
        interior = doc[start_ix:end_ix]
        vprint(f"\n\tinterior before replace:\n\t{repr(interior)}")
        interior = remove_ignored_substrings(interior)
        vprint(f"\n\tchecking interior:\n\t{repr(interior)}")
        return len(interior) == 0
    except IndexError:
        return False  # end of list


def diagnose_malformed(doc, verbose=False):
    reasons = set()

    if "<h2>" in doc:
        # check first-post-is-only-title pattern
        ccs = _get_ccs_with_fixes(doc)
        if len(ccs) > 0:
            start_ix = ccs[0][1]
            end_ix = ccs[1][1] if len(ccs) > 1 else len(doc)
            if "<h2>" in doc[start_ix:end_ix]:
                if post_after_cc_is_empty(doc, ccs, 0):
                    reasons.add("first-post-is-only-title pattern")

    # check quote-hell pattern
    for _ in quote_hell_regex.finditer(doc):
        reasons.add("quote-hell pattern")

    return reasons


def _get_ccs_with_fixes(doc):
    extra_names = doc.split(" ")[1:2]
    if extra_names[0] == 'nostalgebraist-autoresponder':
        extra_names = []

    ccs = find_control_chars_forumlike(doc, extra_names=extra_names)

    # edge case
    if ccs[0][0].startswith("#1 nostalgebraist-autoresponder posted"):
        if ccs[1][0].startswith(" nostalgebraist-autoresponder posted"):
            ccs.pop(1)
    return ccs


def extract_prefix(doc, include_username=False, verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint(f"include_username={include_username}")

    ccs = _get_ccs_with_fixes(doc)

    vprint("base ccs:")
    vprint(ccs)

    pre_content = ""
    if len(ccs) > 1 and "asked:" in ccs[0][0]:
        left = ccs[0][1] if include_username else ccs[0][1] + len(ccs[0][0].rstrip("\n"))
        pre_content = doc[left:ccs[1][1]]
        ccs.pop(0)

    while post_after_cc_is_empty(doc, ccs, 0, verbose=verbose):
        left = ccs[0][1] if include_username else ccs[0][1] + len(ccs[0][0].rstrip("\n"))
        pre_content += doc[left:ccs[1][1]]
        ccs.pop(0)

    vprint("\nccs after pop ask/empty:")
    vprint(ccs)

    if len(ccs) > 1:
        left = ccs[0][1] if include_username else ccs[0][1] + len(ccs[0][0].rstrip("\n"))
        prefix = doc[left:ccs[1][1]]
    else:
        left = ccs[0][1] if include_username else ccs[0][1] + len(ccs[0][0].rstrip("\n"))
        prefix = doc[left:]
        lines = prefix.splitlines()
        vprint("\nlines:")
        vprint(lines)
        lines.pop(2)
        prefix = "\n".join(lines)

    vprint("\nbase prefix:")
    vprint(repr(prefix))

    if prefix.startswith("#"):
        prefix = prefix.partition(" ")[2]

    prefix = pre_content + prefix
    prefix = remove_ignored_substrings(prefix)

    return prefix


def map_docs(docs, include_usernames=True):
    trails = defaultdict(set)

    for i, doc in tqdm(enumerate(docs), total=len(docs)):
        prefix = extract_prefix(doc, include_username=include_usernames)
        prefix_hash = hashlib.md5(prefix.encode("utf-8")).hexdigest()
        trails[prefix_hash].add(i)

    return trails


def map_docs_multiple_groups(*doc_groups, include_usernames=True):
    docs = []
    doc_index_to_group_index = {}

    doc_index_offset = 0
    for group_index, g in enumerate(doc_groups):
        # indexing g[i] enforces orderedness requirement on inputs
        docs.extend([g[i] for i in range(len(g))])

        doc_index_to_group_index.update({i + doc_index_offset: group_index for i in range(len(g))})

        doc_index_offset += len(g)

    trails = map_docs(docs, include_usernames=include_usernames)
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
                           decontaminate_uniform_over_docs=False,  # tends to make val smaller, train bigger
                           choose_longest=False,
                           prefer_longest=False,
                           prefer_longest_temp=1,
                           include_longest_nost_if_applicable=False,
                           ):
    from scipy.special import softmax

    random.seed(random_seed)

    nost_s = "nostalgebraist-autoresponder"

    def _choose_one(trail):
        trail_l = sorted(trail)

        lens = [len(docs[doc_ix]) for doc_ix in trail_l]
        length_sorted_ixs = sorted(range(len(trail_l)), key=lambda i: lens[i])
        longest_ix = length_sorted_ixs[-1]  # argmax

        if include_longest_nost_if_applicable and any(nost_s in docs[doc_ix] for doc_ix in trail_l):
            start_length_sorted_ixs_at = [
                i for i, ix in enumerate(length_sorted_ixs)
                if nost_s in docs[trail_l[ix]]
            ][0]

            kept = length_sorted_ixs[start_length_sorted_ixs_at:]

            trail_l = [trail_l[ix] for ix in kept]

            # TODO: DRY
            lens = [len(docs[doc_ix]) for doc_ix in trail_l]
            length_sorted_ixs = sorted(range(len(trail_l)), key=lambda i: lens[i])
            longest_ix = length_sorted_ixs[-1]  # argmax

        if choose_longest:
            chosen_ix = longest_ix
        elif prefer_longest:
            w = softmax([(e / max(lens)) / prefer_longest_temp
                         for e in lens])
            chosen_ix = random.choices(range(len(trail_l)), weights=w)[0]
        else:
            chosen_ix = random.randrange(len(trail_l))
        was_longest = longest_ix == chosen_ix
        return trail_l[chosen_ix], was_longest

    allowed_doc_indices = set()

    was_longest_all = []

    if decontaminate_only:
        crossing_trails, group_indices_of_trails = find_trails_crossing_groups(trails, doc_index_to_group_index)

        for k, v in tqdm(list(trails.items())):
            if k in crossing_trails:
                if decontaminate_uniform_over_docs:
                    group_indices = sorted(set(group_indices_of_trails[k]))
                    selected_group_ix = random.choice(group_indices)
                else:
                    selected_doc_ix, was_longest = _choose_one(v)
                    if len(v) > 1:
                        was_longest_all.append(was_longest)
                    selected_group_ix = doc_index_to_group_index[selected_doc_ix]
                allowed_doc_indices.update({doc_ix for doc_ix in v
                                            if doc_index_to_group_index[doc_ix] == selected_group_ix
                                            })
            else:
                allowed_doc_indices.update(v)
    else:
        for v in tqdm(list(trails.values())):
            selected_doc_ix, was_longest = _choose_one(v)
            if len(v) > 1:
                was_longest_all.append(was_longest)
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

    if len(was_longest_all) == 0:
        was_longest_all = [0]
    return deduped_groups, sum(was_longest_all) / len(was_longest_all)


def nontrivial_trails(trails):
    return {k: v for k, v in trails.items() if len(v) > 1}


def show_trail(docs, trails, key):
    tdocs = sorted([docs[i] for i in trails[key]], key=len)
    for td in tdocs:
        print(repr(td))
        print()
        print(repr(extract_prefix(td)))
        print('\n------\n')


def load_trails_from_docs(paths, include_usernames=False, exclude_malformed=True):
    if isinstance(paths, str):
        paths = [paths]

    doc_groups = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            ds = f.read()
        g = [d for d in ds.split(EOT) if len(d) > 0]
        n_raw = len(g)
        print(f"read group from file {p}: {n_raw} raw docs")

        if exclude_malformed:
            reasons = [diagnose_malformed(d) for d in g]
            symptom_counts = Counter([r for rs in reasons for r in rs])
            g = [d for d, r in zip(g, reasons) if r == set()]
            n_excluded = n_raw - len(g)
            print(f"read group from file {p}: removed {n_excluded} malformed docs")
            print(f"reasons: {repr(symptom_counts)}")

        print(f"read group from file {p}: {len(g)} docs")
        doc_groups.append(g)
    print()

    docs, trails, doc_index_to_group_index = map_docs_multiple_groups(*doc_groups, include_usernames=include_usernames)

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
