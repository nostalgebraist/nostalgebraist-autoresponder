import hashlib
import random
import re
from collections import defaultdict, Counter
from functools import partial

from tqdm.auto import tqdm as tqdm_base
tqdm = partial(tqdm_base, mininterval=1, smoothing=0)

from tumblr_to_text.classic.autoresponder_static import EOT
from experimental.corpus_text_hacks import extract_time_from_forumlike_doc, get_ccs_with_fixes
from experimental.corpus_slimmers import slim_forumlike_doc

quote_hell_regex = re.compile(r"<blockquote><a href=\"http://[^\.]+\.tumblr\.com/post/[^\"]+\">[^<]+</a>:")
strip_link_attrs_regex = re.compile(r'<a[^<]*>(.*?)</a>')
strip_link_attrs_repl = r'<a>\g<1></a>'
get_h2_regex = re.compile(r'(<h2>.*?</h2>)')


def unique_id_for_doc(doc: str, prep_fn=None):
    if prep_fn is None:
        prep_fn = lambda x: x
    return hashlib.md5(prep_fn(doc).encode("utf-8")).hexdigest()


def prep_fn_ignore_frank_nost(doc):
    def replaceall(s, r):
        for rr in [r, r.lower(), r.upper(), r.capitalize()]:
            s = s.replace(rr, "")
        return s
    doc = replaceall(doc, "frank")
    doc = doc.replace("nostalgebraist-autoresponder", "")
    doc = doc.replace("nostalgebraist", "")
    return doc


unique_id_ignore_frank_nost = partial(unique_id_for_doc, prep_fn=prep_fn_ignore_frank_nost)


def remove_ignored_substrings(s, ignore_titles=False, ignore_link_attrs=True):
    if ignore_titles:
        s = re.sub(r"<h2>.*</h2>", "", s)
    if ignore_link_attrs:
        s = strip_link_attrs_regex.sub(strip_link_attrs_repl, s)
    s = s.replace("\n", "").replace(" ", "")
    s = re.sub(r"\<.*?\>", "", s)
    return s


def get_title(doc):
    ccs = get_ccs_with_fixes(doc)
    if len(ccs) == 0:
        return
    start_ix = ccs[0][1] + len(ccs[0][0])
    end_ix = ccs[1][1] if len(ccs) > 1 else len(doc)

    subs = doc[start_ix:end_ix]

    titles = get_h2_regex.findall(subs)
    title = titles[0] if titles else ""

    other_content = ""
    if title:
        other_content = remove_ignored_substrings(subs, ignore_titles=True)
    return title, other_content


def post_after_cc_is_empty(doc, ccs, i, ignore_titles=False, verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    try:
        start_ix = ccs[i][1]
        start_ix += len(ccs[i][0])
        end_ix = ccs[i + 1][1]
        interior = doc[start_ix:end_ix]
        vprint(f"\n\tinterior before replace:\n\t{repr(interior)}")
        interior = remove_ignored_substrings(interior, ignore_titles=ignore_titles)
        vprint(f"\n\tchecking interior:\n\t{repr(interior)}")
        return len(interior) == 0
    except IndexError:
        return False  # end of list


def diagnose_malformed(doc, verbose=False, use_naive_h2_pattern=False):
    reasons = set()

    if use_naive_h2_pattern and "<h2>" in doc:
        # check first-post-is-only-title pattern
        ccs = get_ccs_with_fixes(doc)
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


def extract_prefix(doc, include_username=False, ignore_titles=False, verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    vprint(f"include_username={include_username}")

    ccs = get_ccs_with_fixes(doc)

    vprint("base ccs:")
    vprint(ccs)

    pre_content = ""
    if len(ccs) > 1 and "asked:" in ccs[0][0]:
        left = ccs[0][1] if include_username else ccs[0][1] + len(ccs[0][0].rstrip("\n"))
        pre_content = doc[left:ccs[1][1]]
        ccs.pop(0)

    while post_after_cc_is_empty(doc, ccs, 0, ignore_titles=ignore_titles, verbose=verbose):
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
        lines = [l for l in lines if not (l.startswith(" Written") and " | " in l and "tags:" in l)]
        # lines.pop(2)
        vprint("\nlines after remove Written:")
        vprint(lines)
        prefix = "\n".join(lines)

    vprint("\nbase prefix:")
    vprint(repr(prefix))

    if prefix.startswith("#"):
        prefix = prefix.partition(" ")[2]

    prefix = pre_content + prefix
    prefix = remove_ignored_substrings(prefix, ignore_titles=ignore_titles)

    return prefix


def find_title_groups(docs, nontrivial_only=True):
    titlestuff = {}

    ixs = list(range(len(docs)))

    for i in tqdm(ixs):
        title, other_content = get_title(docs[i])

        if title:
            if title not in titlestuff:
                titlestuff[title] = {"ixs": [], "others": []}
            titlestuff[title]["ixs"].append(i)
            titlestuff[title]["others"].append(other_content)

    if nontrivial_only:
        titlestuff = {k: d for k, d in titlestuff.items() if len(d['ixs']) > 1}
    return titlestuff


def identify_h2_contaminated_docs(docs):
    excluded_doc_indices = set()

    titlestuff = find_title_groups(docs, nontrivial_only=True)

    for info in titlestuff.values():
        bad_ixs = {i for i, o in zip(info['ixs'], info['others']) if len(o) == 0}
        excluded_doc_indices.update(bad_ixs)

    return excluded_doc_indices


def map_docs(docs, include_usernames=False, ignore_titles=False):
    trails = defaultdict(set)

    for i, doc in tqdm(enumerate(docs), total=len(docs)):
        prefix = extract_prefix(doc, ignore_titles=ignore_titles)
        prefix_hash = hashlib.md5(prefix.encode("utf-8")).hexdigest()
        trails[prefix_hash].add(i)

    return trails


def map_docs_multiple_groups(*doc_groups, include_usernames=False):
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


def load_trails_from_docs(paths,
                          include_usernames=False,
                          exclude_malformed=True,
                          exclude_h2_issue=True,
                          exclude_unslimmable=True,
                          uid_to_metadata=None,
                          uid_fn=unique_id_ignore_frank_nost,
                          return_excluded_uids=False,
                          return_slimmed=False):

    using_uid_map = uid_to_metadata is not None
    doc_to_uid = {}
    excluded_uids = set()
    slimmed = {}

    if isinstance(paths, str):
        paths = [paths]

    doc_groups = []
    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            ds = f.read()
        g = [d for d in ds.split(EOT) if len(d) > 0]
        n_raw = len(g)
        print(f"read group from file {p}:\n\t{n_raw} raw docs\n")

        if using_uid_map:
            g_doc_to_uid = {}
            n_match_uids = 0
            for d in g:
                uid = uid_fn(d)
                lookedup = uid_to_metadata.get(uid)
                g_doc_to_uid[uid] = lookedup
                if lookedup is not None:
                    n_match_uids += 1
            doc_to_uid.update(g_doc_to_uid)
            print(f"read group from file {p}:\n\t{n_match_uids} of {n_raw} ({n_match_uids/n_raw:.1%}) have metadata\n")

        if exclude_unslimmable:
            bad = []
            print('slimming...')
            for d in tqdm(g):
                try:
                    slimmed[d] = slim_forumlike_doc(d, verbose=False)
                except (ValueError, IndexError):
                    bad.append(d)
            n_raw = len(g)
            g = [d for d in g if d in slimmed]
            n_excluded = n_raw - len(g)
            print(f"\tremoved {n_excluded} unslimmable docs ({n_excluded/n_raw:.2%})\n")

            print(f"verifying code works: {len(g)} docs after (vs {n_raw}, diff {n_raw - len(g)})")

            examples = bad[:3]
            for case in examples:
                print(f"\t\texample unslimmable doc:\n\t\t\t{repr(case)}\n")

            if using_uid_map:
                bad = [uid_fn(d) for d in bad]
                bad_with_meta = [uid for uid in bad if uid in uid_to_metadata]

                nbad_meta = len(bad_with_meta)
                print(f"\t{nbad_meta} of {n_excluded} ({nbad_meta/max(1, n_excluded):.2%}) have metadata\n")

                excluded_uids.update(bad_with_meta)

        if exclude_malformed:
            n_raw = len(g)
            reasons = [diagnose_malformed(d) for d in g]
            presentation_counts = Counter([tuple(sorted(r)) for r in reasons])
            symptom_counts = Counter([r for rs in reasons for r in rs])

            example_cases = defaultdict(list)
            for presentation in presentation_counts.keys():
                if presentation == tuple():
                    continue
                for d, r in zip(g, reasons):
                    if presentation == tuple(sorted(r)):
                        example_cases[presentation].append(d)
                        if len(example_cases[presentation]) > 2:
                            break

            bad = [d for d, r in zip(g, reasons) if r != set()]
            g = [d for d, r in zip(g, reasons) if r == set()]
            n_excluded = n_raw - len(g)
            print(f"\tremoved {n_excluded} malformed docs ({n_excluded/n_raw:.2%})\n")
            if using_uid_map:
                bad = [uid_fn(d) for d in bad]
                bad_with_meta = [uid for uid in bad if uid in uid_to_metadata]

                nbad_meta = len(bad_with_meta)
                print(f"\t{nbad_meta} of {n_excluded} ({nbad_meta/max(1, n_excluded):.2%}) have metadata\n")

                excluded_uids.update(bad_with_meta)

            print(f"\t\treasons:\n\t\t\t{repr(symptom_counts)}\n")
            for k in example_cases.keys():
                for case in example_cases[k]:
                    print(f"\t\texample for {k}:\n\t\t\t{repr(case)}\n")

        print(f"\t{len(g)} docs after exclusions\n\n")
        doc_groups.append(g)
    print()

    if exclude_h2_issue:
        all_docs = [d for g in doc_groups for d in g]
        n_raw = len(all_docs)

        excluded_doc_indices = identify_h2_contaminated_docs(all_docs)

        allowed_doc_indices_by_group = {}
        excluded_doc_indices_by_group = {}

        doc_index_offset = 0
        for group_index, g in enumerate(doc_groups):
            group_doc_indices = set(range(doc_index_offset, doc_index_offset + len(g)))

            excluded_doc_indices_by_group[group_index] = {
                ix - doc_index_offset for ix in group_doc_indices.intersection(excluded_doc_indices)
            }
            allowed_doc_indices_by_group[group_index] = {
                ix - doc_index_offset for ix in group_doc_indices.difference(excluded_doc_indices)
            }

            doc_index_offset += len(g)

        n_excluded = len(excluded_doc_indices)
        print(f"\tremoved {n_excluded} h2 bugged docs ({n_excluded/n_raw:.2%})\n")

        n_excluded_verify = sum(len(v) for v in excluded_doc_indices_by_group.values())
        print(f"\t[verifying code works:] removed {n_excluded} h2 bugged docs ({n_excluded/n_raw:.2%})\n")

        for group_index, excluded_indices in excluded_doc_indices_by_group.items():
            n_excluded_group = len(excluded_indices)
            print(f"group {group_index}: {n_excluded_group} excluded")
            if using_uid_map:
                bad = [uid_fn(doc_groups[group_index][i]) for i in excluded_indices]
                bad_with_meta = [uid for uid in bad if uid in uid_to_metadata]
                nbad_meta = len(bad_with_meta)
                print(f"\tgroup {group_index}: {nbad_meta} of {n_excluded_group} ({nbad_meta/max(1, n_excluded_group):.2%}) have metadata\n")
                excluded_uids.update(bad_with_meta)

            examples = [doc_groups[group_index][i] for i in excluded_indices][:2]
            for case in examples:
                print(f"\t\texample for h2 bug (group {group_index}):\n\t\t\t{repr(case)}\n")

        for group_index, allowed_doc_indices in allowed_doc_indices_by_group.items():
            doc_groups[group_index] = [doc_groups[group_index][i] for i in sorted(allowed_doc_indices)]

        print(f"\t[verifying code works:] n docs after h2 fix: {sum(len(g) for g in doc_groups)}\n")

    docs, trails, doc_index_to_group_index = map_docs_multiple_groups(*doc_groups, include_usernames=include_usernames)

    nt = nontrivial_trails(trails)

    trail_stats(docs, trails, nt)

    extra_return_values = {}

    if return_excluded_uids:
        extra_return_values['excluded_uids'] = excluded_uids
    if return_slimmed:
        extra_return_values['slimmed'] = slimmed
    return docs, trails, nt, doc_index_to_group_index, extra_return_values


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


def fallback_keyfn(doc):
    try:
        ts = extract_time_from_forumlike_doc(doc).timestamp()
    except ValueError:
        return ('zzzzzzz', -1)
    ccs = get_ccs_with_fixes(doc)
    blogname = ccs[-1][0].split(" ")[1]
    return (blogname, ts)


def sort_by_username_post_id(docs, uid_to_metadata):
    uname_order = {}

    def keyfn(doc):
        uid = unique_id_ignore_frank_nost(doc)
        meta = uid_to_metadata.get(uid)
        if meta:
            if 'blogname' in meta and 'post_id' in meta:
                key = (meta['blogname'], meta['post_id'])
            else:
                key = meta['fallback_key']
        key = fallback_keyfn(doc)

        # prevent unames from being alphabetical
        if key[0] not in uname_order:
            uname_order[key[0]] = random.randint(0, 10000)
        key = (uname_order[key[0]], key[1])
        return key

    return sorted(docs, key=keyfn), keyfn, uname_order
