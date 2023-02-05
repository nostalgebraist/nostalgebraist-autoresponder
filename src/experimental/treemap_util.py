import re
from datetime import datetime
from dataclasses import dataclass

from experimental.corpus_thread_util import *
from tumblr_to_text.classic.autoresponder_static import EOT
from tumblr_to_text.endtags import move_tags_to_end

collect_metadata_pat = re.compile(
    r"(?P<meta> Written [0-9]{1,2} [APM]{2,} [A-Z][a-z]{1,15} [0-9]{4,} \| [^\n]+\n)(?P<post>.*)",
    flags=re.DOTALL,
)



"""
    TODO:
            - [to verify] proper handling for 'Posts by'
            - [X] collapse forms of a post that do vs. don't have the 'written 11 AM (etc)' segment
                - [X] relatedly, ensure we can force a post to be viewed as 'final' in inference (for getting tags)

    basic plan

            - get rid of "Conversation between" and "Posts by" prefixes
            - always include tags and timestamps if we have them, for every post, even non-final ones
            - the timestamp string ("Written [...]") can be used in prompts to indicate that this is a
              post where tags are known
"""

def collect_available_post_metadata(docs, post_text_to_meta_strings=None, cc_fails=None):
    post_text_to_meta_strings = post_text_to_meta_strings or defaultdict(set)

    cc_fails = cc_fails or []

    pbar = tqdm(docs)

    for i, d in enumerate(pbar):
        try:
            cc = get_ccs_with_fixes(d)[-1]
        except:
            cc_fails.append(d)

        seg = d[(cc[1]+len(cc[0])):]

        for m in collect_metadata_pat.finditer(seg):
            gd = m.groupdict()

            meta = gd['meta']
            user_posted = cc[0]
            post = gd['post']

            post_text_to_meta_strings[(user_posted, post)].add(meta)

    return post_text_to_meta_strings, cc_fails


def get_ts_from_meta_string(meta_string):
    time_segment = meta_string.split("Written ")[1].split(" |")[0]

    return datetime.strptime(time_segment, "%I %p %B %Y")


def pick_between_meta_strings(post_text_to_meta_strings,  no_timestamps=True):
    collapsed_post_text_to_meta_strings = {}

    ks = list(post_text_to_meta_strings.keys())
    pbar = tqdm(ks)

    for k in pbar:

        meta_strings = post_text_to_meta_strings[k]

        if no_timestamps:
            entry = list(meta_strings)[0]

            ts_string, sep, tag_string = entry.partition(" | ")
            entry = " Written PLACEHOLDER" + sep + tag_string
        else:
            if len(meta_strings) > 1:
                entry = sorted(meta_strings, key=get_ts_from_meta_string)[0]
            else:
                entry = list(meta_strings)[0]

        collapsed_post_text_to_meta_strings[k] = entry

    return collapsed_post_text_to_meta_strings


def use_meta_if_available(docs, collapsed_post_text_to_meta_strings, verbose=False):
    """i wrote a regex version first, but it was much slower than this"""
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    pbar = tqdm(docs)

    out = []
    affected = []

    for i, d in enumerate(pbar):
        doc_subbed = ''

        ccs = get_ccs_with_fixes(d)

        offset = ccs[0][1]
        doc_subbed += d[:offset]

        for cc1, cc2 in zip(ccs[:-1], ccs[1:]):
            end_index = cc2[1]
            user_posted_start_index = cc1[1]
            user_posted_end_index = user_posted_start_index + len(cc1[0])

            user_posted = d[user_posted_start_index:user_posted_end_index]
            post = d[user_posted_end_index:end_index].rstrip("\n")

            meta_key = (user_posted, post)

            seg = d[user_posted_end_index:end_index]
            meta_string = collapsed_post_text_to_meta_strings.get(meta_key, '')

            vprint(f"cc {cc1}, seg {seg}")
            vprint(f"meta_key {meta_key}\nmeta_key found: {meta_key in collapsed_post_text_to_meta_strings}")
            vprint()

            doc_subbed += d[offset:user_posted_end_index]
            doc_subbed += meta_string
            doc_subbed += seg
            offset = end_index

        affected.append(doc_subbed != d[:offset])

        cc1 = ccs[-1]
        final = d[cc1[1]:]

        for m in collect_metadata_pat.finditer(final):
            gd = m.groupdict()
            meta = gd['meta']
            user_posted_start_index = cc1[1]
            user_posted_end_index = user_posted_start_index + len(cc1[0])

            user_posted = d[user_posted_start_index:user_posted_end_index]
            post = gd['post']

            meta_key = (user_posted, post)

            meta_string = collapsed_post_text_to_meta_strings.get(meta_key, meta)

            vprint(f"cc {ccs[-1]}, seg {post}")
            vprint(f"meta_key {meta_key}\nmeta_key found: {meta_key in collapsed_post_text_to_meta_strings}")
            vprint()

            doc_subbed += d[offset:user_posted_end_index]
            doc_subbed += meta_string
            doc_subbed += post

        out.append(doc_subbed)

        if i % 500 == 0:
            pbar.set_postfix(n_affected=sum(affected), refresh=False)

    return out, affected


def split_tree(doc, include_username=False, ignore_titles=False, verbose=False):
    """
    TODO: stop handling the "prefix" differently from later posts here
    """
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    def postprocess_segment_to_segrep(raw_segment, pre_content=""):
        segment = raw_segment

        if segment.startswith("#"):
            segment = segment.partition(" ")[2]

        segment = pre_content + segment
        segment = remove_ignored_substrings(segment, ignore_titles=ignore_titles)
        segment = hashlib.md5(segment.encode("utf-8")).hexdigest()
        return segment

    def skip_empties(text, ccs, pre_content=""):
        while post_after_cc_is_empty(text, ccs, 0, ignore_titles=ignore_titles, verbose=verbose):
            left = ccs[0][1] if include_username else ccs[0][1] + len(ccs[0][0].rstrip("\n"))
            pre_content += text[left:ccs[1][1]]
            ccs.pop(0)
        return ccs, pre_content

    out_data = []
    segment_offset = 0

    vprint(f"include_username={include_username}")

    ccs = get_ccs_with_fixes(doc)

    vprint("base ccs:")
    vprint(ccs)

    segment_offset = None

    pre_content = ""
    if len(ccs) > 1 and "asked:" in ccs[0][0]:
        left = ccs[0][1] if include_username else ccs[0][1] + len(ccs[0][0].rstrip("\n"))
        pre_content = doc[left:ccs[1][1]]
        ccs.pop(0)
        segment_offset = left

    vprint(("pre_content", pre_content))

    ccs, pre_content = skip_empties(doc, ccs, pre_content=pre_content)

    vprint(("pre_content", pre_content))

    vprint("\nccs after pop ask/empty:")
    vprint(ccs)

    end_index = None

    if len(ccs) > 1:
        left = ccs[0][1] if include_username else ccs[0][1] + len(ccs[0][0].rstrip("\n"))
        prefix = doc[left:ccs[1][1]]
        end_index = ccs[1][1]
        if segment_offset is None:
            segment_offset = ccs[0][1]
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
        end_index = len(doc)
        cc_offset = 0
        segment_offset = 0


    vprint("\nbase prefix:")
    vprint(repr(prefix))

    seg = doc[segment_offset:end_index]
    segrep = postprocess_segment_to_segrep(prefix, pre_content=pre_content)

    out_data.append(
        dict(
            seg=doc[segment_offset:end_index],
            rep=segrep,
            start=segment_offset,
            end=end_index
        )
    )

    cc_offset = 2
    vprint()
    ccs.append(('', len(doc)))

    while cc_offset < len(ccs):
        cc = ccs[cc_offset]
        vprint(cc)

        segment_offset = end_index

        end_index = cc[1]
        seg = doc[segment_offset:end_index]
        segrep = postprocess_segment_to_segrep(seg)

        out_data.append(
            dict(
                seg=seg,
                rep=segrep,
                start=segment_offset,
                end=end_index
            )
        )
        cc_offset += 1

    return out_data


### tree stuff


@dataclass
class CorpusTreesInfo:
    rep2seg: dict
    rep_unmap: dict
    node2prefix: dict

# @dataclass
# class TreeInfo:
#     path_reps: list


class TreeRepCompare:
    def __init__(self, rep: tuple):
        self.rep = rep

    def __lt__(self, other):
        out = self.rep != other.rep and is_prefix(self.rep, other.rep)
        return out


def is_prefix(this, maybe_prefix):
    return this[:len(maybe_prefix)] == maybe_prefix


def construct_trees(docs):
    rep2seg = {}
    rep_map = {}
    path_reps = []

    for i, d in enumerate(docs):
        out_data = split_tree(d, include_username=True, verbose=False, )
        for d in out_data:
            rep2seg[d['rep']] = d['seg']

        def map_rep(rep):
            if rep not in rep_map:
                rep_map[rep] = len(rep_map)
            return rep_map[rep]

        path_reps.append(tuple(map_rep(d['rep']) for d in out_data))

    rep_unmap = {v: k for k, v in rep_map.items()}

    node2prefix = {}
    for path in path_reps:
        for i, node in enumerate(path):
            node2prefix[node] = path[:i+1]

    corpus_info = CorpusTreesInfo(
        rep2seg=rep2seg,
        rep_unmap=rep_unmap,
        node2prefix=node2prefix,
    )

    trees = defaultdict(set)

    for path in path_reps:
        root = path[0]
        trees[root].add(path)

    return corpus_info, trees


def map_tree(corpus_info: CorpusTreesInfo, path_reps: list):
    tree = defaultdict(set)

    for path in path_reps:
        for depth, node in enumerate(path):
            tree[depth].add(node)

    tree = {depth: {node: i for i, node in enumerate(nodes)} for depth, nodes in tree.items()}

    tree_reps = []
    for path in path_reps:
        rep = []
        for depth, node in enumerate(path):
            rep.append(tree[depth][node])

        tree_reps.append(tuple(rep))
    return tree_reps, tree


def serialize_tree(corpus_info: CorpusTreesInfo, path_reps: list):
    tree_reps, tree = map_tree(corpus_info, path_reps)

    path_order = sorted(range(len(tree_reps)),
                        key=lambda i: TreeRepCompare(tree_reps[i])
                       )

    seg_indices = []
    is_leaf = []
    written = set()

    for j in path_order:
        path = path_reps[j]
        for elem in path:
            if elem not in written:
                seg_indices.append(elem)
                is_leaf.append(elem == path[-1])
                written.add(elem)
    return seg_indices, is_leaf


def move_tags_and_fill_written(seg, is_leaf):
    seg = seg.rstrip("\n")  # newlines in pre-final posts -- will get re-added by move_tags_to_end
    seg = move_tags_to_end(seg)
    written_new = " leaf" if is_leaf else ""  # \t at end of this line already denotes whether tags are available
    seg = seg.replace(" Written PLACEHOLDER", written_new)
    return seg


def write_serialized_tree(corpus_info: CorpusTreesInfo, path_reps: list, seg_postprocessor=move_tags_and_fill_written):
    seg_postprocessor = seg_postprocessor or (lambda x, is_leaf_value: x)

    serial_order, is_leaf = serialize_tree(corpus_info, path_reps)

    serialized = []

    for i, is_leaf_value in zip(serial_order, is_leaf):
        seg = corpus_info.rep2seg[corpus_info.rep_unmap[i]]
        serialized.append(f"<meta>{repr(corpus_info.node2prefix[i])}</meta>{seg_postprocessor(seg, is_leaf_value)}")

    return serialized


def convert_docs_to_trees(docs):
    print('collecting metadata')
    post_text_to_meta_strings = defaultdict(set)
    cc_fails = []

    post_text_to_meta_strings, cc_fails = collect_available_post_metadata(docs)

    print(f"{len(cc_fails)} cc fails")

    print('substituting metadata')
    docs, affected = use_meta_if_available(docs, collapsed_post_text_to_meta_strings)
    print(f"{sum(affected)} of {len(docs)} docs changed by metadata substitution")

    print('constructing trees')
    corpus_info, trees = construct_trees(docs)

    print('serializing')

    serialized = []
    for t in trees.values():
        serialized.append(''.join(write_serialized_tree(corpus_info, list(t))))

    return serialized, trees, corpus_info


# def load_docs(
#     paths,
#     include_usernames=False,
#     exclude_malformed=True,
#     exclude_h2_issue=True,
#     exclude_h2_until_group=None,
#     exclude_unslimmable=True,
#     uid_to_metadata=None,
#     uid_fn=unique_id_ignore_frank_nost,
#     return_excluded_uids=False,
#     return_slimmed=False,
#     doc_preprocessor=strip_post_identifier,
#     exclude_nost_paths=set(),
#     keep_nost_reviews=True,
# ):
#     using_uid_map = uid_to_metadata is not None
#     doc_to_uid = {}
#     excluded_uids = set()
#     slimmed = {}
#     excluded_nost_docs = defaultdict(list)
#
#     if isinstance(paths, str):
#         paths = [paths]
#
#     doc_groups = []
#     for p in paths:
#         # with open(p, "r", encoding="utf-8") as f:
#         #     ds = f.read()
#         # g = [d for d in ds.split(EOT) if len(d) > 0]
#         g = stream_read_docs(p)
#         if doc_preprocessor is not None:
#             g = [doc_preprocessor(d) for d in g]
#         n_raw = len(g)
#         print(f"read group from file {p}:\n\t{n_raw} raw docs\n")
#
#         if p in exclude_nost_paths:
#             g_ = []
#             for _ in range(len(g)):
#                 d = g.pop(0)
#                 if exclude_nost_check(d, keep_nost_reviews=keep_nost_reviews):
#                     g_.append(d)
#                 else:
#                     excluded_nost_docs[p].append(d)
#             g = g_
#             # g = [d for d in g if exclude_nost_check(d, keep_nost_reviews=keep_nost_reviews)]
#             delta = n_raw - len(g)
#             n_raw = len(g)
#             print(f"excluded {delta} nost docs from file {p}:\n\t{n_raw} docs left\n")
#
#         if using_uid_map:
#             g_doc_to_uid = {}
#             n_match_uids = 0
#             for d in g:
#                 uid = uid_fn(d)
#                 lookedup = uid_to_metadata.get(uid)
#                 g_doc_to_uid[uid] = lookedup
#                 if lookedup is not None:
#                     n_match_uids += 1
#             doc_to_uid.update(g_doc_to_uid)
#             print(f"read group from file {p}:\n\t{n_match_uids} of {n_raw} ({n_match_uids/n_raw:.1%}) have metadata\n")
#
#         if exclude_unslimmable:
#             bad = []
#             print('slimming...')
#             for d in tqdm(g):
#                 try:
#                     slimmed[d] = slim_forumlike_doc(d, verbose=False)
#                 except (ValueError, IndexError):
#                     bad.append(d)
#             n_raw = len(g)
#             g = [d for d in g if d in slimmed]
#             n_excluded = n_raw - len(g)
#             print(f"\tremoved {n_excluded} unslimmable docs ({n_excluded/n_raw:.2%})\n")
#
#             print(f"verifying code works: {len(g)} docs after (vs {n_raw}, diff {n_raw - len(g)})")
#
#             examples = bad[:3]
#             for case in examples:
#                 print(f"\t\texample unslimmable doc:\n\t\t\t{repr(case)}\n")
#
#             if using_uid_map:
#                 bad = [uid_fn(d) for d in bad]
#                 bad_with_meta = [uid for uid in bad if uid in uid_to_metadata]
#
#                 nbad_meta = len(bad_with_meta)
#                 print(f"\t{nbad_meta} of {n_excluded} ({nbad_meta/max(1, n_excluded):.2%}) have metadata\n")
#
#                 excluded_uids.update(bad_with_meta)
#
#         if exclude_malformed:
#             n_raw = len(g)
#             reasons = [diagnose_malformed(d) for d in g]
#             presentation_counts = Counter([tuple(sorted(r)) for r in reasons])
#             symptom_counts = Counter([r for rs in reasons for r in rs])
#
#             example_cases = defaultdict(list)
#             for presentation in presentation_counts.keys():
#                 if presentation == tuple():
#                     continue
#                 for d, r in zip(g, reasons):
#                     if presentation == tuple(sorted(r)):
#                         example_cases[presentation].append(d)
#                         if len(example_cases[presentation]) > 2:
#                             break
#
#             bad = [d for d, r in zip(g, reasons) if r != set()]
#             g = [d for d, r in zip(g, reasons) if r == set()]
#             n_excluded = n_raw - len(g)
#             print(f"\tremoved {n_excluded} malformed docs ({n_excluded/n_raw:.2%})\n")
#             if using_uid_map:
#                 bad = [uid_fn(d) for d in bad]
#                 bad_with_meta = [uid for uid in bad if uid in uid_to_metadata]
#
#                 nbad_meta = len(bad_with_meta)
#                 print(f"\t{nbad_meta} of {n_excluded} ({nbad_meta/max(1, n_excluded):.2%}) have metadata\n")
#
#                 excluded_uids.update(bad_with_meta)
#
#             print(f"\t\treasons:\n\t\t\t{repr(symptom_counts)}\n")
#             for k in example_cases.keys():
#                 for case in example_cases[k]:
#                     print(f"\t\texample for {k}:\n\t\t\t{repr(case)}\n")
#
#         print(f"\t{len(g)} docs after exclusions\n\n")
#         doc_groups.append(g)
#     print()
#
#     if exclude_h2_issue:
#         if exclude_h2_until_group is None:
#             exclude_h2_until_group = len(doc_groups)
#
#         all_h2_docs = [d for g in doc_groups[:exclude_h2_until_group] for d in g]
#         n_raw = len(all_h2_docs)
#
#         excluded_doc_indices = identify_h2_contaminated_docs(all_h2_docs)
#
#         allowed_doc_indices_by_group = {}
#         excluded_doc_indices_by_group = {}
#
#         doc_index_offset = 0
#         for group_index, g in enumerate(doc_groups):
#             group_doc_indices = set(range(doc_index_offset, doc_index_offset + len(g)))
#
#             excluded_doc_indices_by_group[group_index] = {
#                 ix - doc_index_offset for ix in group_doc_indices.intersection(excluded_doc_indices)
#             }
#             allowed_doc_indices_by_group[group_index] = {
#                 ix - doc_index_offset for ix in group_doc_indices.difference(excluded_doc_indices)
#             }
#
#             doc_index_offset += len(g)
#
#         n_excluded = len(excluded_doc_indices)
#         print(f"\tremoved {n_excluded} h2 bugged docs ({n_excluded/n_raw:.2%})\n")
#
#         n_excluded_verify = sum(len(v) for v in excluded_doc_indices_by_group.values())
#         print(f"\t[verifying code works:] removed {n_excluded} h2 bugged docs ({n_excluded/n_raw:.2%})\n")
#
#         for group_index, excluded_indices in excluded_doc_indices_by_group.items():
#             n_excluded_group = len(excluded_indices)
#             print(f"group {group_index}: {n_excluded_group} excluded")
#             if using_uid_map:
#                 bad = [uid_fn(doc_groups[group_index][i]) for i in excluded_indices]
#                 bad_with_meta = [uid for uid in bad if uid in uid_to_metadata]
#                 nbad_meta = len(bad_with_meta)
#                 print(f"\tgroup {group_index}: {nbad_meta} of {n_excluded_group} ({nbad_meta/max(1, n_excluded_group):.2%}) have metadata\n")
#                 excluded_uids.update(bad_with_meta)
#
#             examples = [doc_groups[group_index][i] for i in excluded_indices][:2]
#             for case in examples:
#                 print(f"\t\texample for h2 bug (group {group_index}):\n\t\t\t{repr(case)}\n")
#
#         for group_index, allowed_doc_indices in allowed_doc_indices_by_group.items():
#             doc_groups[group_index] = [doc_groups[group_index][i] for i in sorted(allowed_doc_indices)]
#
#         print(f"\t[verifying code works:] n docs after h2 fix: {sum(len(g) for g in doc_groups)}\n")
#
#     return doc_groups
