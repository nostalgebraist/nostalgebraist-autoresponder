import re
from datetime import datetime
from dataclasses import dataclass

from experimental.corpus_thread_util import *
from tumblr_to_text.classic.autoresponder_static import EOT, DEFAULT_CSC
from tumblr_to_text.endtags import move_tags_to_end

import tqdm.contrib.concurrent as tqdm_contrib

from tqdm.auto import tqdm as tqdm_base
from tqdm.auto import trange as trange_base
tqdm = partial(tqdm_base, mininterval=1, smoothing=0)
trange = partial(trange_base, mininterval=1, smoothing=0)


collect_metadata_pat = re.compile(
    r"(?P<meta> Written [0-9]{1,2} [APM]{2,} [A-Z][a-z]{1,15} [0-9]{4,} \| [^\n]+(?:\n|$))(?P<post>.*)",
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


def collect_metadata_from_post_segments(doc):
    cc_failed_doc = None
    results = []

    try:
        cc = get_ccs_with_fixes(doc)[-1]
    except:
        cc_failed_doc = doc
        return results, cc_failed_doc

    seg = doc[(cc[1]+len(cc[0])):]

    for m in collect_metadata_pat.finditer(seg):
        gd = m.groupdict()

        meta = gd['meta']
        user_posted = cc[0]
        post = gd['post']

        results.append((user_posted, post, meta))
    return results, cc_failed_doc


def collect_available_post_metadata(docs, post_text_to_meta_strings=None, cc_fails=None, use_mp=True, max_workers=6, chunksize=16384):
    print('hi')
    post_text_to_meta_strings = post_text_to_meta_strings or defaultdict(set)

    cc_fails = cc_fails or []

    if use_mp:
        out = tqdm_contrib.process_map(
            collect_metadata_from_post_segments,
            docs,
            max_workers=max_workers, chunksize=chunksize,
            mininterval=1, smoothing=0,
        )
    else:
        out = [collect_metadata_from_post_segments(x) for x in tqdm(docs)]

    for results, cc_failed_doc in out:
        if cc_failed_doc:
            cc_fails.append(cc_failed_doc)

        for user_posted, post, meta in results:
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


def use_meta_if_available_single_doc(doc, collapsed_post_text_to_meta_strings, verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    if any([DEFAULT_CSC[k] in doc for k in ["ORIG_FICTION_CHAR_FORUMLIKE", "REVIEW_CHAR_FORUMLIKE"]]):
        return doc

    d = doc  # TODO: naming
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

    return doc_subbed


def use_meta_if_available(docs, collapsed_post_text_to_meta_strings, verbose=False, use_mp=True, max_workers=6, chunksize=16384):
    """i wrote a regex version first, but it was much slower than this"""
    out = []
    affected = []

    if use_mp:
        out = tqdm_contrib.process_map(
            partial(
                use_meta_if_available_single_doc,
                collapsed_post_text_to_meta_strings=collapsed_post_text_to_meta_strings
            ),
            docs,
            max_workers=max_workers, chunksize=chunksize,
            mininterval=1, smoothing=0,
        )
    else:
        out = [use_meta_if_available_single_doc(d, collapsed_post_text_to_meta_strings) for d in tqdm(docs)]

    return out


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

    for i, d in enumerate(tqdm(docs)):
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
    if " Written PLACEHOLDER" in seg:
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


def preprocess_docs_for_trees(docs, use_mp=(True, True), max_workers=6, chunksize=16384):
    print('collecting metadata')
    post_text_to_meta_strings = defaultdict(set)
    cc_fails = []

    post_text_to_meta_strings, cc_fails = collect_available_post_metadata(
        docs,
        use_mp=use_mp[0],
        max_workers=max_workers,
        chunksize=chunksize,
    )

    collapsed_post_text_to_meta_strings = pick_between_meta_strings(post_text_to_meta_strings)

    print(f"{len(cc_fails)} cc fails")

    print('substituting metadata')
    docs = use_meta_if_available(
        docs, collapsed_post_text_to_meta_strings, use_mp=use_mp[1],
        max_workers=max_workers,
        chunksize=chunksize,
    )
    # print(f"{sum(affected)} of {len(docs)} docs changed by metadata substitution")

    return docs


def convert_docs_to_trees(docs):
    print('constructing trees')
    corpus_info, trees = construct_trees(docs)

    print('serializing')

    serialized = []
    for t in tqdm(trees.values(), total=len(trees)):
        serialized.append(''.join(write_serialized_tree(corpus_info, list(t))))

    return serialized, trees, corpus_info
