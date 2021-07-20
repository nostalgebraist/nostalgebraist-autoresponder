import hashlib
from collections import defaultdict
from tqdm.auto import tqdm

from tumblr_to_text.classic.autoresponder_static import find_control_chars_forumlike


def extract_prefix(doc, verbose=False):
    def vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)

    extra_names = doc.split(" ")[1:2]
    if extra_names[0] == 'nostalgebraist-autoresponder':
        extra_names = []

    ccs = find_control_chars_forumlike(doc, extra_names=extra_names)
    vprint(ccs)

    pre_content = ""
    if len(ccs) > 1 and "asked:" in ccs[0][0]:
        pre_content = doc[ccs[0][1]:ccs[1][1]]
        ccs.pop(0)
    vprint(ccs)

    if len(ccs) > 1:
        prefix = doc[ccs[0][1]:ccs[1][1]]
    else:
        prefix = doc[ccs[0][1]:]
        lines = prefix.splitlines()
        lines.pop(2)
        prefix = "\n".join(lines)
    vprint(prefix)

    if prefix.startswith("#"):
        prefix = prefix.partition(" ")[2]

    prefix = pre_content + prefix
    prefix = prefix.replace("\n", "").replace(" ", "")
    return prefix


def map_docs(docs):
    trails = defaultdict(set)

    for i, doc in tqdm(enumerate(docs), total=len(docs)):
        prefix = extract_prefix(doc)
        prefix_hash = hashlib.md5(prefix.encode("utf-8")).hexdigest()
        trails[prefix_hash].add(i)

    return trails


def nontrivial_trails(trails):
    return {k: v for k, v in trails.items() if len(v) > 1}


def show_trail(docs, trails, key):
    tdocs = sorted([docs[i] for i in trails[key]], key=len)
    for td in tdocs:
        print(repr(td))
        print()
        print(repr(extract_prefix(td)))
        print('\n------\n')
