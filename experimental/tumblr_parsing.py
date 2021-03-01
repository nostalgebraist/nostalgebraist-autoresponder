from typing import List, Optional
from collections import defaultdict
from itertools import zip_longest


class TumblrContentBlockBase:
    def to_html(self) -> str:
        raise NotImplementedError


class LegacyBlock(TumblrContentBlockBase):
    def __init__(self, body: str):
        self._body = body

    @property
    def body(self):
        return self._body

    def to_html(self):
        return self._body


class NPFFormattingRange:
    def __init__(
        self,
        start: int,
        end: int,
        type: str,
        url: Optional[str] = None,
        blog: Optional[dict] = None,
        hex: Optional[str] = None,
    ):
        self.start = start
        self.end = end
        self.type = type

        self.url = url
        self.blog = blog
        self.hex = hex

    def to_html(self):
        result = {"start": self.start, "end": self.end}

        types_to_style_tags = {
            "bold": "b",
            "italic": "i",
            "small": "small",
            "strikethrough": "strike",
        }

        if self.type in types_to_style_tags:
            tag = types_to_style_tags[self.type]
            result["start_insert"] = f"<{tag}>"
            result["end_insert"] = f"</{tag}>"
        elif self.type == "link":
            result["start_insert"] = f"<a href=\"{self.url}\">"
            result["end_insert"] = f"</a>"
        elif self.type == "mention":
            result["start_insert"] = f'<a class="tumblelog" href=\"{self.url}\">'
            result["end_insert"] = f"</a>"
        elif self.type == "color":
            result["start_insert"] = f'<span style="color:{self.hex}">'
            result["end_insert"] = f"</span>"
        else:
            raise ValueError(self.type)
        return result


class NPFSubtype:
    def __init__(self, subtype: str):
        self.subtype = subtype

    def format_html(self, text: str):
        if self.subtype == "heading1":
            return f"<p><h2>{text}</h2></p>"  # TODO: verify
        elif self.subtype == "heading2":
            return f"<p><h2>{text}</h2></p>"  # TODO: verify
        elif self.subtype == "ordered-list-item":
            return f"<li>{text}</li>"
        elif self.subtype == "unordered-list-item":
            return f"<li>{text}</li>"
        else:
            return f"<p>{text}</p>"


class NPFTextBlock(TumblrContentBlockBase):
    def __init__(
        self,
        text: str,
        subtype: Optional[NPFSubtype] = None,
        indent_level: Optional[int] = None,
        formatting: List[NPFFormattingRange] = None,
    ):
        self.text = text
        self.subtype = NPFSubtype("no_subtype") if subtype is None else subtype
        self.indent_level = 0 if indent_level is None else indent_level
        self.formatting = [] if formatting is None else formatting

    @property
    def subtype_name(self):
        return self.subtype.subtype

    def apply_formatting(self):
        insertions = [formatting.to_html() for formatting in self.formatting]

        insert_ix_to_inserted_text = defaultdict(list)
        for insertion in insertions:
            insert_ix_to_inserted_text[insertion["start"]].append(
                insertion["start_insert"]
            )
            insert_ix_to_inserted_text[insertion["end"]].append(insertion["end_insert"])

        split_ixs = {0, len(self.text)}
        split_ixs.update(insert_ix_to_inserted_text.keys())
        split_ixs = sorted(split_ixs)

        accum = []

        for ix1, ix2 in zip_longest(split_ixs, split_ixs[1:], fillvalue=split_ixs[-1]):
            accum.extend(insert_ix_to_inserted_text[ix1])
            accum.append(self.text[ix1:ix2])

        return "".join(accum)

    def to_html(self):
        formatted = self.apply_formatting()

        if self.subtype is not None:
            formatted = self.subtype.format_html(formatted)

        return formatted

    @staticmethod
    def from_payload(payload: dict) -> "NPFTextBlock":
        return NPFTextBlock(
            text=payload["text"],
            subtype=NPFSubtype(subtype=payload.get("subtype", "no_subtype")),
            indent_level=payload.get("indent_level"),
            formatting=[
                NPFFormattingRange(**entry) for entry in payload.get("formatting", [])
            ],
        )


class TumblrContentBase:
    def __init__(self, content: List[TumblrContentBlockBase]):
        self.content = content

    def to_html(self) -> str:
        raise NotImplementedError


class NPFBlockAnnotated(TumblrContentBlockBase):
    def __init__(self, base_block: TumblrContentBlockBase):
        self.base_block = base_block

        self.prefix = ""
        self.suffix = ""
        self.indent_delta = None

    def to_html(self):
        inside = self.base_block.to_html()
        return self.prefix + inside + self.suffix


class NPFContent(TumblrContentBase):
    def __init__(self, blocks: List[TumblrContentBlockBase]):
        self.blocks = [
            block if isinstance(block, NPFBlockAnnotated) else NPFBlockAnnotated(block)
            for block in blocks
        ]

    def _assign_indents(self):
        # TODO: handle the way that 1st blockquote/list level has indent_level 0
        cur_level = 0

        for block in self.blocks:
            block.indent_delta = block.base_block.indent_level - cur_level
            cur_level = block.base_block.indent_level

        # TODO: what happens if nonzero indent_level at final block?

    def _assign_nonlocal_tags(self):
        subtype_and_sign_to_tag = {
            ("indented", True): "<blockquote>",
            ("indented", False): "</blockquote>",
            ("ordered-list-item", True): "<ol>",
            ("ordered-list-item", False): "</ol>",
            ("unordered-list-item", True): "<ul>",
            ("unordered-list-item", False): "</ul>",
        }

        for block in self.blocks:
            if block.indent_delta == 0:
                continue

            sign = block.indent_delta > 0
            abs = block.indent_delta if sign else -1 * block.indent_delta

            key = (block.base_block.subtype_name, sign)
            if key not in subtype_and_sign_to_tag:
                raise ValueError(key)  # TODO: improve

            tag = subtype_and_sign_to_tag[key]
            tags = abs * tag

            if sign:
                block.prefix = tags
            else:
                block.suffix = tags

    def to_html(self):
        self._assign_indents()
        self._assign_nonlocal_tags()

        return "".join([block.to_html() for block in self.blocks])


class TumblrPostBase:
    def __init__(
        self,
        blog_name: str,
        content: TumblrContentBase,
    ):
        self._blog_name = blog_name
        self._content = content

    @property
    def blog_name(self):
        return self._blog_name

    @property
    def content(self):
        return self._content


class TumblrPost(TumblrPostBase):
    def __init__(
        self,
        blog_name: str,
        content: TumblrContentBase,
        tags: Optional[List[str]],
    ):
        self._blog_name = blog_name
        self._content = content
        self._tags = tags

    @property
    def tags(self):
        return self._tags


class TumblrAsk(TumblrPostBase):
    pass


class TumblrAnswer(TumblrPost):
    def __init__(self, ask: TumblrAsk, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ask = ask

    @property
    def ask(self):
        return self._ask


class TumblrThread:
    def __init__(self, posts: List[TumblrPost]):
        self._posts = posts

    @property
    def posts(self):
        return self._posts
