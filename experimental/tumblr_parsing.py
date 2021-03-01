from typing import List, Optional
from collections import defaultdict


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
    def __init__(self,
                 start: int,
                 end: int,
                 type: str,
                 url: Optional[str],
                 blog: Optional[dict],
                 hex: Optional[str]
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
            result["start_insert"] = f"<a href={self.url}>"
            result["end_insert"] = f"</a>"
        elif self.type == "mention":
            result["start_insert"] = f"<a class=\"tumblelog\" href={self.url}>"
            result["end_insert"] = f"</a>"
        elif self.type == "color":
            result["start_insert"] = f"<span style=\"color:{self.hex}\">"
            result["end_insert"] = f"</span>"
        else:
            raise ValueError(self.type)
        return result


class NPFSubtype:
    def __init__(self, subtype: str):
        self.subtype = subtype

    def format_html(self, text: str):
        if self.subtype == "heading1":
            return f"<h2>{text}</h2>"  # TODO: verify
        elif self.subtype == "heading2":
            return f"<h2>{text}</h2>"  # TODO: verify
        elif self.subtype == "indented":
            return f"<blockquote>{text}</blockquote>"
        elif self.subtype == "ordered-list-item":
            return f"<li>{text}</li>"
        elif self.subtype == "unordered-list-item":
            return f"<li>{text}</li>"
        else:
            return text


class NPFTextBlock(TumblrContentBlockBase):
    def __init__(self,
                 text: str,
                 subtype: Optional[NPFSubtype],
                 indent_level: Optional[int],
                 formatting: List[NPFFormattingRange]
                 ):
        self.text = text
        self.subtype = subtype
        self.indent_level = indent_level
        self.formatting = formatting

    def apply_formatting(self):
        insertions = [formatting.to_html() for formatting in self.formatting]

        insert_ix_to_inserted_text = defaultdict(list)
        for insertion in insertions:
            insert_ix_to_inserted_text[insertion['start']].append(insertion['start_insert'])
            insert_ix_to_inserted_text[insertion['end']].append(insertion['end_insert'])

        split_ixs = {0, len(self.text)}
        split_ixs.update(insert_ix_to_inserted_text.keys())
        split_ixs = sorted(split_ixs)

        segments = [self.text[ix1:ix2] for ix1, ix2 in zip(split_ixs[:-1], split_ixs[1:])]

        accum = []
        for seg, split_ix in zip(segments, split_ixs):
            accum.append("".join(insert_ix_to_inserted_text[split_ix]))
            accum.append(seg)
        return "".join(accum)

    def to_html(self):
        formatted = self.apply_formatting()

        if self.subtype is not None:
            formatted = self.subtype.format_html(formatted)

        return formatted

    @staticmethod
    def from_payload(payload: dict) -> 'NPFTextBlock':
        return NPFTextBlock(
            text=payload['text'],
            subtype=None if payload.get('subtype') is None else NPFSubtype(subtype=payload.get('subtype')),
            indent_level=payload.get('indent_level'),
            formatting=[NPFFormattingRange(**entry) for entry in payload.get('formatting', [])],
        )


TumblrContent = List[TumblrContentBlockBase]


class TumblrPostBase:
    def __init__(
        self,
        blog_name: str,
        content: TumblrContent,
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
        content: TumblrContent,
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
