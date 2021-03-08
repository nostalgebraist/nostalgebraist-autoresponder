"""work in progress"""
from typing import List, Optional
from collections import defaultdict
from itertools import zip_longest
from copy import deepcopy


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
            result["start_insert"] = f'<a href="{self.url}">'
            result["end_insert"] = f"</a>"
        elif self.type == "mention":
            blog_url = self.blog.get('url')
            result["start_insert"] = f'<a class="tumblelog" href="{blog_url}">'
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
        text_or_break = text if len(text) > 0 else "<br>"
        if self.subtype == "heading1":
            return f"<p><h2>{text_or_break}</h2></p>"  # TODO: verify
        elif self.subtype == "heading2":
            return f"<p><h2>{text_or_break}</h2></p>"  # TODO: verify
        elif self.subtype == "ordered-list-item":
            return f"<li>{text}</li>"
        elif self.subtype == "unordered-list-item":
            return f"<li>{text}</li>"
        else:
            return f"<p>{text_or_break}</p>"


class NPFBlock(TumblrContentBlockBase):
    def from_payload(payload: dict) -> 'NPFBlock':
        if payload.get('type') == 'text':
            return NPFTextBlock.from_payload(payload)
        elif payload.get('type') == 'image':
            return NPFImageBlock.from_payload(payload)
        else:
            raise ValueError(payload.get('type'))


class NPFTextBlock(NPFBlock):
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


class NPFNonTextBlockMixin:
    @property
    def subtype_name(self):
        return 'no_subtype'

    @property
    def indent_level(self):
        return 0


class NPFImageBlock(NPFBlock, NPFNonTextBlockMixin):
    def __init__(self,
                 media: List[dict],
                 alt_text: Optional[str] = None):
        self._media = media
        self._alt_text = alt_text

    @property
    def media(self):
        return self._media

    @property
    def alt_text(self):
        return self._alt_text

    @staticmethod
    def from_payload(payload: dict) -> 'NPFImageBlock':
        return NPFImageBlock(media=payload['media'],
                             alt_text=payload.get('alt_text'))

    def to_html(self) -> str:
        # TODO: implement
        return "<p><b>Image blocks not yet implemented</b></p>"


class NPFLayout:
    @property
    def layout_type(self):
        return self._layout_type

    def from_payload(payload: dict) -> 'NPFLayout':
        if payload.get('type') == 'rows':
            return NPFLayoutRows.from_payload(payload)
        elif payload.get('type') == 'ask':
            return NPFLayoutAsk.from_payload(payload)
        else:
            raise ValueError(payload.get('type'))


class NPFLayoutMode:
    def __init__(self, mode_type: str):
        self._mode_type = mode_type

    @property
    def mode_type(self):
        return self._mode_type

    @staticmethod
    def from_payload(payload: dict) -> "NPFLayoutMode":
        return NPFLayoutMode(mode_type=payload['type'])


class NPFRow:
    def __init__(self,
                 blocks: List[int],
                 mode: Optional[NPFLayoutMode] = None,
                 ):
        self._blocks = blocks
        self._mode = mode

    @property
    def blocks(self):
        return self._blocks

    @property
    def mode(self):
        return self._mode

    @staticmethod
    def from_payload(payload: dict) -> "NPFRow":
        return NPFRow(blocks=payload['blocks'], mode=payload.get('mode'))


class NPFLayoutRows(NPFLayout):
    def __init__(self,
                 rows: List[NPFRow],
                 truncate_after: Optional[int] = None,
                 ):
        self._rows = rows
        self._truncate_after = truncate_after
        self._layout_type = "rows"

    @property
    def rows(self):
        return self._rows

    @property
    def truncate_after(self):
        return self._truncate_after

    @staticmethod
    def from_payload(payload: dict) -> "NPFLayoutRows":
        rows = [entry['blocks'] for entry in payload['display']]
        return NPFLayoutRows(rows=rows,
                             truncate_after=payload.get('truncate_after'))


class NPFLayoutAsk(NPFLayout):
    def __init__(self,
                 blocks: List[int],
                 attribution: Optional[dict] = None,
                 ):
        self._blocks = blocks
        self._attribution = attribution
        self._layout_type = "ask"

    @property
    def blocks(self):
        return self._blocks

    @property
    def attribution(self):
        return self._attribution

    @property
    def asking_name(self):
        if self.attribution is None:
            return 'Anonymous'
        return self.attribution['url'].partition('.tumblr.com')[0].partition('//')[2]

    @staticmethod
    def from_payload(payload: dict) -> "NPFLayoutAsk":
        return NPFLayoutAsk(blocks=payload['blocks'],
                            attribution=payload.get('attribution'))


class NPFBlockAnnotated(NPFBlock):
    def __init__(self,
                 base_block: NPFBlock,
                 is_ask_block: bool = False,
                 ask_layout: Optional[NPFLayoutAsk] = None):
        self.base_block = base_block

        self.prefix = ""
        self.suffix = ""
        self.indent_delta = None
        self.is_ask_block = is_ask_block
        self.ask_layout = ask_layout

    def reset_annotations(self):
        new = NPFBlockAnnotated(base_block=self.base_block)
        for attr, value in new.__dict__.items():
            setattr(self, attr, value)

    def as_ask_block(self, ask_layout: NPFLayoutAsk) -> 'NPFBlockAnnotated':
        new = deepcopy(self)
        new.is_ask_block = True
        new.ask_layout = ask_layout
        return new

    @property
    def asking_name(self):
        if not self.is_ask_block:
            return None
        if self.ask_layout is None:
            return None
        return self.ask_layout.asking_name

    def to_html(self) -> str:
        inside = self.base_block.to_html()
        return self.prefix + inside + self.suffix


class TumblrContentBase:
    def __init__(self, content: List[TumblrContentBlockBase]):
        self.content = content

    def to_html(self) -> str:
        raise NotImplementedError


class NPFContent(TumblrContentBase):
    def __init__(self,
                 blocks: List[NPFBlock],
                 layout: List[NPFLayout],
                 blog_name: str):
        self.raw_blocks = [
            block if isinstance(block, NPFBlockAnnotated) else NPFBlockAnnotated(block)
            for block in blocks
        ]
        self.layout = layout
        self.blog_name = blog_name

        self.blocks = self._make_blocks()

    def _make_blocks(self) -> List[NPFBlockAnnotated]:
        if len(self.layout) == 0:
            return self.raw_blocks
        else:
            # TODO: figure out how to handle truncate_after
            ordered_block_ixs = []
            ask_ixs = set()
            ask_ixs_to_layouts = {}
            for layout_entry in self.layout:
                if layout_entry.layout_type == "rows":
                    for row_ixs in layout_entry.rows:
                        # note: this doesn't properly handle multi-column rows
                        # TODO: handle multi-column rows
                        ordered_block_ixs.extend(row_ixs)
                elif layout_entry.layout_type == "ask":
                    ordered_block_ixs.extend(layout_entry.blocks)
                    ask_ixs.update(layout_entry.blocks)
                    ask_ixs_to_layouts.update(
                        {ix: layout_entry
                         for ix in layout_entry.blocks}
                    )
            if all([layout_entry.layout_type == "ask"
                    for layout_entry in self.layout]):
                extras = [ix for ix in range(len(self.raw_blocks))
                          if ix not in ask_ixs]
                ordered_block_ixs.extend(extras)
            return [
                self.raw_blocks[ix].as_ask_block(
                    ask_layout=ask_ixs_to_layouts[ix]
                )
                if ix in ask_ixs
                else self.raw_blocks[ix]
                for ix in ordered_block_ixs
            ]

    @property
    def ask_blocks(self) -> List[NPFBlockAnnotated]:
        return [bl for bl in self.blocks if bl.is_ask_block]

    @staticmethod
    def from_payload(payload: dict) -> 'NPFContent':
        blocks = [NPFBlock.from_payload(bl) for bl in payload['content']]
        layout = [NPFLayout.from_payload(lay) for lay in payload['layout']]
        blog_name = payload['blog']['name']
        return NPFContent(blocks=blocks, layout=layout, blog_name=blog_name)

    def _reset_annotations(self):
        for bl in self.blocks:
            bl.reset_annotations()

    def _assign_indents(self):
        #  i think the below comment is out of date and this works now?  TODO: verify
        #
        #  this doesn't quite work
        #  stepping out a level should close the tag *currently* top of stack
        #  not the tag *opened by the block that steps out*

        indenting_subtypes = {"indented", "ordered-list-item", "unordered-list-item"}
        prev_subtypes = [None] + [
            block.base_block.subtype_name for block in self.blocks
        ]

        cur_level = 0

        for block, prev_subtype in zip(self.blocks, prev_subtypes):
            indent_delta = block.base_block.indent_level - cur_level
            this_indents = block.base_block.subtype_name in indenting_subtypes
            prev_indents = prev_subtype in indenting_subtypes

            if indent_delta != 0:
                block.indent_delta = indent_delta
            elif this_indents and not prev_indents:
                block.indent_delta = 1
            elif prev_indents and not this_indents:
                block.indent_delta = -1
            else:
                block.indent_delta = 0

            cur_level = block.base_block.indent_level

    def _assign_nonlocal_tags(self):
        subtype_and_sign_to_tag = {
            ("indented", True): "<blockquote>",
            ("indented", False): "</blockquote>",
            ("ordered-list-item", True): "<ol>",
            ("ordered-list-item", False): "</ol>",
            ("unordered-list-item", True): "<ul>",
            ("unordered-list-item", False): "</ul>",
        }

        stack = []

        for block in self.blocks:
            if block.indent_delta == 0:
                continue

            sign = block.indent_delta > 0
            abs = block.indent_delta if sign else -1 * block.indent_delta

            if sign:
                subtype_key = block.base_block.subtype_name
                stack.append(subtype_key)
            else:
                subtype_key = stack.pop()

            key = (subtype_key, sign)
            if key not in subtype_and_sign_to_tag:
                raise ValueError(key)  # TODO: improve

            tag = subtype_and_sign_to_tag[key]
            tags = abs * tag

            block.prefix = tags + block.prefix

        closers = []
        while len(stack) > 0:
            subtype_key = stack.pop()
            key = (subtype_key, False)
            closers.append(subtype_and_sign_to_tag[key])

        self.blocks[-1].suffix += "".join(closers)

    def _assign_ask_tags(self):
        if len(self.ask_blocks) > 0:
            first_ask_block = self.ask_blocks[0]
            last_ask_block = self.ask_blocks[-1]

            # TODO: make this work if there's already a prefix/suffix on the blocks
            # TODO: make the formatting real
            first_ask_block.prefix = f"<p><b>{first_ask_block.asking_name} asked:</b></p>" + first_ask_block.prefix
            last_ask_block.suffix += "<p><b>Ask ends</b></p>"

        # blogname / answering name prefix
        # TODO: make this work if there's already a prefix/suffix on the blocks
        # TODO: make the formatting real
        first_nonask_block = self.blocks[len(self.ask_blocks)]
        print(f"prefix before: {repr(first_nonask_block.prefix)}")
        first_nonask_block.prefix = f"<p><b>{self.blog_name}:</b></p>" + first_nonask_block.prefix
        print(f"prefix after: {repr(first_nonask_block.prefix)}")

    def to_html(self):
        self._reset_annotations()
        self._assign_indents()
        self._assign_nonlocal_tags()
        self._assign_ask_tags()

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

    def to_html(self) -> str:
        # TODO: tags?
        return self._content.to_html()


class TumblrThread:
    def __init__(self, posts: List[TumblrPost]):
        self._posts = posts

    @property
    def posts(self):
        return self._posts

    def to_html(self) -> str:
        result = ""

        for post in self.posts[:-1]:
            result = f"<blockquote>{result}{post.to_html()}</blockquote>"

        result += self.posts[-1].to_html()
        return result
