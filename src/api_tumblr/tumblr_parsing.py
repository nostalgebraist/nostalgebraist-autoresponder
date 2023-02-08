import html as html_lib
from typing import List, Optional, Tuple
from collections import defaultdict
from itertools import zip_longest
from copy import deepcopy


def _get_blogname_from_payload(post_payload):
    """retrieves payload --> broken_blog_name, or payload --> blog --> name"""
    if 'broken_blog_name' in post_payload:
        return post_payload['broken_blog_name']
    return post_payload['blog']['name']


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
        elif len(text) == 0:
            return ""
        else:
            return f"<p>{text_or_break}</p>"


class NPFBlock(TumblrContentBlockBase):
    def from_payload(payload: dict) -> 'NPFBlock':
        if payload.get('type') == 'text':
            return NPFTextBlock.from_payload(payload)
        elif payload.get('type') == 'image':
            return NPFImageBlock.from_payload(payload)
        elif payload.get('type') == 'poll':
            return NPFPollBlock.from_payload(payload)
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
            accum.append(html_lib.escape(self.text[ix1:ix2]))

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

    @property
    def original_dimensions(self) -> Optional[Tuple[int, int]]:
        for entry in self.media:
            if entry.get('has_original_dimensions'):
                return (entry['width'], entry['height'])

    @staticmethod
    def from_payload(payload: dict) -> 'NPFImageBlock':
        return NPFImageBlock(media=payload['media'],
                             alt_text=payload.get('alt_text'))

    def _pick_one_size(self, target_width: int = 640) -> dict:
        by_width_descending = sorted(self.media, key=lambda entry: entry['width'], reverse=True)
        for entry in by_width_descending:
            if entry['width'] <= target_width:
                return entry
        return by_width_descending[-1]

    def to_html(self, target_width: int = 640) -> str:
        selected_size = self._pick_one_size(target_width)

        original_dimensions_attrs_str = ""
        if self.original_dimensions is not None:
            orig_w, orig_h = self.original_dimensions
            original_dimensions_attrs_str = f" data-orig-height=\"{orig_h}\" data-orig-width=\"{orig_w}\""

        img_tag = f"<img src=\"{selected_size['url']}\"{original_dimensions_attrs_str}/>"

        figure_tag = f"<figure class=\"tmblr-full\"{original_dimensions_attrs_str}>{img_tag}</figure>"

        return figure_tag


class NPFPollBlock(NPFBlock, NPFNonTextBlockMixin):
    def __init__(
        self,
        client_id,
        question,
        answers,
        settings,
        created_at,
        timestamp,
    ):
        self.client_id = client_id
        self.question = question
        self.answers = answers
        self.settings = settings
        self.created_at = created_at
        self.timestamp = timestamp


    @staticmethod
    def from_payload(payload: dict) -> 'NPFPollBlock':
        return NPFPollBlock(**{k: payload[k] for k in payload if k not in {'type'}})

    def to_html(self):
        """
        Polls don't appear at all in official legacy HTML. The CSS classes here are made-up,
        and exist mainly to make polls distinguishable from other objects in the DOM tree
        we'll produce.
        """
        question_segment = f'<p class="poll-question">{self.question}</p>'

        answers_segment = '<ol class="poll-answers">'
        for answer in self.answers:
            answers_segment += f'<li class="poll-answer">{answer["answer_text"]}</li>'
        answers_segment += '</ol>'

        return question_segment + answers_segment


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
                 blog_name: str,
                 id: Optional[int] = None,
                 genesis_post_id: Optional[int] = None,
                 post_url: Optional[str] = None):
        self.raw_blocks = [
            block if isinstance(block, NPFBlockAnnotated) else NPFBlockAnnotated(block)
            for block in blocks
        ]
        self.layout = layout
        self.blog_name = blog_name
        self.id = id
        self.genesis_post_id = genesis_post_id
        self._post_url = post_url

        self.blocks = self._make_blocks()

    @property
    def post_url(self) -> str:
        if self._post_url is None and self.id is not None:
            # N.B. this doesn't have the "slug", while the API's post_url does
            return f"https://{self.blog_name}.tumblr.com/post/{self.id}/"
        return self._post_url

    @property
    def legacy_prefix_link(self):
        return f"<p><a class=\"tumblr_blog\" href=\"{self.post_url}\">{self.blog_name}</a>:</p>"

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

                        # note: deduplication here is needed b/c of april 2021 tumblr npf ask bug
                        deduped_ixs = [ix for ix in row_ixs if ix not in ordered_block_ixs]
                        ordered_block_ixs.extend(deduped_ixs)
                elif layout_entry.layout_type == "ask":
                    # note: deduplication here is needed b/c of april 2021 tumblr npf ask bug
                    deduped_ixs = [ix for ix in layout_entry.blocks if ix not in ordered_block_ixs]
                    ordered_block_ixs.extend(deduped_ixs)
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

    @property
    def ask_layout(self) -> Optional[NPFLayoutAsk]:
        ask_layouts = [lay for lay in self.layout if lay.layout_type == "ask"]
        if len(ask_layouts) > 0:
            return ask_layouts[0]

    @property
    def has_ask(self) -> bool:
        return len(self.ask_blocks) > 0

    @staticmethod
    def from_payload(payload: dict, raise_on_unimplemented: bool = False) -> 'NPFContent':
        blocks = []
        for bl in payload['content']:
            try:
                blocks.append(NPFBlock.from_payload(bl))
            except ValueError as e:
                if raise_on_unimplemented:
                    raise e
                # generic default/fake filler block
                blocks.append(NPFTextBlock(""))

        layout = []
        for lay in payload['layout']:
            try:
                layout.append(NPFLayout.from_payload(lay))
            except ValueError as e:
                if raise_on_unimplemented:
                    raise e

        blog_name = _get_blogname_from_payload(payload)

        if 'id' in payload:
            id = payload['id']
        elif 'post' in payload:
            # trail format
            id = payload['post']['id']
        else:
            # broken trail item format
            id = None
        id = int(id) if id is not None else None

        genesis_post_id = payload.get('genesis_post_id')
        genesis_post_id = int(genesis_post_id) if genesis_post_id is not None else None

        post_url = payload.get('post_url')
        return NPFContent(blocks=blocks, layout=layout, blog_name=blog_name, id=id, genesis_post_id=genesis_post_id, post_url=post_url)

    def _reset_annotations(self):
        for bl in self.blocks:
            bl.reset_annotations()
        self.blocks = self._make_blocks()

    def _assign_indents(self):
        indenting_subtypes = {"indented", "ordered-list-item", "unordered-list-item"}
        prev_subtypes = [None] + [
            block.base_block.subtype_name for block in self.blocks
        ]

        cur_level = 0

        for block in self.blocks:
            this_indents = block.base_block.subtype_name in indenting_subtypes
            full_indent_level = this_indents + block.base_block.indent_level

            indent_delta = full_indent_level - cur_level

            block.indent_delta = indent_delta
            cur_level = full_indent_level

    def _assign_nonlocal_tags(self):
        """
        TODO:

        make this wrap multiple "inside" blocks in one <li></li> when needed
        as in the "nested in two blockquotes and a list" example in the docs
        """
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

            for _ in range(abs):
                if sign:
                    subtype_key = block.base_block.subtype_name
                    stack.append(subtype_key)
                else:
                    subtype_key = stack.pop()

                key = (subtype_key, sign)
                if key not in subtype_and_sign_to_tag:
                    # tumblr appears to use non-indenting subtypes with indent_level to indicate indentation
                    # this suddenly appeared around 9/19/21
                    subtype_key = "indented"
                    key = (subtype_key, sign)

                tag = subtype_and_sign_to_tag[key]

                block.prefix = tag + block.prefix

        closers = []
        while len(stack) > 0:
            subtype_key = stack.pop()
            key = (subtype_key, False)
            if key not in subtype_and_sign_to_tag:
                subtype_key = "indented"
                key = (subtype_key, sign)
            closers.append(subtype_and_sign_to_tag[key])

        try:
            self.blocks[-1].suffix += "".join(closers)
        except IndexError:
            # if there are 0 blocks
            pass

    def to_html(self):
        self._reset_annotations()
        self._assign_indents()
        self._assign_nonlocal_tags()

        return "".join([block.to_html() for block in self.blocks[len(self.ask_blocks):]])

    @property
    def ask_content(self) -> Optional['NPFAsk']:
        if self.has_ask:
            return NPFAsk.from_parent_content(self)


class NPFAsk(NPFContent):
    def __init__(self,
                 blocks: List[NPFBlock],
                 ask_layout: NPFLayout):
        super().__init__(
            blocks=blocks,
            layout=[],
            blog_name=ask_layout.asking_name,
        )

    @property
    def asking_name(self) -> str:
        return self.blog_name

    @staticmethod
    def from_parent_content(parent_content: NPFContent) -> Optional['NPFAsk']:
        if parent_content.has_ask:
            return NPFAsk(
                blocks=deepcopy(parent_content.ask_blocks),
                ask_layout=deepcopy(parent_content.ask_layout),
            )


class TumblrPostBase:
    def __init__(
        self,
        blog_name: str,
        id: int,
        content: TumblrContentBase,
        genesis_post_id: Optional[int] = None
    ):
        self._blog_name = blog_name
        self._content = content
        self._id = id
        self._genesis_post_id = genesis_post_id

    @property
    def blog_name(self):
        return self._blog_name

    @property
    def id(self):
        return self._id

    @property
    def content(self):
        return self._content

    @property
    def genesis_post_id(self):
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

    @property
    def id(self):
        return self._content.id

    @property
    def genesis_post_id(self):
        return self._content.genesis_post_id

    def to_html(self) -> str:
        return self._content.to_html()


class TumblrThread:
    def __init__(self, posts: List[TumblrPost], timestamp: int):
        self._posts = posts
        self._timestamp = timestamp

    @property
    def posts(self):
        return self._posts

    @property
    def timestamp(self):
        return self._timestamp

    @staticmethod
    def from_payload(payload: dict) -> 'TumblrThread':
        post_payloads = payload.get('trail', []) + [payload]
        posts = [
            TumblrPost(
                blog_name=_get_blogname_from_payload(post_payload),
                content=NPFContent.from_payload(post_payload),
                tags=post_payload.get('tags', [])
            )
            for post_payload in post_payloads
        ]

        timestamp = payload['timestamp']

        return TumblrThread(posts, timestamp)

    @staticmethod
    def _format_post_as_quoting_previous(post: TumblrPost, prev: TumblrPost, quoted: str) -> str:
        return f"{prev.content.legacy_prefix_link}<blockquote>{quoted}</blockquote>{post.to_html()}"

    def to_html(self) -> str:
        result = ""

        post_part = self.posts[0].to_html()
        result += post_part

        for prev, post in zip(self.posts[:-1], self.posts[1:]):
            result = TumblrThread._format_post_as_quoting_previous(post, prev, result)

        return result

    @property
    def ask_content(self) -> Optional[NPFAsk]:
        op_content = self.posts[0].content
        if op_content.has_ask:
            return op_content.ask_content
