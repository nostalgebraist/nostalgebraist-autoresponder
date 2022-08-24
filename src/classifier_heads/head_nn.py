import weakref
from typing import Union, List, NamedTuple, Optional
from functools import partial

import numpy as np
import torch
import torch.nn as nn

from transformer_utils.util.module_utils import get_child_module_by_names

from transformers.activations import ACT2FN
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig

from stable_library_code.transformers.gpt2.configuration_gpt2 import GPT2Config
from stable_library_code.transformers.gpt2.modeling_gpt2 import GPT2LMHeadModel
from stable_library_code.transformers.gpt_neo.modeling_gpt_neo import (
    GPTNeoAttentionMixin,
    GPTNeoForCausalLM,
)
from stable_library_code.transformers.gpt_neo.partial_forward import partial_forward as ref_partial_forward

from transformers.models.gpt_neo.modeling_gpt_neo import fixed_pos_embedding, apply_rotary_pos_emb

from transformer_utils.partial_forward import partial_forward, add_partial_forward_hooks

GPT2TokenizerType = Union[GPT2Tokenizer, GPT2TokenizerFast]
GPTConfigType = Union[GPT2Config, GPTNeoConfig]
GPTModelType = Union[GPT2LMHeadModel, GPTNeoForCausalLM]


def prep_inputs(batch_texts, tokenizer, max_length=2048, pad_to_mult=256, device="cpu"):
    batch_texts_ = []
    for bt in batch_texts:
        to_append = bt
        if not to_append.endswith(tokenizer.eos_token):
            to_append = to_append + tokenizer.eos_token
        batch_texts_.append(to_append)
    batch_texts = batch_texts_

    feed_in = tokenizer(
        batch_texts, padding=True, truncation=True, max_length=max_length
    )

    for k in feed_in.keys():
        feed_in[k] = np.asarray(feed_in[k])

    if pad_to_mult:
        true_len = len(feed_in['input_ids'][0])
        pad_to_len = pad_to_mult * (true_len // pad_to_mult + 1)

        pad_to_len = min(pad_to_len, max_length)

        if true_len < pad_to_mult:
            pad_to_len = true_len

        feed_in = tokenizer(
            batch_texts, padding='max_length', truncation=True, max_length=pad_to_len
        )

        for k in feed_in.keys():
          feed_in[k] = np.asarray(feed_in[k])

    input_ids = feed_in["input_ids"]
    input_ids_with_pads = feed_in["input_ids"].copy()
    attention_mask = feed_in["attention_mask"]

    input_ids[input_ids == tokenizer.pad_token_id] = 0

    input_ids = torch.as_tensor(input_ids).pin_memory().to(device)
    input_ids_with_pads = torch.as_tensor(input_ids_with_pads).pin_memory().to(device)
    attention_mask = torch.as_tensor(attention_mask).pin_memory().to(device)

    return input_ids, attention_mask, input_ids_with_pads


def extract_activations(model, layer_names: list, prefixes: list = [], verbose=True):
    for attr in ["_extracted_activations", "_extracted_activation_handles"]:
        if not hasattr(model, attr):
            setattr(model, attr, {})

    def _get_layer(name):
        return get_child_module_by_names(model, prefixes + [str(name)])

    def _make_record_output_hook(name):
        model._extracted_activations[name] = None

        def _record_output_hook(module, input, output) -> None:
            model._extracted_activations[name] = output[0]

        return _record_output_hook

    def _hook_already_there(name):
        handle = model._extracted_activation_handles.get(name)
        if not handle:
            return False
        layer = _get_layer(name)
        return handle.id in layer._forward_hooks

    for name in layer_names:
        if _hook_already_there(name):
            if verbose:
                print(f"skipping layer {name}, hook already exists")
            continue
        layer = _get_layer(name)
        handle = layer.register_forward_hook(_make_record_output_hook(name))
        model._extracted_activation_handles[name] = handle


def select_at_last_token(select_from, tokens, pad_token_id=50257):
    mask_isnt_pad = tokens != pad_token_id
    select_ixs = mask_isnt_pad.cumsum(dim=1).argmax(dim=1)
    iselect = torch.index_select(select_from, dim=1, index=select_ixs)
    final = torch.diagonal(iselect).T
    return final


class NostARHeadAttention(nn.Module, GPTNeoAttentionMixin):
    """Adapted from transformers library's `GPTNeoSelfAttention`"""

    def __init__(
        self,
        base_model_config: GPTConfigType,
        n_head: int,
        attn_dropout=0.0,
        res_dropout=0.0,
        layer_norm_epsilon=1e-5,
        proj_ratio: float = 1.,
        use_proj=True,
        pool_to_vector=True,
        qkv_dim=None,
        rotary=False,
        rotary_dim=64,
    ):
        super().__init__()

        max_positions = base_model_config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.uint8)
            ).view(1, 1, max_positions, max_positions),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.pool_to_vector = pool_to_vector
        self.qkv_dim = None or self.embed_dim

        self.n_head = n_head
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_dropout = nn.Dropout(res_dropout)

        self.embed_dim = base_model_config.hidden_size
        self.head_dim = self.qkv_dim // self.n_head
        self.proj_dim = int(proj_ratio * self.qkv_dim)
        if self.head_dim * self.n_head != self.qkv_dim:
            raise ValueError(
                f"embed_dim must be divisible by n_head (got `embed_dim`: {self.embed_dim} and `n_head`: {self.n_head})."
            )

        self.ln = nn.LayerNorm(self.embed_dim, eps=layer_norm_epsilon)

        self.k_proj = nn.Linear(self.embed_dim, self.qkv_dim, bias=True)  # vs bias=False in GPTNeo -nost
        self.v_proj = nn.Linear(self.embed_dim, self.qkv_dim, bias=True)  # vs bias=False in GPTNeo -nost
        self.q_proj = nn.Linear(self.embed_dim, self.qkv_dim, bias=True)  # vs bias=False in GPTNeo -nost

        self.use_proj = use_proj or (self.embed_dim != self.qkv_dim)

        if self.use_proj:
            self.out_proj = nn.Linear(self.qkv_dim, self.proj_dim, bias=True)

        self.rotary = rotary
        self.rotary_dim = rotary_dim
        if self.rotary:
            sin, cos = fixed_pos_embedding(dim=self.rotary_dim, seq_len=max_positions)
            self.register_buffer("sin", sin)
            self.register_buffer("cos", cos)

    def classic_init(self, gain=1.):
        with torch.no_grad():
            qkv_weight = torch.empty(self.embed_dim, 3 * self.qkv_dim, requires_grad=False)
            torch.nn.init.orthogonal_(qkv_weight, gain=gain)

            q_weight, k_weight, v_weight = torch.split(qkv_weight, self.embed_dim, dim=-1)

            self.q_proj.weight.copy_(q_weight)
            self.k_proj.weight.copy_(k_weight)
            self.v_proj.weight.copy_(v_weight)

            print(f"classic_init: initialized qkv from qkv_weight with shape {qkv_weight.shape}")
            del qkv_weight

    def forward(
        self,
        hidden_states,
        input_ids_with_pads,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        hidden_states = hidden_states.to(dtype=self.ln.weight.dtype,
                                         device=self.ln.weight.device)
        hidden_states = self.ln(hidden_states)

        if self.pool_to_vector:
            hidden_state_at_last_token = select_at_last_token(
                hidden_states, input_ids_with_pads.to(self.ln.weight.device)
            ).unsqueeze(-2)

            query = self.q_proj(hidden_state_at_last_token)
        else:
            query = self.q_proj(hidden_states)

        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)

        if self.rotary:
            seq_len = key.shape[1]
            offset = 0
            if self.rotary_dim < self.head_dim:
                k_rot = key[:, :, :, :self.rotary_dim]
                k_pass = key[:, :, :, self.rotary_dim:]

                q_rot = query[:, :, :, :self.rotary_dim]
                q_pass = query[:, :, :, self.rotary_dim:]

                k_rot = apply_rotary_pos_emb(k_rot, (self.sin, self.cos), offset=offset).to(k_rot.dtype)
                q_rot = apply_rotary_pos_emb(q_rot, (self.sin, self.cos), offset=offset).to(q_rot.dtype)

                key = torch.cat([k_rot, k_pass], dim=-1)
                query = torch.cat([q_rot, q_pass], dim=-1)
            else:
                key = apply_rotary_pos_emb(key, (self.sin, self.cos), offset=offset).to(key.dtype)
                query = apply_rotary_pos_emb(query, (self.sin, self.cos), offset=offset).to(query.dtype)
            key = key.permute(0, 2, 1, 3)
            query = query.permute(0, 2, 1, 3)

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ].bool()

        attn_output, attn_weights = self._attn(
            query,
            key,
            value,
            causal_mask,
            self.masked_bias,
            self.attn_dropout,
            attention_mask,
            head_mask,
        )

        attn_output = self._merge_heads(attn_output, self.n_head, self.head_dim)
        if self.use_proj:
            attn_output = self.out_proj(attn_output)
        attn_output = self.res_dropout(attn_output)

        if self.pool_to_vector:
            attn_output = attn_output.squeeze(-2)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, (attentions)


class NostARHeadMLP(nn.Module):
    def __init__(
        self, input_size: int, intermediate_size: int, res_dropout: float = 0.0
    ):
        super().__init__()
        self.c_fc = nn.Linear(input_size, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, input_size)
        self.act = ACT2FN["gelu"]
        self.dropout = nn.Dropout(res_dropout)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class NostARHeadBlock(nn.Module):
    def __init__(self, attn_params, mlp_params):
        super().__init__()
        self.attn = NostARHeadAttention(pool_to_vector=False, **attn_params)
        self.mlp = NostARHeadMLP(**mlp_params)

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.attn(hidden_states)
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states


NostARHeadArchitectureParams = NamedTuple(
    "NostARHeadArchitectureParams",
    layer_nums=List[int],
    n_head=Union[int, List[int]],
    mlp_ratio=Union[int, float],
    attn_dropout=float,
    res_dropout=float,
    init_gain=float,
    init_gain_logit_head=float,
    classic_behavior_attn_init=bool,
    proj_ratio=Union[int, float],
    use_final_mlp=bool,
    n_blocks=int,
    mlp_ratio_blocks=Union[int, float],
    n_head_blocks=int,
    qkv_dim_blocks=Optional[int],
    qkv_dim_final=Optional[int],
    rotary_blocks=bool,
    rotary_dim_blocks=int,
)


def validate_arch_params(params: NostARHeadArchitectureParams):
    if not isinstance(params.n_head, int):
        if len(params["n_head"]) != len(params["layer_nums"]):
            msg = "n_head and layer_nums must be equal length, got "
            msg += f"n_head={params['n_head']}, layer_nums={params['layer_nums']}"
            raise ValueError(msg)

    if params.n_blocks > 0 and len(params["layer_nums"]) > 1:
        raise ValueError('blocks only supported with one base layer')


class NostARHead(nn.Module):
    def __init__(
        self,
        base_model: GPTModelType,  # TODO: make compat with GPTNeoModel, etc?
        tokenizer: GPT2TokenizerType,
        params: NostARHeadArchitectureParams,
        partial_forward_type="tfu",  # debug
        initialize_weights=True,
        params_extras=None
    ):
        validate_arch_params(params)

        super().__init__()

        if partial_forward_type != 'tfu':
            raise ValueError(partial_forward_type)

        self._base_model = weakref.ref(base_model)
        self._tokenizer = weakref.ref(tokenizer)
        self.params = params
        self.partial_forward_type = partial_forward_type
        self.params_extras = {} if params_extras is None else params_extras

        self._setup()
        if initialize_weights:
            self.init_weights()

    @property
    def use_proj(self):
        return self.params_extras.get("use_proj", True)

    @property
    def base_model(self) -> GPTModelType:
        return self._base_model()

    @property
    def tokenizer(self) -> GPT2TokenizerType:
        return self._tokenizer()

    @property
    def layer_nums(self):
        return self.params.layer_nums

    @property
    def n_blocks(self):
        return self.params.n_blocks

    @property
    def layer_names(self):
        return [f'h.{i}' for i in self.layer_nums]

    @property
    def n_head(self) -> List[int]:
        if isinstance(self.params.n_head, int):
            return [self.params.n_head for _ in self.layer_nums]
        return self.params.n_head

    def __str__(self):
        return f"NostARHead(params={repr(self.params)})"

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights."""
        if module is self.logit_head:
            torch.nn.init.orthogonal_(module.weight, gain=self.params.init_gain_logit_head)
            print(
                f"initialized logit_head with gain {self.params.init_gain_logit_head:.2f}"
            )
        elif any([module is m for m in self.attns]) and self.params.classic_behavior_attn_init:
            print(f"calling classic init for {repr(module)} with gain {self.params.init_gain:.2f}")
            module.classic_init(gain=self.params.init_gain)
        elif isinstance(module, (nn.Linear,)):
            print(
                f"initialized {repr(module)} with gain {self.params.init_gain:.2f}"
            )
            torch.nn.init.orthogonal_(module.weight, gain=self.params.init_gain)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _setup_blocks(self):
        attn_params = dict(
            n_head=self.params.n_head_blocks,
            qkv_dim=self.params.qkv_dim_blocks,
            attn_dropout=self.params.attn_dropout,
            res_dropout=self.params.res_dropout,
            proj_ratio=self.params.proj_ratio,
            use_proj=True,
            rotary=self.params.rotary_blocks,
            rotary_dim=self.params.rotary_dim_blocks,
        )
        mlp_params = dict(
            input_size=self.base_model.config.hidden_size,
            intermediate_size=int(self.base_model.config.hidden_size * self.params.mlp_ratio_blocks)
            res_dropout=self.params.res_dropout,
        )
        self.blocks = nn.ModuleList(
            [
                NostARHeadBlock(
                    attn_params=attn_params,
                    mlp_params=mlp_params,
                )
                for _ in range(self.n_blocks)
            ]
        )

        self.layer_names_to_blocks = {self.layer_names[0]: self.blocks}

    def _setup_attns(self):
        self.attns = nn.ModuleList(
            [
                NostARHeadAttention(
                    n_head=nh,
                    base_model_config=self.base_model.config,
                    attn_dropout=self.params.attn_dropout,
                    res_dropout=self.params.res_dropout,
                    proj_ratio=self.params.proj_ratio,
                    use_proj=self.use_proj
                )
                for nh in self.n_head
            ]
        )

        self.layer_names_to_attns = {
            lnum: attn for lnum, attn in zip(self.layer_names, self.attns)
        }
        self.layer_nums_to_attns = {
            lnum: attn for lnum, attn in zip(self.layer_nums, self.attns)
        }

    def _setup(self):
        for param in self.base_model.parameters():
            param.requires_grad = False

        add_partial_forward_hooks(self.base_model.transformer, output_names=self.layer_names)

        self._setup_blocks()

        self._setup_attns()

        mlp_input_size = len(self.layer_nums) * int(self.params.proj_ratio * self.base_model.config.hidden_size)
        mlp_intermediate_size = int(mlp_input_size * self.params.mlp_ratio)

        if self.params.use_final_mlp:
            self.mlp = NostARHeadMLP(
                input_size=mlp_input_size,
                intermediate_size=mlp_intermediate_size,
                res_dropout=self.params.res_dropout,
            )

        self.logit_head = nn.Linear(mlp_input_size, 2)

    def forward(
        self,
        input_ids,
        input_ids_with_pads,
        attention_mask,
        head_mask=None,
        output_attentions=False,
    ):
        with torch.no_grad():
            extracted_activations = partial_forward(
                model=self.base_model.transformer,
                output_names=self.layer_names,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False
            )

        for block in self.blocks:
            name = self.layer_names[0]
            extracted_activations[name] = block(extracted_activations[name])

        attn_outs = [
            attn(extracted_activations[name], input_ids_with_pads)[0]
            for name, attn in self.layer_names_to_attns.items()
        ]

        hidden_state = torch.cat(attn_outs, dim=-1)

        if self.params.use_final_mlp:
            hidden_state = hidden_state + self.mlp(hidden_state)

        logits = self.logit_head(hidden_state)

        return logits
