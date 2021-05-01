import weakref
from typing import Union, List, NamedTuple

import numpy as np
import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoAttentionMixin, GPTNeoForCausalLM
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig


def prep_inputs(batch_texts, tokenizer, device='cpu'):
    feed_in = tokenizer(batch_texts, padding=True)

    for k in feed_in.keys():
        feed_in[k] = np.asarray(feed_in[k])

    input_ids_with_pads = feed_in['input_ids'].copy()
    input_ids_with_pads = torch.as_tensor(input_ids_with_pads).to(device)

    feed_in['input_ids'][feed_in['input_ids'] == tokenizer.pad_token_id] = 0

    for k in feed_in.keys():
        feed_in[k] = torch.as_tensor(feed_in[k]).to(device)

    input_ids = feed_in['input_ids']
    attention_mask = feed_in['attention_mask']
    return input_ids, attention_mask, input_ids_with_pads


def get_child_module_by_names(module, names):
    obj = module
    for getter in map(lambda name: lambda obj: getattr(obj, name), names):
        obj = getter(obj)
    return obj


def extract_activations(model, layer_names: list, prefixes: list = [], verbose=True):
    for attr in ['_extracted_activations', '_extracted_activation_handles']:
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
    def __init__(self,
                 base_model_config: GPTNeoConfig,
                 n_head: int,
                 attn_dropout=0.,
                 res_dropout=0.,
                 layer_norm_epsilon=1e-5,
                 ):
        super().__init__()

        max_positions = base_model_config.max_position_embeddings
        self.register_buffer(
            "bias",
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.uint8)).view(
                1, 1, max_positions, max_positions
            ),
        )
        self.register_buffer("masked_bias", torch.tensor(-1e9))

        self.n_head = n_head
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.res_dropout = nn.Dropout(res_dropout)

        self.embed_dim = base_model_config.hidden_size
        self.head_dim = self.embed_dim // self.n_head
        if self.head_dim * self.n_head != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by n_head (got `embed_dim`: {self.embed_dim} and `n_head`: {self.n_head})."
            )

        self.ln = nn.LayerNorm(self.embed_dim, eps=layer_norm_epsilon)

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states,
        input_ids_with_pads,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        hidden_states = self.ln(hidden_states)

        hidden_state_at_last_token = select_at_last_token(hidden_states, input_ids_with_pads).unsqueeze(-2)

        query = self.q_proj(hidden_state_at_last_token)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)

        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length].bool()

        attn_output, attn_weights = self._attn(
            query, key, value, causal_mask, self.masked_bias, self.attn_dropout, attention_mask, head_mask
        )

        attn_output = self._merge_heads(attn_output, self.n_head, self.head_dim)
        attn_output = self.out_proj(attn_output)
        attn_output = self.res_dropout(attn_output)

        attn_output = attn_output.squeeze(-2)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # a, (attentions)


class NostARHeadMLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 intermediate_size: int,
                 res_dropout: float = 0.
                 ):
        super().__init__()
        self.c_fc = nn.Linear(input_size, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, input_size)
        self.act = ACT2FN['gelu']
        self.dropout = nn.Dropout(res_dropout)

    def forward(self, hidden_states):
        hidden_states = self.c_fc(hidden_states)
        hidden_states = self.act(hidden_states)
        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# TODO: move to selector_estimator_neo.py
NostARHeadOptimizerParams = NamedTuple(
    'NostARHeadOptimizerParams',
    epochs=int,
    batch_size=int,
    base_lr=float,
    weight_decay=float,
    min_lr_frac=float,
    warmup_ratio=float,
)


NostARHeadArchitectureParams = NamedTuple(
    'NostARHeadArchitectureParams',
    layer_nums=List[int],
    n_head=Union[int, List[int]],
    mlp_ratio=Union[int, float],
    attn_dropout=float,
    res_dropout=float,
)


def validate_arch_params(params: NostARHeadArchitectureParams):
    if not isinstance(params.n_head, int):
        if len(params['n_head']) != len(params['layer_nums']):
            msg = "n_head and layer_nums must be equal length, got "
            msg += f"n_head={params['n_head']}, layer_nums={params['layer_nums']}"
            raise ValueError(msg)


GPT2TokenizerType = Union[GPT2Tokenizer, GPT2TokenizerFast]


class NostARHead(nn.Module):
    def __init__(
        self,
        base_model: GPTNeoForCausalLM,  # TODO: make compat with GPTNeoModel, etc?
        tokenizer: GPT2TokenizerType,
        params: NostARHeadArchitectureParams,
    ):
        validate_arch_params(params)

        super().__init__()

        self._base_model = weakref.ref(base_model)
        self._tokenizer = weakref.ref(tokenizer)
        self.params = params

        self._setup()

    @property
    def base_model(self) -> GPTNeoForCausalLM:
        return self._base_model()

    @property
    def tokenizer(self) -> GPT2TokenizerType:
        return self._tokenizer()

    @property
    def layer_nums(self):
        return self.params.layer_nums

    @property
    def n_head(self) -> List[int]:
        if isinstance(self.params.n_head, int):
            return [self.params.n_head for _ in self.layer_nums]
        return self.params.n_head

    def __str__(self):
        return f"NostARHead(params={repr(self.params)})"

    def _setup_attns(self):
        self.attns = nn.ModuleList([
            NostARHeadAttention(
                n_head=nh,
                base_model_config=self.base_model.config,
                attn_dropout=self.params.attn_dropout,
                res_dropout=self.params.res_dropout,
            )
            for nh in self.n_head
        ])

        self.layer_nums_to_attns = {
            lnum: attn
            for lnum, attn in zip(self.layer_nums, self.attns)
        }

    def _setup(self):
        extract_activations(
            self.base_model,
            layer_names=self.layer_nums,
            prefixes=['transformer', 'h']
        )

        self._setup_attns()

        mlp_input_size = len(self.layer_nums) * self.base_model.config.hidden_size
        mlp_intermediate_size = int(mlp_input_size * self.params.mlp_ratio)

        self.mlp = NostARHeadMLP(
            input_size=mlp_input_size,
            intermediate_size=mlp_intermediate_size
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
            self.base_model(input_ids=input_ids, attention_mask=attention_mask)

        attn_outs = [
            attn(self.base_model._extracted_activations[lnum], input_ids_with_pads)[0]
            for lnum, attn in self.layer_nums_to_attns.items()
        ]

        hidden_state = torch.cat(attn_outs, dim=-1)

        hidden_state = hidden_state + self.mlp(hidden_state)

        logits = self.logit_head(hidden_state)

        return logits
