import weakref
from typing import Union, List, NamedTuple

import numpy as np
import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from transformers.models.gpt2.tokenization_gpt2 import GPT2Tokenizer
from transformers.models.gpt_neo.configuration_gpt_neo import GPTNeoConfig
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

from stable_library_code.transformers.gpt2.modeling_gpt2 import GPT2LMHeadModel
from stable_library_code.transformers.gpt_neo.modeling_gpt_neo import (
    GPTNeoAttentionMixin,
    GPTNeoForCausalLM,
)

from stable_library_code.transformers.gpt2.partial_forward import partial_forward as gpt2_partial_forward
from stable_library_code.transformers.gpt_neo.partial_forward import partial_forward as gpt_neo_partial_forward

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

    if pad_to_mult:
        true_len = len(feed_in['input_ids'][0])
        pad_to_len = pad_to_mult * (true_len // pad_to_mult + 1)
        pad_to_len = min(pad_to_len, max_length)
        feed_in = tokenizer(
            batch_texts, padding='max_length', truncation=True, max_length=pad_to_len
        )

    for k in feed_in.keys():
        feed_in[k] = np.asarray(feed_in[k])

    input_ids_with_pads = feed_in["input_ids"].copy()
    input_ids_with_pads = torch.as_tensor(input_ids_with_pads).pin_memory().to(device)

    feed_in["input_ids"][feed_in["input_ids"] == tokenizer.pad_token_id] = 0

    for k in feed_in.keys():
        feed_in[k] = torch.as_tensor(feed_in[k]).to(device)

    input_ids = feed_in["input_ids"]
    attention_mask = feed_in["attention_mask"]
    return input_ids, attention_mask, input_ids_with_pads


def get_child_module_by_names(module, names):
    obj = module
    for getter in map(lambda name: lambda obj: getattr(obj, name), names):
        obj = getter(obj)
    return obj


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

        hidden_state_at_last_token = select_at_last_token(
            hidden_states, input_ids_with_pads
        ).unsqueeze(-2)

        query = self.q_proj(hidden_state_at_last_token)
        key = self.k_proj(hidden_states)
        value = self.v_proj(hidden_states)

        query = self._split_heads(query, self.n_head, self.head_dim)
        key = self._split_heads(key, self.n_head, self.head_dim)
        value = self._split_heads(value, self.n_head, self.head_dim)

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
        attn_output = self.out_proj(attn_output)
        attn_output = self.res_dropout(attn_output)

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


NostARHeadArchitectureParams = NamedTuple(
    "NostARHeadArchitectureParams",
    layer_nums=List[int],
    n_head=Union[int, List[int]],
    mlp_ratio=Union[int, float],
    attn_dropout=float,
    res_dropout=float,
    init_gain=float,
    init_gain_logit_head=float,
)


def validate_arch_params(params: NostARHeadArchitectureParams):
    if not isinstance(params.n_head, int):
        if len(params["n_head"]) != len(params["layer_nums"]):
            msg = "n_head and layer_nums must be equal length, got "
            msg += f"n_head={params['n_head']}, layer_nums={params['layer_nums']}"
            raise ValueError(msg)


class NostARHead(nn.Module):
    def __init__(
        self,
        base_model: GPTModelType,  # TODO: make compat with GPTNeoModel, etc?
        tokenizer: GPT2TokenizerType,
        params: NostARHeadArchitectureParams,
    ):
        validate_arch_params(params)

        super().__init__()

        self._base_model = weakref.ref(base_model)
        self._tokenizer = weakref.ref(tokenizer)
        self.params = params

        self._setup()
        self.init_weights()

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
            nn.init.orthogonal_(module.weight, gain=self.params.init_gain_logit_head)
            print(
                f"initialized logit_head with gain {self.params.init_gain_logit_head:.2f}"
            )
        elif isinstance(module, (nn.Linear,)):
            nn.init.orthogonal_(module.weight, gain=self.params.init_gain)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _setup_attns(self):
        self.attns = nn.ModuleList(
            [
                NostARHeadAttention(
                    n_head=nh,
                    base_model_config=self.base_model.config,
                    attn_dropout=self.params.attn_dropout,
                    res_dropout=self.params.res_dropout,
                )
                for nh in self.n_head
            ]
        )

        self.layer_nums_to_attns = {
            lnum: attn for lnum, attn in zip(self.layer_nums, self.attns)
        }

    def _setup(self):
        if isinstance(self.base_model, GPT2LMHeadModel):
            self.partial_forward = gpt2_partial_forward
        elif isinstance(self.base_model, GPTNeoForCausalLM):
            self.partial_forward = gpt_neo_partial_forward
        else:
            raise ValueError(self.base_model)

        for param in self.base_model.parameters():
            param.requires_grad = False

        # extract_activations(
        #     self.base_model,
        #     layer_names=self.layer_nums,
        #     prefixes=['transformer', 'h']
        # )

        self._setup_attns()

        mlp_input_size = len(self.layer_nums) * self.base_model.config.hidden_size
        mlp_intermediate_size = int(mlp_input_size * self.params.mlp_ratio)

        self.mlp = NostARHeadMLP(
            input_size=mlp_input_size, intermediate_size=mlp_intermediate_size
        )

        self.logit_head = nn.Linear(mlp_input_size, 2)

    def forward(
        self,
        input_ids,
        input_ids_with_pads,
        attention_mask,
        head_mask=None,
        output_attentions=False,
        use_amp_in_base_forward=False,
    ):
        with torch.no_grad():
            extracted_activations = self.partial_forward(
                model=self.base_model.transformer,
                layer_nums=self.layer_nums,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_amp=use_amp_in_base_forward
            )

        attn_outs = [
            attn(extracted_activations[lnum], input_ids_with_pads)[0]
            for lnum, attn in self.layer_nums_to_attns.items()
        ]

        hidden_state = torch.cat(attn_outs, dim=-1)

        hidden_state = hidden_state + self.mlp(hidden_state)

        logits = self.logit_head(hidden_state)

        return logits
