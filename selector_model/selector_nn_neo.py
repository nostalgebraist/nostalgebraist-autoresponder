import numpy as np
import torch
import torch.nn as nn

from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoAttentionMixin
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

    return feed_in, input_ids_with_pads


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
                 res_dropout=0.,):
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

        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

    def forward(
        self,
        hidden_states,
        tokens,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        hidden_state_at_last_token = select_at_last_token(hidden_states, tokens).unsqueeze(-2)
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
