"""
Pre-allocates a fixed buffer for the key/value cache used in autoregressive sampling, to avoid growth/fragmentation in CUDA memory from repeatedly `torch.cat`-ing CUDA tensors with local scope.

Monkey-patches HF transformers code.

NOTE: the approach on this branch (`rotary-shift-10-10-22`) didn't work,
possibly because of fp16 roundoff error. :(
"""
from typing import Union
from contextlib import contextmanager

import torch
import transformers.models.gpt_neo.modeling_gpt_neo
import magma

def make_kv_cache_hook(bs, maxlen):
    def hook(model):
        print(f'kv cache hook called with bs={bs}, maxlen={maxlen}')
        for l in model.transformer.h:
            shp_k = [
                bs,
                model.config.num_heads,
                maxlen,
                model.config.hidden_size // model.config.num_heads
            ]
            shp_v = [
                bs,
                model.config.num_heads,
                maxlen,
                model.config.hidden_size // model.config.num_heads
            ]
            if hasattr(l.attn.attention, 'bufk'):
                continue
            l.attn.attention.register_buffer(
                f"bufk",
                torch.zeros(shp_k, device=model.device, dtype=torch.float16),
                persistent=False
            )
            l.attn.attention.register_buffer(
                f"bufv",
                torch.zeros(shp_v, device=model.device, dtype=torch.float16),
                persistent=False
            )
            l.attn.attention.seqlen = None
        return model
    return hook


def slice_scatter_1(a, b, offset=0):
    ix = torch.arange(offset, offset+b.shape[1], device=a.device)[None, :, None, None].expand_as(b)
    a.scatter_(dim=1, src=b, index=ix)

def slice_scatter_2(a, b, offset=0):
    ix = torch.arange(offset, offset+b.shape[2], device=a.device)[None, None, :, None].expand_as(b)
    a.scatter_(dim=2, src=b, index=ix)

def shift_rotary_pos_emb(x, sincos, offset=0):
    """shifts absolute position backwards by offset"""
    sin, cos = map(lambda t: torch.tile(t[offset:offset+1,:], (x.shape[1], 1)), sincos)
    return transformers.models.gpt_neo.modeling_gpt_neo.apply_rotary_pos_emb(
        x,
        (-sin, cos),
        offset=0
    ).to(x.dtype)

def set_past(self, layer_past):
    past_key = layer_past[0]
    past_value = layer_past[1]

    seqlen = past_value.shape[2]
    self.seqlen = seqlen

    slice_scatter_2(self.bufk, past_key)
    slice_scatter_2(self.bufv, past_value)

def clear_past(self):
    self.seqlen = None

def shift_past(self, offset):
    if self.seqlen is None:
        raise ValueError

    self.seqlen -= offset

    self.bufv = self.bufv.roll(-offset, 2)

    key = self.bufk[:, :, -self.seqlen:, :]

    key = key.permute(0, 2, 1, 3)

    k_rot = key[:, :, :, :self.rotary_dim]
    k_pass = key[:, :, :, self.rotary_dim:]

    k_rot = shift_rotary_pos_emb(k_rot, (self.sin, self.cos), offset=offset)

    key = torch.cat([k_rot, k_pass], dim=-1)
    key = key.permute(0, 2, 1, 3)

    slice_scatter_2(self.bufk, key)

def model__set_past(self, past_key_values):
    for block, layer_past in zip(self.transformer.h, past_key_values):
        block.attn.attention.set_past(layer_past)

def model__clear_past(self):
    for block in self.transformer.h:
        block.attn.attention.clear_past()

def model__shift_past(self, offset):
    for block in self.transformer.h:
        block.attn.attention.shift_past(offset)

def model__collect_past(self):
    past = []
    for block in self.transformer.h:
        pk = block.attn.attention.bufk[:, :, :block.attn.attention.seqlen, :]
        pv = block.attn.attention.bufv[:, :, :block.attn.attention.seqlen, :]
        layer_past = (pk, pv)
        past.append(layer_past)
    return tuple(past)

def model__use_kv_buffer(self, enabled=True):
    for block in self.transformer.h:
        block.attn.attention.use_kv_buffer = enabled
        if not enabled:
            block.attn.attention.seqlen = None

@property
def model__using_kv_buffer(self):
    return getattr(self.transformer.h[0].attn.attention, 'use_kv_buffer', False)

def kv_buffer_gpt_neo_selfattn_forward(
    self,
    hidden_states,
    attention_mask=None,
    layer_past=None,
    head_mask=None,
    use_cache=False,
    output_attentions=False,
):
    use_kv_buffer = getattr(self, 'use_kv_buffer', False)

    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    query = self._split_heads(query, self.num_heads, self.head_dim, self.rotary)
    key = self._split_heads(key, self.num_heads, self.head_dim, self.rotary)
    value = self._split_heads(value, self.num_heads, self.head_dim, False)

    if self.rotary:
        offset_q = 0
        if self.seqlen is not None:
            offset_q = self.seqlen
        elif layer_past is not None:
            offset_q = layer_past[0].shape[-2]
        offset_k = offset_q
        if self.rotary_dim < self.head_dim:
            k_rot = key[:, :, :, :self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]

            q_rot = query[:, :, :, :self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]

            k_rot = transformers.models.gpt_neo.modeling_gpt_neo.apply_rotary_pos_emb(k_rot, (self.sin, self.cos), offset=offset_k).to(k_rot.dtype)
            q_rot = transformers.models.gpt_neo.modeling_gpt_neo.apply_rotary_pos_emb(q_rot, (self.sin, self.cos), offset=offset_q).to(q_rot.dtype)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        elif self.rotary:
            key = transformers.models.gpt_neo.modeling_gpt_neo.apply_rotary_pos_emb(key, (self.sin, self.cos), offset=offset).to(key.dtype)
            query = transformers.models.gpt_neo.modeling_gpt_neo.apply_rotary_pos_emb(query, (self.sin, self.cos), offset=offset).to(query.dtype)
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

    if use_kv_buffer:
        if self.seqlen is not None:
            slice_scatter_2(self.bufk, key, offset=self.seqlen)
            key = self.bufk[:, :, :self.seqlen+1, :]

            slice_scatter_2(self.bufv, value, offset=self.seqlen)
            value = self.bufv[:, :, :self.seqlen+1, :]
        elif layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]

            seqlen = past_value.shape[2]

            slice_scatter_2(self.bufk, key, offset=seqlen)
            key = self.bufk[:, :, :seqlen+1, :]

            slice_scatter_2(self.bufv, value, offset=seqlen)
            value = self.bufv[:, :, :seqlen+1, :]
        elif use_cache:
            slice_scatter_2(self.bufk, key)
            slice_scatter_2(self.bufv, value)
    else:
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2).to(key.dtype)
            value = torch.cat((past_value, value), dim=-2).to(value.dtype)

    if use_cache is True:
        present = (key, value)
    else:
        present = None

    query_length, key_length = query.size(-2), key.size(-2)
    causal_mask = self.bias[:, :, key_length - query_length : key_length, :key_length]

    attn_output, attn_weights = self._attn(
        query, key, value, causal_mask, self.masked_bias, self.attn_dropout, attention_mask, head_mask, self.scale_attn, self.full_bf16
    )

    attn_output = self._merge_heads(attn_output, self.num_heads, self.head_dim)
    attn_output = self.out_proj(attn_output)
    attn_output = self.resid_dropout(attn_output)

    outputs = (attn_output, present)
    if output_attentions:
        outputs += (attn_weights,)

    if self.seqlen is not None:
        self.seqlen += 1

    return outputs


def setup_kv_buffer(
    model: Union[transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM, magma.Magma],
    batch_size,
    max_sequence_length,
):
    is_magma_wrapper = isinstance(model, magma.Magma)
    lm = model.lm if is_magma_wrapper else model

    orig_adapters_attached = False

    if is_magma_wrapper:
        orig_adapters_attached = len(model.adapter_map) == 0

    if orig_adapters_attached:
        model.detach_adapters()

    if not hasattr(lm.transformer.h[0].attn.attention, 'bufk'):
        transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention.forward = kv_buffer_gpt_neo_selfattn_forward

        make_kv_cache_hook(batch_size, max_sequence_length)(lm)
        lm.use_kv_buffer()

    if orig_adapters_attached:
        model.add_adapters()

transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention.set_past = set_past
transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention.clear_past = clear_past
transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention.shift_past = shift_past

transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM.set_past = model__set_past
transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM.clear_past = model__clear_past
transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM.shift_past = model__shift_past
transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM.collect_past = model__collect_past

transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM.use_kv_buffer = model__use_kv_buffer

transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM.using_kv_buffer = model__using_kv_buffer

@contextmanager
def kv_buffer_scope(
    model: Union[transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoForCausalLM, magma.Magma],
    enabled: bool,
):
    is_magma_wrapper = isinstance(model, magma.Magma)
    lm = model.lm if is_magma_wrapper else model

    orig_adapters_attached = False

    if is_magma_wrapper:
        orig_adapters_attached = len(model.adapter_map) == 0

    if orig_adapters_attached:
        model.detach_adapters()

    orig_enabled = lm.using_kv_buffer
    lm.use_kv_buffer(enabled)

    if orig_adapters_attached:
        model.add_adapters()

    try:
        yield None
    finally:
        if orig_adapters_attached:
            model.detach_adapters()

        lm.use_kv_buffer(orig_enabled)

        if orig_adapters_attached:
            model.add_adapters()
