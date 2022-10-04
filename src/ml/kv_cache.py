"""
Pre-allocates a fixed buffer for the key/value cache used in autoregressive sampling, to avoid growth/fragmentation in CUDA memory from repeatedly `torch.cat`-ing CUDA tensors with local scope.

Monkey-patches HF transformers code.
"""
import torch
import transformers.models.gpt_neo.modeling_gpt_neo

def make_kv_cache_hook(bs, maxlen):
    def hook(model):
        print(f'kv cache hook called with bs={bs}, maxlen={maxlen}')
        for l in model.transformer:
            shp_k = [
                bs,
                maxlen,
                model.lm.config.num_heads,
                model.lm.config.hidden_size // model.lm.config.num_heads
            ]
            shp_v = [
                bs,
                model.lm.config.num_heads,
                maxlen,
                model.lm.config.hidden_size // model.lm.config.num_heads
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


def slice_scatter(a, b, offset=0):
    ix = torch.arange(offset, offset+b.shape[2], device=a.device)[None, None, :, None].expand_as(b)
    a.scatter_(dim=2, src=b, index=ix)

def set_past(self, layer_past):
    past_key = layer_past[0]
    past_value = layer_past[1]

    seqlen = past_key.shape[2]
    self.seqlen = seqlen

    slice_scatter(self.bufk, past_key)
    slice_scatter(self.bufv, past_value)

def clear_past(self):
    self.seqlen = None

def shift_past(self, offset):
    if self.seqlen is None:
        raise ValueError

    self.seqlen -= offset

    self.bufk = self.bufk.roll(-offset, 2)
    self.bufv = self.bufv.roll(-offset, 2)

def kv_buffer_gpt_neo_selfattn_forward(
    self,
    hidden_states,
    attention_mask=None,
    layer_past=None,
    head_mask=None,
    use_cache=False,
    output_attentions=False,
):
    query = self.q_proj(hidden_states)
    key = self.k_proj(hidden_states)
    value = self.v_proj(hidden_states)

    query = self._split_heads(query, self.num_heads, self.head_dim, self.rotary)
    key = self._split_heads(key, self.num_heads, self.head_dim, self.rotary)
    value = self._split_heads(value, self.num_heads, self.head_dim, False)

    if self.seqlen is not None:
        slice_scatter(self.bufk, key, offset=self.seqlen)
        key = self.bufk[:, :, :self.seqlen+1, :]

        slice_scatter(self.bufv, value, offset=self.seqlen)
        value = self.bufv[:, :, :self.seqlen+1, :]
    elif layer_past is not None:
        past_key = layer_past[0]
        past_value = layer_past[1]

        seqlen = past_key.shape[2]

        slice_scatter(self.bufk, key, offset=seqlen)
        key = self.bufk[:, :, :seqlen+1, :]

        slice_scatter(self.bufv, value, offset=seqlen)
        value = self.bufv[:, :, :seqlen+1, :]
    elif use_cache:
        slice_scatter(self.bufk, key)
        slice_scatter(self.bufv, value)

    if use_cache is True:
        present = (key, value)
    else:
        present = None

    if self.rotary:
        offset_q = 0
        if self.seqlen is not None:
            offset_q = self.seqlen
        elif layer_past is not None:
            offset_q = layer_past[0].shape[-2]
        offset_k = 0 if prerot else offset_q
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
    model,  # magma wrapper -- TODO: work with ordinary HF model
    batch_size,
    max_sequence_length,
):
    model.add_adapters()

    if not hasattr(model.transformer[0].attn.module.attention, 'bufk'):
        model.detach_adapters()
        make_kv_cache_hook(batch_size, max_sequence_length)(model)
        model.add_adapters()

        transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention.forward = kv_buffer_gpt_neo_selfattn_forward

        transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention.set_past = set_past
        transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention.clear_past = clear_past
        transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention.shift_past = shift_past

    model.detach_adapters()
