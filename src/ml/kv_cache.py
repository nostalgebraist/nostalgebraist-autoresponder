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
            shp = [bs, model.lm.config.num_heads, maxlen, model.lm.config.hidden_size // model.lm.config.num_heads]
            if hasattr(l.attn.attention, 'bufk'):
                continue
            l.attn.attention.register_buffer(
                f"bufk",
                torch.zeros(shp, device=model.device, dtype=torch.float16),
                persistent=False
            )
            l.attn.attention.register_buffer(
                f"bufv",
                torch.zeros(shp, device=model.device, dtype=torch.float16),
                persistent=False
            )
        return model
    return hook


def slice_scatter(a, b, offset=0):
    ix = torch.arange(offset, offset+b.shape[2], device=a.device)[None, None, :, None].expand_as(b)
    a.scatter_(dim=2, src=b, index=ix)


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

    if self.rotary:
        seq_len = key.shape[1]
        offset = 0
        if layer_past is not None:
            offset = layer_past[0].shape[-2]
            seq_len += offset
        if self.rotary_dim < self.head_dim:
            k_rot = key[:, :, :, :self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim:]

            q_rot = query[:, :, :, :self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim:]

            k_rot = transformers.models.gpt_neo.modeling_gpt_neo.apply_rotary_pos_emb(k_rot, (self.sin, self.cos), offset=offset).to(k_rot.dtype)
            q_rot = transformers.models.gpt_neo.modeling_gpt_neo.apply_rotary_pos_emb(q_rot, (self.sin, self.cos), offset=offset).to(q_rot.dtype)

            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        elif self.rotary:
            key = transformers.models.gpt_neo.modeling_gpt_neo.apply_rotary_pos_emb(key, (self.sin, self.cos), offset=offset).to(key.dtype)
            query = transformers.models.gpt_neo.modeling_gpt_neo.apply_rotary_pos_emb(query, (self.sin, self.cos), offset=offset).to(query.dtype)
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

    if layer_past is not None:
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

    return outputs

def setup_kv_buffer(
    model,  # magma wrapper -- TODO: work with ordinary HF model
    batch_size,
):
    model.add_adapters()

    if not hasattr(model.transformer[0].attn.module.attention, 'bufk'):
        model.detach_adapters()
        make_kv_cache_hook(batch_size, generator_model.max_feed_size_with_cache)(model)
        model.add_adapters()

        transformers.models.gpt_neo.modeling_gpt_neo.GPTNeoSelfAttention.forward = kv_buffer_gpt_neo_selfattn_forward

    model.detach_adapters()
