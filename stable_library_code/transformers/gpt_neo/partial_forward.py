from typing import List
from stable_library_code.transformers.gpt_neo.modeling_gpt_neo import *


def partial_forward(
    model,
    layer_nums: List[int],
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    position_ids=None,
    head_mask=None,
    use_amp=False,
):
    device = input_ids.device

    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    batch_size = input_ids.shape[0]

    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    past_length = 0
    past_key_values = tuple([None] * len(model.h))

    position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # Attention mask.
    if attention_mask is not None:
        assert batch_size > 0, "batch_size has to be defined and > 0"
        global_attention_mask = attention_mask.view(batch_size, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        global_attention_mask = global_attention_mask[:, None, None, :]

        # Since global_attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        global_attention_mask = global_attention_mask.to(dtype=model.dtype)  # fp16 compatibility
        global_attention_mask = (1.0 - global_attention_mask) * -10000.0
    else:
        global_attention_mask = None

    # Local causal attention mask
    batch_size, seq_length = input_shape
    full_seq_length = seq_length + past_length
    local_attention_mask = GPTNeoAttentionMixin.create_local_attention_mask(
        batch_size, full_seq_length, model.config.window_size, device, attention_mask
    )

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x num_heads x N x N
    # head_mask has shape n_layer x batch x num_heads x N x N
    head_mask = model.get_head_mask(head_mask, model.config.num_layers)

    inputs_embeds = model.wte(input_ids)
    try:
        position_embeds = model.wpe(position_ids)
    except Exception as e:
        print(f"failed trying to call model.wpe on position_ids {position_ids.shape}")
        raise e
    hidden_states = inputs_embeds + position_embeds

    hidden_states = model.drop(hidden_states)

    extracted_activations = {}
    for i, (block, layer_past) in enumerate(zip(model.h, past_key_values)):
        if i > max(layer_nums):
            break

        autocast_this_block = use_amp and (i not in layer_nums)
        if not autocast_this_block:
            hidden_states = hidden_states.to(torch.float32)

        if i in layer_nums:
            extracted_activations[i] = hidden_states

        attn_type = model.config.attention_layers[i]
        attn_mask = global_attention_mask if attn_type == "global" else local_attention_mask

        with torch.cuda.amp.autocast(enabled=autocast_this_block):
            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attn_mask,
                head_mask=head_mask[i],
            )

        # print(f"block {i}: {torch.cuda.memory_allocated() / (1024**2):.0f} | {hidden_states.shape} | {outputs[0].shape}")

        hidden_states = outputs[0]
    return extracted_activations
