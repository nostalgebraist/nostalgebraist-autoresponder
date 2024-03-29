from typing import List
import torch
# from stable_library_code.transformers.gpt_neo.modeling_gpt_neo import *


def partial_forward(
    model,
    layer_nums: List[int],
    input_ids=None,
    past_key_values=None,
    attention_mask=None,
    position_ids=None,
    head_mask=None,
    use_amp=False,  # ignored
):
    input_shape = input_ids.size()
    input_ids = input_ids.view(-1, input_shape[-1])
    batch_size = input_ids.shape[0]

    device = input_ids.device

    if position_ids is not None:
        position_ids = position_ids.view(-1, input_shape[-1])

    past_length = 0
    past_key_values = tuple([None] * len(model.h))
    if position_ids is None:
        position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
        position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

    # GPT2Attention mask.
    if attention_mask is not None:
        assert batch_size > 0, "batch_size has to be defined and > 0"
        attention_mask = attention_mask.view(batch_size, -1)
        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        attention_mask = attention_mask[:, None, None, :]

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        attention_mask = attention_mask.to(dtype=model.dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0

    encoder_attention_mask = None

    # Prepare head mask if needed
    # 1.0 in head_mask indicate we keep the head
    # attention_probs has shape bsz x n_heads x N x N
    # head_mask has shape n_layer x batch x n_heads x N x N
    head_mask = model.get_head_mask(head_mask, model.config.n_layer)

    inputs_embeds = model.wte(input_ids)
    position_embeds = model.wpe(position_ids)
    hidden_states = inputs_embeds + position_embeds

    hidden_states = model.drop(hidden_states)

    extracted_activations = {}
    for i, (block, layer_past) in enumerate(zip(model.h, past_key_values)):
        if i > max(layer_nums):
            break

        if i in layer_nums:
            extracted_activations[i] = hidden_states

        outputs = block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask[i],
            encoder_attention_mask=encoder_attention_mask,
        )

        hidden_states = outputs[0]
    return extracted_activations
