import sys
import tensorflow as tf

sys.path.append("gpt-2/src/")

import model


def selector_attn(x, scope, n_state, *, past, hparams, n_head=None, gain=None, adapt=False,
                  X=None, batch_size=None, selection_tok=None):
    if n_head is None:
        n_head = hparams.n_head
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % n_head == 0
    if past is not None:
        assert (
            past.shape.ndims == 5
        )  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(model.split_states(x, n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return model.merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        print(("w", w))
        _, _, nd, ns = model.shape_list(w)
        b = model.attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(65500 if w.dtype != tf.float32 else 1e10, w.dtype) * (1 - b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        print(("w", w))
        # w = mask_attn_weights(w)
        attn_dropout = (
            hparams.get("attn_dropout_adapt", 0)
            if adapt
            else hparams.get("attn_dropout", 0)
        )
        if hparams.get("attn_dropout_before_softmax", False):
            w = model.dropout(w, attn_dropout)
        w = model.softmax(w)
        if not hparams.get("attn_dropout_before_softmax", False):
            w = model.dropout(w, attn_dropout)
        a = tf.matmul(w, v)
        return a

    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        c = model.conv1d(x, "c_attn", n_state * 3, hparams=hparams)
        q, k, v = tf.split(c, 3, axis=2)
        q = extract_selection_ix(X, q, batch_size, selection_tok)["extracted"]
        print(("q", q))
        q = q[:, tf.newaxis, :]
        print(("q", q))
        print(("k", k))
        print(("v", v))
        q, k, v = map(split_heads, [q, k, v])
        print(("q", q))
        print(("k", k))
        print(("v", v))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = model.conv1d(a, "c_proj", n_state, hparams=hparams)
        res_dropout = (
            hparams.get("res_dropout_adapt", 0)
            if adapt
            else hparams.get("res_dropout", 0)
        )
        a = model.dropout(a, res_dropout)
        return a, present


def attn_only_block(x, scope, *, past, hparams, do_input_norm=True,
                    X=None, batch_size=None, selection_tok=None):
    dtype = hparams.dtype if hparams else tf.float32
    do_resid = hparams.do_resid if hparams else True
    print(f"do_resid: {do_resid}")
    print(f"do_input_norm: {do_input_norm}")
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value

        if do_input_norm:
            x_attn_in = model.norm(x, "ln_1", hparams=hparams)
        else:
            x_attn_in = x

        if hparams.get("selector_style_attn", False):
            a, present = selector_attn(x_attn_in, "attn", nx, past=past, hparams=hparams,
                                       X=X, batch_size=batch_size, selection_tok=selection_tok)
        else:
            a, present = model.attn(x_attn_in, "attn", nx, past=past, hparams=hparams)
        if do_resid:
            x = x + a
        else:
            x = a

        return x, present


def mlp_acti_dropout(
    x, scope, n_state, *, hparams, gain=None, n_final=None, dropout_final=True
):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        if n_final is None:
            n_final = nx
        h = model.gelu(model.conv1d(x, "c_fc", n_state, hparams=hparams, gain=gain))
        h = model.dropout(h, hparams.acti_dropout)
        h2 = model.conv1d(h, "c_proj", n_final, hparams=hparams, gain=gain)
        if dropout_final:
            h2 = model.dropout(h2, hparams.res_dropout)
        return h2


def extract_selection_ix(tokens, extract_from, batch_size, selection_tok):
    mask = tf.equal(tf.dtypes.cast(tokens, tf.int32), selection_tok)
    extracted_ragged = tf.ragged.boolean_mask(extract_from, mask)

    row_lengths = extracted_ragged.row_lengths()
    row_ixs = row_lengths - 1
    selection_ix = tf.stack(
        [tf.range(0, batch_size, dtype=tf.int64), row_ixs],
        axis=1,
    )

    extracted = tf.gather_nd(
        extracted_ragged.to_tensor(),
        selection_ix,
    )

    return {"extracted": extracted, "selection_ix": selection_ix}


def extract_selection_ix_position(tokens, batch_size, selection_tok):
    return extract_selection_ix(
        tokens, tf.sort(tf.argsort(tokens)), batch_size, selection_tok
    )


def selector(
    hparams,
    X,
    hparams_select,
    layer_nums: list,
    batch_size,
    selection_tok,
    scope="model",
    select_scope="select",
    reuse=tf.AUTO_REUSE,
    use_mlp: bool = True,
    resid_mlp: bool = True,
    mlp_ratio=1,
    use_only_logit_diff=False,
    use_logit_diff_basis=False,
):
    results = {}

    activations = model.model(
        hparams=hparams,
        X=X,
        scope=scope,
        reuse=reuse,
        return_activations_at=layer_nums,
        return_activations_only=True,
    )["activations"]

    hs_select = []
    with tf.variable_scope(select_scope, reuse=reuse, dtype=hparams_select.dtype):
        for act_name, act in activations:
            h_select, _ = attn_only_block(
                act,
                f"h_select_{act_name}",
                hparams=hparams_select,
                past=None,
                do_input_norm=True,
                X=X,
                batch_size=batch_size,
                selection_tok=selection_tok
            )
            hs_select.append(h_select)

            h_select_in = tf.concat(hs_select, axis=-1)
            if hparams.get("selector_style_attn", False):
                print(("h_select_in", "h_select_in"))
                h_select_in = h_select_in[:, 0, :]
                print(("h_select_in", "h_select_in"))
            else:
                h_select_in_at_selection_ix = extract_selection_ix(
                    X, h_select_in, batch_size=batch_size, selection_tok=selection_tok
                )["extracted"]
            # selection_ix_position = tf.cast(
            #     tf.reshape(
            #         extract_selection_ix_position(
            #             X, batch_size=batch_size, selection_tok=selection_tok
            #         )["extracted"],
            #         [-1, 1],
            #     ),
            #     tf.float32,
            # )

        if use_mlp:
            nx = h_select_in_at_selection_ix.shape[-1].value
            m = mlp_acti_dropout(
                h_select_in_at_selection_ix,
                "select_mlp",
                int(mlp_ratio * nx),
                hparams=hparams_select,
                n_final=None,
                dropout_final=True,
            )
            if resid_mlp:
                h_select_in_at_selection_ix = m + h_select_in_at_selection_ix
            else:
                h_select_in_at_selection_ix = m

        logit_size = 1 if use_only_logit_diff else 2

        w_select = model.get_variable("w_select")
        if w_select is None:
            initializer = model.get_initializer(
                hparams_select,
                select_scope,
                fan_in=nx,
                gain=None
                if hparams_select.get("he_init")
                else hparams_select.get("init_lreg_gain", 0.02),
            )
            w_select = tf.get_variable(
                "w_select",
                [nx, logit_size],
                initializer=initializer(dtype=hparams.dtype),
            )

        b_select = model.get_variable("b_select")
        if b_select is None:
            b_select = tf.get_variable(
                "b_select",
                [2],
                initializer=tf.constant_initializer(0, dtype=hparams.dtype),
            )

        select_logits = tf.matmul(h_select_in_at_selection_ix, w_select) + b_select

    if use_logit_diff_basis:
        logit_sum = select_logits[:, 0]
        logit_diff = select_logits[:, 1]

        neg_logit = logit_sum / 2 - logit_diff / 2
        pos_logit = logit_sum / 2 + logit_diff / 2
        results["logit_sum"] = logit_sum
        results["logit_diff"] = logit_diff
        results["logits_select"] = tf.stack([neg_logit, pos_logit], axis=1)
    elif use_only_logit_diff:
        results["logits_select"] = tf.concat(
            [tf.zeros_like(select_logits), select_logits], axis=1
        )
        results["logit_diff"] = select_logits
    else:
        results["logits_select"] = select_logits

    return results
