import sys
import tensorflow as tf

sys.path.append("gpt-2/src/")

import model


def attn_only_block(x, scope, *, past, hparams, do_input_norm=True):
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
            )
            hs_select.append(h_select)

            h_select_in = tf.concat(hs_select, axis=-1)

            h_select_in_at_selection_ix = extract_selection_ix(
                X, h_select_in, batch_size=batch_size, selection_tok=selection_tok
            )["extracted"]
            selection_ix_position = tf.cast(
                tf.reshape(
                    extract_selection_ix_position(
                        X, batch_size=batch_size, selection_tok=selection_tok
                    )["extracted"],
                    [-1, 1],
                ),
                tf.float32,
            )

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

        w_select = model.get_variable("w_select")
        if w_select is None:
            initializer = get_initializer(
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

            initializer = model.get_initializer(hparams_select, scope)
            w_select = model.get_variable(
                "w_select",
                [len(layer_nums) * hparams.n_embd, 2],
                initializer=initializer(0.02, dtype=hparams.dtype),
            )

        b_select = model.get_variable("b_select")
        if b_select is None:
            b_select = model.get_variable(
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
