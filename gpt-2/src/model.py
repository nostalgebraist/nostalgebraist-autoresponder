import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

from tensorflow.python.tpu.ops import tpu_ops


def default_hparams():
    return HParams(
        n_vocab=50257,
        n_ctx=1024,
        n_embd=768,
        n_head=12,
        n_layer=12,
        acti_dropout=0.0,
        attn_dropout=0.0,
        res_dropout=0.0,
        dtype=tf.float32,
        use_adapters=False,
        adapter_scope="adapt",
        attn_adapters=True,
        mlp_adapters=True,
        adapters_share_proj=False,
        adapters_ln_after_downproj=True,
        acti_dropout_adapt=0.0,
        attn_dropout_adapt=0.0,
        res_dropout_adapt=0.0,
        n_adapt_attn=128,
        nhead_adapt_attn=4,
        n_adapt_mlp=256,
        orth_init=False,
        he_init=False,
        init_default_gain=1.0,
        adapt_layers=[0],
        selector_style_attn=False,
    )


def hparams_1558M():
    return HParams(
        n_vocab=50257,
        n_ctx=1024,
        n_embd=1600,
        n_head=25,
        n_layer=48,
        acti_dropout=0.0,
        attn_dropout=0.0,
        res_dropout=0.0,
        dtype=tf.float32,
        use_adapters=False,
        adapter_scope="adapt",
        attn_adapters=True,
        mlp_adapters=True,
        adapters_share_proj=False,
        adapters_ln_after_downproj=True,
        acti_dropout_adapt=0.0,
        attn_dropout_adapt=0.0,
        res_dropout_adapt=0.0,
        n_adapt_attn=128,
        nhead_adapt_attn=4,
        n_adapt_mlp=256,
        orth_init=False,
        he_init=False,
        init_default_gain=1.0,
        adapt_layers=[0],
    )


import os


def get_variable(name):
    name = os.path.join(tf.get_variable_scope().name, name)
    vs = tf.trainable_variables()

    for x in vs:
        if x.name.startswith(name + ":"):
            return x


def shape_list(x):
    """Deal with dynamic shape in tensorflow cleanly."""
    static = x.shape.as_list()
    dynamic = tf.shape(x)
    return [dynamic[i] if s is None else s for i, s in enumerate(static)]


def softmax(x, axis=-1):
    x = x - tf.reduce_max(x, axis=axis, keepdims=True)
    ex = tf.exp(x)
    return ex / tf.reduce_sum(ex, axis=axis, keepdims=True)


def gelu(x):
    return 0.5 * x * (1 + tf.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))


def norm(x, scope, *, axis=-1, epsilon=1e-5, hparams=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        n_state = x.shape[-1].value
        g = get_variable("g") or tf.get_variable(
            "g", [n_state], initializer=tf.constant_initializer(1, dtype=dtype)
        )
        b = get_variable("b") or tf.get_variable(
            "b", [n_state], initializer=tf.constant_initializer(0, dtype=dtype)
        )
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x - u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x * g + b
        return x


def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m // n])


def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a * b])


def get_initializer(hparams, scope, fan_in, gain=None):
    if gain is None:
        gain = hparams.init_default_gain
        if hparams.he_init:
            gain = np.sqrt(2.0 / fan_in)
            if not hparams.get("silent", True):
                print(f"he init in scope {scope}: fan_in {fan_in}, gain {gain:.4f}")
        else:
            if not hparams.get("silent", True):
                print(f"default init in scope {scope}: gain {gain:.4f}")
    else:
        if not hparams.get("silent", True):
            print(f"override init in scope {scope}, gain {gain:.4f}")
    initializer = lambda **kwargs: tf.random_normal_initializer(stddev=gain, **kwargs)
    if hparams.get("orth_init"):
        initializer = lambda **kwargs: tf.compat.v1.orthogonal_initializer(
            gain=gain, **kwargs
        )
    return initializer


def conv1d(x, scope, nf, *, gain=None, hparams=None):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        *start, nx = shape_list(x)
        initializer = get_initializer(hparams, scope, nx, gain)
        w = get_variable("w") or tf.get_variable(
            "w", [1, nx, nf], initializer=initializer(dtype=dtype)
        )
        b = get_variable("b") or tf.get_variable(
            "b", [nf], initializer=tf.constant_initializer(0, dtype=dtype)
        )
        c = tf.reshape(
            tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf])) + b,
            start + [nf],
        )
        return c


def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams, n_head=None, gain=None, adapt=False):
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
        return tf.transpose(split_states(x, n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w * b - tf.cast(65500 if w.dtype != tf.float32 else 1e10, w.dtype) * (1 - b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        attn_dropout = (
            hparams.get("attn_dropout_adapt", 0)
            if adapt
            else hparams.get("attn_dropout", 0)
        )
        if hparams.get("attn_dropout_before_softmax", False):
            w = dropout(w, attn_dropout)
        w = softmax(w)
        if not hparams.get("attn_dropout_before_softmax", False):
            w = dropout(w, attn_dropout)
        a = tf.matmul(w, v)
        return a

    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        c = conv1d(x, "c_attn", n_state * 3, hparams=hparams)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, "c_proj", n_state, hparams=hparams)
        res_dropout = (
            hparams.get("res_dropout_adapt", 0)
            if adapt
            else hparams.get("res_dropout", 0)
        )
        a = dropout(a, res_dropout)
        return a, present


def mlp(x, scope, n_state, *, hparams, gain=None, adapt=False):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, "c_fc", n_state, gain=gain, hparams=hparams))
        acti_dropout = (
            hparams.get("acti_dropout_adapt", 0)
            if adapt
            else hparams.get("acti_dropout", 0)
        )
        h = dropout(h, acti_dropout)
        h2 = conv1d(h, "c_proj", nx, gain=gain, hparams=hparams)
        res_dropout = (
            hparams.get("res_dropout_adapt", 0)
            if adapt
            else hparams.get("res_dropout", 0)
        )
        h2 = dropout(h2, res_dropout)
        return h2


def dropout(x, pdrop=0.1, train=True):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, rate=pdrop)
    return x


def block(
    x,
    scope,
    *,
    past,
    past_adapt,
    hparams,
    adapt=False,
    h_next=None,
    mix_next_weight_next=1.0,
):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        # if h_next is not None and x.shape[-2] > 1:
        #     x = normed_mix(x, h_next, mix_next_weight_next, "normed_mix")
        x_ln1 = norm(x, "ln_1", hparams=hparams)
        a, present = attn(x_ln1, "attn", nx, past=past, hparams=hparams)
        x = x + a

    if adapt and hparams.attn_adapters:
        adapter_scope = hparams.get("adapter_scope", "adapt")
        with tf.variable_scope(adapter_scope):
            if hparams.adapters_ln_after_downproj:
                proj_down_fn = lambda _: norm(
                    conv1d(_, f"proj_down_attn", hparams.n_adapt_attn, hparams=hparams),
                    "adapt_ln_1_attn",
                    hparams=hparams,
                )
                adapt_in = x
            else:
                proj_down_fn = lambda _: conv1d(
                    _, f"proj_down_attn", hparams.n_adapt_attn, hparams=hparams
                )
                adapt_in = x_ln1
            proj_up_fn = lambda _: conv1d(_, "proj_up_attn", nx, hparams=hparams)
            attn_fn = lambda _: attn(
                _,
                "attn",
                hparams.n_adapt_attn,
                past=past_adapt,
                hparams=hparams,
                n_head=hparams.nhead_adapt_attn,
                adapt=True,
            )
            if hparams.adapters_share_proj:
                proj_down = proj_down_fn(adapt_in)
                with tf.variable_scope(scope, dtype=dtype):
                    a_adapt_down, present_adapt = attn_fn(proj_down)
                a_adapt = proj_up_fn(a_adapt_down)
            else:
                with tf.variable_scope(scope, dtype=dtype):
                    proj_down = proj_down_fn(adapt_in)
                    a_adapt_down, present_adapt = attn_fn(proj_down)
                    a_adapt = proj_up_fn(a_adapt_down)
            x = x + a_adapt
    else:
        present_adapt = present

    with tf.variable_scope(scope, dtype=dtype):
        x_ln2 = norm(x, "ln_2", hparams=hparams)
        m = mlp(x_ln2, "mlp", nx * 4, hparams=hparams)
        x = x + m

    if adapt and hparams.mlp_adapters:
        with tf.variable_scope(adapter_scope):
            if hparams.adapters_ln_after_downproj:
                proj_down_mlp_fn = lambda _: norm(
                    conv1d(_, "proj_down_mlp", hparams.n_adapt_mlp, hparams=hparams),
                    "adapt_ln_1_mlp",
                    hparams=hparams,
                )
                adapt_in = x
            else:
                proj_down_mlp_fn = lambda _: conv1d(
                    _, "proj_down_mlp", hparams.n_adapt_mlp, hparams=hparams
                )
                adapt_in = x_ln2
            mlp_fn = lambda _: dropout(gelu(_), hparams.acti_dropout_adapt)
            proj_up_mlp_fn = lambda _: conv1d(_, "proj_up_mlp", nx, hparams=hparams)

            if hparams.adapters_share_proj:
                proj_down = proj_down_mlp_fn(adapt_in)
                m_adapt_down = mlp_fn(
                    proj_down
                )  # this creates no variables so doesn't need a scope context manager
                m_adapt = proj_up_fn(m_adapt_down)
            else:
                with tf.variable_scope(scope, dtype=dtype):
                    proj_down = proj_down_mlp_fn(adapt_in)
                    m_adapt_down = mlp_fn(proj_down)
                    m_adapt = proj_up_mlp_fn(m_adapt_down)
            x = x + m_adapt

    return x, present, present_adapt


def past_shape(*, hparams, batch_size=None, sequence=None):
    return [
        batch_size,
        hparams.n_layer,
        2,
        hparams.n_head,
        sequence,
        hparams.n_embd // hparams.n_head,
    ]


def past_shape_adapt(*, hparams, batch_size=None, sequence=None):
    return [
        batch_size,
        hparams.n_layer,
        2,
        hparams.nhead_adapt_attn,
        sequence,
        hparams.n_adapt_attn // hparams.nhead_adapt_attn,
    ]


def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name="value")
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1] * ndims)


def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)


N_REPLICAS = 8
LAYERS_PER_REPLICA = 4
TPU_PERMUTE = False


def model(
    hparams,
    X,
    past=None,
    past_adapt=None,
    scope="model",
    reuse=tf.AUTO_REUSE,
    return_activations_at=[],
    return_activations_only=False,
    midpoint=None,
    midpoint_vect=None,
    stop_grad_at_midpoint=False,
    stop_at_h_out=False,
    h_out_vect=None,
    silent=False,
):
    activations = []
    h_names = []

    if not silent:
        print(f"using hparams: {hparams}")
    X = tf.cast(X, tf.int32)
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, reuse=reuse, dtype=dtype):
        results = {}
        batch, sequence = shape_list(X)

        wpe = get_variable("wpe") or tf.get_variable(
            "wpe",
            [hparams.n_ctx, hparams.n_embd],
            initializer=tf.random_normal_initializer(stddev=0.01, dtype=dtype),
        )
        wte = get_variable("wte") or tf.get_variable(
            "wte",
            [hparams.n_vocab, hparams.n_embd],
            initializer=tf.random_normal_initializer(stddev=0.02, dtype=dtype),
        )
        wte_expansion, wte_full = None, wte
        past_length = 0 if past is None else tf.shape(past)[-2]
        position_emb = tf.gather(wpe, positions_for(X, past_length))
        h = tf.gather(wte, X) + position_emb

        # Transformer
        presents = []
        presents_adapt = []
        pasts = (
            tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        )
        assert len(pasts) == hparams.n_layer

        adapt_layers = hparams.get("adapt_layers", list(range(hparams.n_layer)))

        pasts_adapt = (
            tf.unstack(past_adapt, axis=1)
            if past_adapt is not None
            else [None] * hparams.n_layer
        )
        assert len(pasts_adapt) == hparams.n_layer
        for layer, (past, past_adapt) in enumerate(zip(pasts, pasts_adapt)):
            adapt_here = hparams.get("use_adapters", False) and (layer in adapt_layers)
            h, present, present_adapt = block(
                h,
                "h%d" % layer,
                past=past,
                past_adapt=past_adapt,
                hparams=hparams,
                adapt=adapt_here,
            )

            if layer in return_activations_at:
                h_name = f"h{layer}"
                if not silent:
                    print(f"{h_name} found")
                h_names.append(h_name)
                activations.append(h)

                if return_activations_only and layer == max(return_activations_at):
                    break

            if midpoint is not None:
                if layer == midpoint:
                    results[f"h{midpoint}"] = h
                if stop_grad_at_midpoint and layer < midpoint:
                    h = tf.stop_gradient(h)
            if midpoint_vect is not None:
                if layer == midpoint:
                    return tf.reduce_mean(tf.einsum("aij,aij->ai", h, midpoint_vect))
            if False:  # layer % 5 == 0:#if layer == 10:
                tf.add_to_collection("checkpoints", h)
            presents.append(present)
            presents_adapt.append(present_adapt)

        if not return_activations_only:
            results["present"] = tf.stack(presents, axis=1)
            results["present_adapt"] = tf.stack(presents_adapt, axis=1)
            h = norm(h, "ln_f", hparams=hparams)
            results["h_out"] = h
            if h_out_vect is not None:
                return tf.reduce_mean(
                    tf.einsum("aij,aij->ai", results["h_out"], h_out_vect)
                )
            if stop_at_h_out:
                return results

            # Language model loss.  Do tokens <n predict token n?
            # logits = tf.matmul(h, wte_full, transpose_b=True)
            h_flat = tf.reshape(h, [batch * sequence, hparams.n_embd])
            logits = tf.matmul(h_flat, wte_full, transpose_b=True)
            logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
            results["logits"] = logits

        results["activations"] = list(zip(h_names, activations))
        return results
