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
        res_dropout=0.,
        attn_dropout=0.,
        dtype=tf.float32
    )

import os

def get_variable(name):
    name = os.path.join(tf.get_variable_scope().name, name)
    vs = tf.trainable_variables()
    for x in vs:
        if x.name.startswith(name + ':'):
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
    return 0.5*x*(1+tf.tanh(np.sqrt(2/np.pi)*(x+0.044715*tf.pow(x, 3))))

def norm(x, scope, *, axis=-1, epsilon=1e-5, hparams=None):
    """Normalize to mean = 0, std = 1, then do a diagonal affine transform."""
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        n_state = x.shape[-1].value
        g = get_variable('g') or tf.get_variable('g', [n_state], initializer=tf.constant_initializer(1, dtype=dtype))
        b = get_variable('b') or tf.get_variable('b', [n_state], initializer=tf.constant_initializer(0, dtype=dtype))
        u = tf.reduce_mean(x, axis=axis, keepdims=True)
        s = tf.reduce_mean(tf.square(x-u), axis=axis, keepdims=True)
        x = (x - u) * tf.rsqrt(s + epsilon)
        x = x*g + b
        return x

def split_states(x, n):
    """Reshape the last dimension of x into [n, x.shape[-1]/n]."""
    *start, m = shape_list(x)
    return tf.reshape(x, start + [n, m//n])

def merge_states(x):
    """Smash the last two dimensions of x into a single dimension."""
    *start, a, b = shape_list(x)
    return tf.reshape(x, start + [a*b])

def conv1d(x, scope, nf, *, w_init_stdev=0.02, hparams=None):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        *start, nx = shape_list(x)
        w = get_variable('w') or tf.get_variable('w', [1, nx, nf], initializer=tf.random_normal_initializer(stddev=w_init_stdev, dtype=dtype))
        b = get_variable('b') or tf.get_variable('b', [nf], initializer=tf.constant_initializer(0, dtype=dtype))
        c = tf.reshape(tf.matmul(tf.reshape(x, [-1, nx]), tf.reshape(w, [-1, nf]))+b, start+[nf])
        return c

def attention_mask(nd, ns, *, dtype):
    """1's in the lower triangle, counting from the lower right corner.

    Same as tf.matrix_band_part(tf.ones([nd, ns]), -1, ns-nd), but doesn't produce garbage on TPUs.
    """
    i = tf.range(nd)[:,None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)


def attn(x, scope, n_state, *, past, hparams):
    assert x.shape.ndims == 3  # Should be [batch, sequence, features]
    assert n_state % hparams.n_head == 0
    if past is not None:
        assert past.shape.ndims == 5  # Should be [batch, 2, heads, sequence, features], where 2 is [k, v]

    def split_heads(x):
        # From [batch, sequence, features] to [batch, heads, sequence, features]
        return tf.transpose(split_states(x, hparams.n_head), [0, 2, 1, 3])

    def merge_heads(x):
        # Reverse of split_heads
        return merge_states(tf.transpose(x, [0, 2, 1, 3]))

    def mask_attn_weights(w):
        # w has shape [batch, heads, dst_sequence, src_sequence], where information flows from src to dst.
        _, _, nd, ns = shape_list(w)
        b = attention_mask(nd, ns, dtype=w.dtype)
        b = tf.reshape(b, [1, 1, nd, ns])
        w = w*b - tf.cast(65500 if w.dtype != tf.float32 else 1e10, w.dtype)*(1-b)
        return w

    def multihead_attn(q, k, v):
        # q, k, v have shape [batch, heads, sequence, features]
        w = tf.matmul(q, k, transpose_b=True)
        w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))

        w = mask_attn_weights(w)
        w = softmax(w)
        w = dropout(w, hparams.attn_dropout)
        a = tf.matmul(w, v)
        return a

    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        c = conv1d(x, 'c_attn', n_state*3, hparams=hparams)
        q, k, v = map(split_heads, tf.split(c, 3, axis=2))
        present = tf.stack([k, v], axis=1)
        if past is not None:
            pk, pv = tf.unstack(past, axis=1)
            k = tf.concat([pk, k], axis=-2)
            v = tf.concat([pv, v], axis=-2)
        a = multihead_attn(q, k, v)
        a = merge_heads(a)
        a = conv1d(a, 'c_proj', n_state, hparams=hparams)
        a = dropout(a, hparams.res_dropout)
        return a, present


def mlp(x, scope, n_state, *, hparams):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        h = gelu(conv1d(x, 'c_fc', n_state, hparams=hparams))
        h2 = conv1d(h, 'c_proj', nx, hparams=hparams)
        h2 = dropout(h2, hparams.res_dropout)
        return h2

def dropout(x, pdrop=0.1, train=True):
    if train and pdrop > 0:
        x = tf.nn.dropout(x, rate=pdrop)
    return x

def block(x, scope, *, past, hparams):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, dtype=dtype):
        nx = x.shape[-1].value
        a, present = attn(norm(x, 'ln_1', hparams=hparams), 'attn', nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, 'ln_2', hparams=hparams), 'mlp', nx*4, hparams=hparams)
        x = x + m
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    return [batch_size, hparams.n_layer, 2, hparams.n_head, sequence, hparams.n_embd // hparams.n_head]

def expand_tile(value, size):
    """Add a new axis of given size."""
    value = tf.convert_to_tensor(value, name='value')
    ndims = value.shape.ndims
    return tf.tile(tf.expand_dims(value, axis=0), [size] + [1]*ndims)

def positions_for(tokens, past_length):
    batch_size = tf.shape(tokens)[0]
    nsteps = tf.shape(tokens)[1]
    return expand_tile(past_length + tf.range(nsteps), batch_size)

N_REPLICAS = 8
LAYERS_PER_REPLICA = 4
TPU_PERMUTE = False

def model(hparams, X, past=None, scope='model', reuse=tf.AUTO_REUSE):
    dtype = hparams.dtype if hparams else tf.float32
    with tf.variable_scope(scope, reuse=reuse, dtype=dtype):
        results = {}
        batch, sequence = shape_list(X)

        wpe = get_variable('wpe') or tf.get_variable('wpe', [hparams.n_ctx, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.01, dtype=dtype))
        wte = get_variable('wte') or tf.get_variable('wte', [hparams.n_vocab, hparams.n_embd],
                             initializer=tf.random_normal_initializer(stddev=0.02, dtype=dtype))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        # Transformer
        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        assert len(pasts) == hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, 'h%d' % layer, past=past, hparams=hparams)
            if (layer % LAYERS_PER_REPLICA == 0) and TPU_PERMUTE:
                source_target_pairs = []
                cur_segment = layer // LAYERS_PER_REPLICA
                for replica_ix in range(N_REPLICAS):
                  if replica_ix == cur_segment:
                      source_target_pairs.append([replica_ix, replica_ix+1])
                  elif replica_ix != cur_segment+1:
                      source_target_pairs.append([replica_ix, replica_ix])

                print(f"layer: {layer}")
                print(f"source_target_pairs: {source_target_pairs}")
                print()

                h = tpu_ops.collective_permute(h, source_target_pairs)
            # if layer % 5 == 0:#if layer == 10:
            #     tf.add_to_collection('checkpoints', h)
            presents.append(present)
        results['present'] = tf.stack(presents, axis=1)
        h = norm(h, 'ln_f', hparams=hparams)

        # Language model loss.  Do tokens <n predict token n?
        h_flat = tf.reshape(h, [batch*sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        logits = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        results['logits'] = logits
        return results
