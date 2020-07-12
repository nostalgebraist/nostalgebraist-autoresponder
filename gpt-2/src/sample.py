import tensorflow as tf

import model

def top_k_logits(logits, k, epsilon=-1e10):
    if k == 0:
        # no truncation
        return logits

    def _top_k():
        values, _ = tf.nn.top_k(logits, k=k)
        min_values = values[:, -1, tf.newaxis]
        return tf.where(
            logits < min_values,
            tf.ones_like(logits, dtype=logits.dtype) * epsilon,
            logits,
        )
    return tf.cond(
       tf.equal(k, 0),
       lambda: logits,
       lambda: _top_k(),
    )


def top_p_logits(logits, p, epsilon=-1e10):
    with tf.variable_scope('top_p_logits'):
        logits_sort = tf.sort(logits, direction='DESCENDING')
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.where(probs_sums < p, logits_sort, tf.ones_like(logits_sort)*1000) # [batchsize, vocab]
        min_logits = tf.reduce_min(logits_masked, axis=1, keepdims=True) # [batchsize, 1]
        return tf.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * epsilon,
            logits,
        )


def midde_p_logits(logits, p_base):
    with tf.variable_scope('middle_p_logits'):
        p = 1. - ((1. - p_base)/2.)
        logits_sort_desc = tf.sort(logits, direction='DESCENDING')
        probs_sort_desc = tf.nn.softmax(logits_sort_desc)
        probs_sums_desc = tf.cumsum(probs_sort_desc, axis=1, exclusive=True)
        logits_masked_desc = tf.where(probs_sums_desc < p, logits_sort_desc, tf.ones_like(logits_sort_desc)*1000) # [batchsize, vocab]

        logits_sort_asc = tf.sort(logits, direction='ASCENDING')
        probs_sort_asc = tf.nn.softmax(logits_sort_asc)
        probs_sums_asc = tf.cumsum(probs_sort_asc, axis=1, exclusive=True)
        logits_masked_asc = tf.where(probs_sort_asc < p, logits_sort_asc, tf.ones_like(logits_sort_asc)*1000) # [batchsize, vocab]

        min_logits = tf.reduce_min(logits_masked_desc, axis=1, keepdims=True) # [batchsize, 1]
        max_logits = tf.reduce_max(logits_masked_asc, axis=1, keepdims=True) # [batchsize, 1]
        return tf.where(
            tf.math.logical_or(logits < min_logits, logits > max_logits),
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )



def sample_sequence(*, hparams, length, start_token=None, batch_size=None, context=None, temperature=1, top_k=0, top_p=0.0, middle_p=0.0, epsilon=-1e10, stop_at_EOT=False, eot_workaround=False, better_length=True, enc=None, steps_per_print=10, prints_per_newline=18):
    if start_token is None:
        assert context is not None, 'Specify exactly one of start_token and context!'
    else:
        assert context is None, 'Specify exactly one of start_token and context!'
        context = tf.fill([batch_size, 1], start_token)

    if enc is not None:
        EOT_TOKEN = enc.encode("<|endoftext|>")[0]
        EOT_TOKEN2 = enc.encode("<|endoftext|>")[1]
        print(f"EOT_TOKEN: {EOT_TOKEN}")

        if not eot_workaround:
            EOT_TOKEN2 = enc.encode("<|endoftext|>")[1]
        print(f"EOT_TOKEN2: {EOT_TOKEN2}")

    elif stop_at_EOT:
        raise ValueError("must supply enc when stop_at_EOT=True")

    def step(hparams, tokens, past=None):
        lm_output = model.model(hparams=hparams, X=tokens, past=past, reuse=tf.AUTO_REUSE)
        if hparams.dtype != tf.float32:
            lm_output["logits"] = tf.cast(lm_output["logits"], tf.float32)

        logits = lm_output['logits'][:, :, :hparams.n_vocab]
        presents = lm_output['present']
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))
        return {
            'logits': logits,
            'presents': presents,
        }

    with tf.name_scope('sample_sequence'):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        context_output = step(hparams, context[:, :-1])

        def body(past, prev, output):
            steps_so_far = tf.reduce_sum(tf.ones_like(output), 1)[0]
            print_with_newline_op = tf.print(steps_so_far, end="...\n")
            print_no_newline_op = tf.print(steps_so_far, end="...")
            print_nothing_op = tf.print("", end="")

            print_something_op = tf.cond(tf.equal(tf.mod(steps_so_far, steps_per_print*prints_per_newline), 0),
                                         lambda: print_with_newline_op,
                                         lambda: print_no_newline_op)

            print_op = tf.cond(tf.equal(tf.mod(steps_so_far, steps_per_print), 0),
                               lambda: tf.cond(tf.equal(tf.mod(steps_so_far, steps_per_print*prints_per_newline), 0),
                                               lambda: tf.print(steps_so_far, end="...\n"),
                                               lambda: tf.print(steps_so_far, end="...")),
                               lambda: tf.print("", end=""))

            with tf.control_dependencies([print_op]):
                next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            logits = next_outputs['logits'][:, -1, :]  / tf.to_float(temperature)
            if middle_p > 0.0:
                logits = midde_p_logits(logits, p_base=middle_p)
            elif top_p > 0.0:
                logits = top_p_logits(logits, p=top_p, epsilon=epsilon)
            else:
                logits = top_k_logits(logits, k=top_k, epsilon=epsilon)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)
            return [
                tf.concat([past, next_outputs['presents']], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
            ]

        def cond(*args):
            if better_length:
                length_cond = tf.reduce_sum(tf.ones_like(args[2]), 1)[0] < length
            else:
                length_cond = True

            if stop_at_EOT:
                print("stop_at_EOT in cond")
                if eot_workaround:
                    eot_cond = tf.logical_not(tf.reduce_all( tf.reduce_any(tf.equal(tf.dtypes.cast(args[2][:, 1:], tf.int32), EOT_TOKEN) , axis=1 ), axis=0 ) )
                else:
                    eot_cond = tf.logical_not(tf.reduce_all( tf.reduce_any(tf.logical_and(tf.equal(tf.dtypes.cast(args[2][:, :-1], tf.int32), EOT_TOKEN), tf.equal(tf.dtypes.cast(args[2][:, 1:], tf.int32), EOT_TOKEN2), ) , axis=1 ), axis=0 ) )
            else:
                eot_cond = True

            return tf.logical_and(length_cond, eot_cond)

        _, _, tokens = tf.while_loop(
            cond=cond, body=body,
            maximum_iterations=length if not better_length else None,
            loop_vars=[
                context_output['presents'],
                context[:, -1],
                context,
            ],
            shape_invariants=[
                tf.TensorShape(model.past_shape(hparams=hparams, batch_size=batch_size)),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
        )

        return tokens
