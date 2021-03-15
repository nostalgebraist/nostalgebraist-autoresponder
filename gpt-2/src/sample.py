# mirostat refs:
#
# https://arxiv.org/pdf/2007.14966.pdf
# https://github.com/basusourya/mirostat/blob/master/mirostat.py

import tensorflow as tf
import numpy as np

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
    with tf.variable_scope("top_p_logits"):
        logits_sort = tf.sort(logits, direction="DESCENDING")
        probs_sort = tf.nn.softmax(logits_sort)
        probs_sums = tf.cumsum(probs_sort, axis=1, exclusive=True)
        logits_masked = tf.where(
            probs_sums < p, logits_sort, tf.ones_like(logits_sort) * 1000
        )  # [batchsize, vocab]
        min_logits = tf.reduce_min(
            logits_masked, axis=1, keepdims=True
        )  # [batchsize, 1]
        return tf.where(
            logits < min_logits,
            tf.ones_like(logits, dtype=logits.dtype) * epsilon,
            logits,
        )


def mirostat_logits(
    logits,
    surprise_target,
    mirostat_mu,
    mirostat_t,
    mirostat_t_sq_sum,
    mirostat_harmonic,
    mirostat_trunc,
):
    n_vocab = tf.cast(logits.shape[-1].value, tf.float32)

    probs = tf.nn.softmax(logits)
    probs_sort = tf.sort(probs, direction="DESCENDING")

    b = tf.log(probs_sort[:, :mirostat_trunc] / probs_sort[:, 1 : mirostat_trunc + 1])
    s = tf.reduce_sum(mirostat_t * b, axis=1) / mirostat_t_sq_sum

    eps = s - 1
    part1 = tf.pow(2.0, mirostat_mu) * eps
    part2 = 1.0 - tf.pow(n_vocab, -eps)
    part3 = part1 / part2
    k = tf.pow(part3, 1.0 / s)
    k = tf.cast(tf.round(k), tf.int32)

    batch_ks = tf.unstack(k, axis=0)
    logits_ms = tf.stack(
        [
            top_k_logits(logits, tf.minimum(bk, logits.shape[-1].value))[j, :]
            for j, bk in enumerate(batch_ks)
        ],
        axis=0,
    )
    return logits_ms, probs, k, s


def mirostat_logits_v2(
    logits,
    surprise_target,
    mirostat_mu,
    mirostat_t,
    mirostat_t_sq_sum,
    mirostat_harmonic,
    mirostat_trunc,
):
    n_vocab = tf.cast(logits.shape[-1].value, tf.float32)

    probs = tf.nn.softmax(logits)
    probs_sort = tf.sort(probs, direction="DESCENDING")

    surprises = tf.log(probs_sort) / (-1.0 * np.log(2))

    denominator = tf.cumsum(probs_sort, axis=1)
    numerator = tf.cumsum(surprises * probs_sort, axis=1)
    surprises_expectations = numerator / denominator
    # surprises_expectations = surprises_csum * mirostat_harmonic

    k_summand = tf.where(
        surprises_expectations < mirostat_mu[:, tf.newaxis],
        tf.ones_like(surprises_expectations, dtype=tf.int32),
        tf.zeros_like(surprises_expectations, dtype=tf.int32),
    )
    k = tf.reduce_sum(k_summand, 1)

    batch_ks = tf.unstack(k, axis=0)
    logits_ms = tf.stack(
        [
            top_k_logits(logits, tf.minimum(bk, logits.shape[-1].value))[j, :]
            for j, bk in enumerate(batch_ks)
        ],
        axis=0,
    )
    s = tf.zeros_like(mirostat_mu)
    return logits_ms, probs, k, s


def midde_p_logits(logits, retain_middle=None, chop_lowest=None, chop_highest=None):
    with tf.variable_scope("middle_p_logits"):
        if retain_middle is None and (chop_lowest is None or chop_highest is None):
            raise ValueError(
                "must supply retain_middle, or both chop_lowest and chop_highest"
            )
        if chop_lowest is None:
            p = 1.0 - ((1.0 - retain_middle) / 2.0)
        else:
            p = 1.0 - chop_lowest
        logits_sort_desc = tf.sort(logits, direction="DESCENDING")
        probs_sort_desc = tf.nn.softmax(logits_sort_desc)
        probs_sums_desc = tf.cumsum(probs_sort_desc, axis=1, exclusive=True)
        logits_masked_desc = tf.where(
            probs_sums_desc < p, logits_sort_desc, tf.ones_like(logits_sort_desc) * 1000
        )  # [batchsize, vocab]

        if chop_highest is None:
            p = 1.0 - ((1.0 - retain_middle) / 2.0)
        else:
            p = 1.0 - chop_highest
        logits_sort_asc = tf.sort(logits, direction="ASCENDING")
        probs_sort_asc = tf.nn.softmax(logits_sort_asc)
        probs_sums_asc = tf.cumsum(probs_sort_asc, axis=1, exclusive=True)
        logits_masked_asc = tf.where(
            probs_sort_asc < p, logits_sort_asc, tf.ones_like(logits_sort_asc) * 1000
        )  # [batchsize, vocab]

        min_logits = tf.reduce_min(
            logits_masked_desc, axis=1, keepdims=True
        )  # [batchsize, 1]
        max_logits = tf.reduce_max(
            logits_masked_asc, axis=1, keepdims=True
        )  # [batchsize, 1]
        return tf.where(
            tf.math.logical_or(logits < min_logits, logits > max_logits),
            tf.ones_like(logits, dtype=logits.dtype) * -1e10,
            logits,
        )


def apply_avoid_tag_only_posts(
    logits,
    prev,
    output,
    initial_newline_count,
    newline_token,
    space_token,
):
    # build boolean trigger

    last_token_was_newline = tf.equal(
        tf.dtypes.cast(prev, tf.int32),
        newline_token
    )  # [batchsize,]

    newline_mask = tf.equal(
        tf.dtypes.cast(output, tf.int32),
        newline_token
    )  # [batchsize, ntok]

    newline_count = tf.reduce_sum(
        tf.dtypes.cast(newline_mask, tf.int32),
        axis=-1
    )  # [batchsize,]

    only_one_newline = tf.equal(newline_count - initial_newline_count, 1)  # [batchsize,]

    trigger_condition = tf.math.logical_and(
        last_token_was_newline,
        only_one_newline
    )  # [batchsize,]

    # build replacement
    space_logits = logits[:, space_token]  # [batchsize,]
    space_logits = tf.where(
        trigger_condition,
        tf.ones_like(space_logits, dtype=space_logits.dtype) * -1e10,
        space_logits,
    )

    # insert replacement
    # this probably "should" be a tf.scatter_nd
    # but i'm not ready to spend 5 hours again trying to understand its call signature :P
    logits = tf.concat(
        [
            logits[:, :space_token],
            space_logits[:, tf.newaxis],
            logits[:, space_token + 1 :]
        ],
        axis=1
    )

    return logits


def sample_sequence(
    *,
    hparams,
    length,
    start_token=None,
    batch_size=None,
    context=None,
    temperature=1,
    top_k=0,
    top_p=0.0,
    middle_p=0.0,
    chop_lowest=None,
    chop_highest=None,
    epsilon=-1e10,
    stop_at_EOT=False,
    eot_workaround=False,
    better_length=True,
    enc=None,
    steps_per_print=10,
    prints_per_newline=18,
    swap_memory=True,
    return_presents=False,
    pasts=None,
    mirostat=False,
    mirostat_surprise_target=3.0,
    mirostat_mu_init=None,
    mirostat_lr=1.0,
    mirostat_trunc=50000,
    mirostat_mu_lower_clip=0.0,
    mirostat_v2=False,
    disable_prints=False,
    avoid_tag_only_posts=True,
    breakruns=False,
    breakruns_tau=0.05,
):
    if breakruns or mirostat_mu_init is None:
        mu_init_scale = 0. if breakruns else (1.0 if mirostat_v2 else 2.0)
        mirostat_mu_init = tf.tile(
            [mu_init_scale * mirostat_surprise_target], [batch_size]
        )
    mirostat_mus_init = mirostat_mu_init[:, tf.newaxis]
    mirostat_ks_init = tf.zeros_like(mirostat_mus_init, dtype=tf.int32)
    mirostat_ss_init = tf.zeros_like(mirostat_mus_init, dtype=tf.float32)
    mirostat_surprises_init = tf.zeros_like(mirostat_mus_init, dtype=tf.float32)

    mirostat_logits_fn = mirostat_logits_v2 if mirostat_v2 else mirostat_logits

    if mirostat_v2:
        mirostat_trunc = hparams.n_vocab

    if start_token is None:
        assert context is not None, "Specify exactly one of start_token and context!"
    else:
        assert context is None, "Specify exactly one of start_token and context!"
        context = tf.fill([batch_size, 1], start_token)

    if enc is not None:
        EOT_TOKEN = enc.encode("<|endoftext|>")[0]
        print(f"EOT_TOKEN: {EOT_TOKEN}")

        if not eot_workaround:
            EOT_TOKEN2 = enc.encode("<|endoftext|>")[1]
            print(f"EOT_TOKEN2: {EOT_TOKEN2}")

        NEWLINE_TOKEN = enc.encode("\n")[-1]
        SPACE_TOKEN = enc.encode(" ")[-1]

    elif stop_at_EOT:
        raise ValueError("must supply enc when stop_at_EOT=True")

    # count initial newlines
    initial_newline_mask = tf.equal(
        tf.dtypes.cast(context, tf.int32),
        NEWLINE_TOKEN
    )  # [batchsize, ntok]

    initial_newline_count = tf.reduce_sum(
        tf.dtypes.cast(initial_newline_mask, tf.int32),
        axis=-1
    )  # [batchsize,]

    def step(hparams, tokens, past=None, past_adapt=None):
        lm_output = model.model(
            hparams=hparams,
            X=tokens,
            past=past,
            past_adapt=past_adapt,
            reuse=tf.AUTO_REUSE,
        )
        if hparams.dtype != tf.float32:
            lm_output["logits"] = tf.cast(lm_output["logits"], tf.float32)

        logits = lm_output["logits"][:, :, : hparams.n_vocab]
        presents = lm_output["present"]
        presents.set_shape(model.past_shape(hparams=hparams, batch_size=batch_size))

        if hparams.use_adapters and hparams.attn_adapters:
            print("attn_adapters in sample_sequence")
            presents_adapt = lm_output["present_adapt"]
            presents_adapt.set_shape(
                model.past_shape_adapt(hparams=hparams, batch_size=batch_size)
            )

            return {
                "logits": logits,
                "presents": presents,
                "presents_adapt": presents_adapt,
            }
        print("no attn_adapters in sample_sequence")
        return {
            "logits": logits,
            "presents": presents,
        }

    with tf.name_scope("sample_sequence"):
        # Don't feed the last context token -- leave that to the loop below
        # TODO: Would be slightly faster if we called step on the entire context,
        # rather than leaving the last token transformer calculation to the while loop.
        if pasts is None:
            context_output = step(hparams, context[:, :-1])
            pasts = context_output["presents"]

        def body(
            past,
            prev,
            output,
            mirostat_mu,
            mirostat_mus,
            mirostat_ks,
            mirostat_ss,
            mirostat_surprises,
        ):
            if disable_prints:
                next_outputs = step(hparams, prev[:, tf.newaxis], past=past)
            else:
                steps_so_far = tf.reduce_sum(tf.ones_like(output), 1)[0]
                print_with_newline_op = tf.print(steps_so_far, end="...\n")
                print_no_newline_op = tf.print(steps_so_far, end="...")
                print_nothing_op = tf.print("", end="")

                print_something_op = tf.cond(
                    tf.equal(
                        tf.mod(steps_so_far, steps_per_print * prints_per_newline), 0
                    ),
                    lambda: print_with_newline_op,
                    lambda: print_no_newline_op,
                )

                print_op = tf.cond(
                    tf.equal(tf.mod(steps_so_far, steps_per_print), 0),
                    lambda: tf.cond(
                        tf.equal(
                            tf.mod(steps_so_far, steps_per_print * prints_per_newline),
                            0,
                        ),
                        lambda: tf.print(steps_so_far, end="...\n"),
                        lambda: tf.print(steps_so_far, end="..."),
                    ),
                    lambda: tf.print("", end=""),
                )

                with tf.control_dependencies([print_op]):
                    next_outputs = step(hparams, prev[:, tf.newaxis], past=past)

            # if mirostat:
            #     logits = next_outputs["logits"][:, -1, :]
            # else:
            #     logits = next_outputs["logits"][:, -1, :] / tf.to_float(temperature)
            #
            #     probs_orig = tf.nn.softmax(logits)
            #     next_mirostat_mu = mirostat_mus[:, -1]
            #     k_ms = mirostat_ks[:, -1]
            #     s_ms = mirostat_ss[:, -1]

            if breakruns:
              eff_temperature = temperature + breakruns_tau * mirostat_mu
            else:
              eff_temperature = temperature

            logits = next_outputs["logits"][:, -1, :] / tf.to_float(eff_temperature)

            if avoid_tag_only_posts:
                logits = apply_avoid_tag_only_posts(
                    logits,
                    prev,
                    output,
                    initial_newline_count=initial_newline_count,
                    newline_token=NEWLINE_TOKEN,
                    space_token=SPACE_TOKEN)

            # if mirostat:
            ixs = np.arange(1, mirostat_trunc + 2, 1, dtype=float)
            mirostat_t = np.log(ixs[1:] / ixs[:-1])
            mirostat_t_sq_sum = (mirostat_t ** 2).sum()
            mirostat_harmonic = 1.0 / ixs[:-1]

            logits_mirostat, probs_orig, k_ms, s_ms = mirostat_logits_fn(
                logits,
                mirostat_surprise_target,
                mirostat_mu,
                mirostat_t,
                mirostat_t_sq_sum,
                mirostat_harmonic,
                mirostat_trunc,
            )
            if mirostat:
                logits = logits_mirostat
            elif middle_p > 0.0:
                logits = midde_p_logits(logits, p_base=middle_p)
            elif chop_lowest is not None and chop_highest is not None:
                logits = midde_p_logits(
                    logits, chop_lowest=chop_lowest, chop_highest=chop_highest
                )
            elif top_p > 0.0:
                logits = top_p_logits(logits, p=top_p, epsilon=epsilon)
            else:
                logits = top_k_logits(logits, k=top_k, epsilon=epsilon)
            samples = tf.multinomial(logits, num_samples=1, output_dtype=tf.int32)

            probs_of_selected = tf.squeeze(
                tf.gather(probs_orig, samples, axis=1, batch_dims=1), axis=[-1]
            )
            surprise_of_selected = tf.log(1.0 / probs_of_selected) / np.log(2)
            error_surprise = surprise_of_selected - mirostat_surprise_target
            if breakruns:
              bump = tf.reduce_all(probs_orig <= probs_of_selected[:, tf.newaxis], axis=-1)
              bump_fl = tf.cast(bump, tf.float32)
              next_mirostat_mu = bump_fl*(mirostat_mu+1)
            else:
              next_mirostat_mu = mirostat_mu - mirostat_lr * error_surprise
              next_mirostat_mu = tf.maximum(next_mirostat_mu, mirostat_mu_lower_clip)

            return [
                tf.concat([past, next_outputs["presents"]], axis=-2),
                tf.squeeze(samples, axis=[1]),
                tf.concat([output, samples], axis=1),
                next_mirostat_mu,
                tf.concat([mirostat_mus, next_mirostat_mu[:, tf.newaxis]], axis=-1),
                tf.concat([mirostat_ks, k_ms[:, tf.newaxis]], axis=-1),
                tf.concat([mirostat_ss, s_ms[:, tf.newaxis]], axis=-1),
                tf.concat(
                    [mirostat_surprises, surprise_of_selected[:, tf.newaxis]], axis=-1
                ),
            ]

        def cond(*args):
            if better_length:
                length_cond = tf.reduce_sum(tf.ones_like(args[2]), 1)[0] < length
            else:
                length_cond = True

            if stop_at_EOT:
                print("stop_at_EOT in cond")
                if eot_workaround:
                    eot_cond = tf.logical_not(
                        tf.reduce_all(
                            tf.reduce_any(
                                tf.equal(
                                    tf.dtypes.cast(args[2][:, 1:], tf.int32), EOT_TOKEN
                                ),
                                axis=1,
                            ),
                            axis=0,
                        )
                    )
                else:
                    eot_cond = tf.logical_not(
                        tf.reduce_all(
                            tf.reduce_any(
                                tf.logical_and(
                                    tf.equal(
                                        tf.dtypes.cast(args[2][:, :-1], tf.int32),
                                        EOT_TOKEN,
                                    ),
                                    tf.equal(
                                        tf.dtypes.cast(args[2][:, 1:], tf.int32),
                                        EOT_TOKEN2,
                                    ),
                                ),
                                axis=1,
                            ),
                            axis=0,
                        )
                    )
            else:
                eot_cond = True

            return tf.logical_and(length_cond, eot_cond)

        print(f"using swap_memory={swap_memory}")
        (
            presents,
            _,
            tokens,
            mirostat_mu_final,
            mirostat_mus,
            mirostat_ks,
            mirostat_ss,
            mirostat_surprises,
        ) = tf.while_loop(
            cond=cond,
            body=body,
            maximum_iterations=length if not better_length else None,
            loop_vars=[
                pasts,
                context[:, -1],
                context,
                mirostat_mu_init,
                mirostat_mus_init,
                mirostat_ks_init,
                mirostat_ss_init,
                mirostat_surprises_init,
            ],
            shape_invariants=[
                tf.TensorShape(
                    model.past_shape(hparams=hparams, batch_size=batch_size)
                ),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
                tf.TensorShape([batch_size, None]),
            ],
            back_prop=False,
            swap_memory=swap_memory,
        )

        retval = {
            "tokens": tokens,
            "mirostat_mus": mirostat_mus,
            "mirostat_ks": mirostat_ks,
            "mirostat_ss": mirostat_ss,
            "mirostat_surprises": mirostat_surprises,
        }
        if return_presents:
            retval["presents"] = presents
        return retval
