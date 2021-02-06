#!/usr/bin/env python3
# Usage:
#  PYTHONPATH=src ./train --dataset <file|directory|glob>
import os
import sys

sys.path += [os.path.join(os.path.dirname(os.path.abspath(__file__)), "src")]

import argparse
import json
import numpy as np
import tensorflow as tf
import time
import tqdm
from tensorflow.core.protobuf import rewriter_config_pb2
from tensorflow.python import pywrap_tensorflow

from tensorflow.python.framework import ops
from tensorflow.python.tpu.ops import tpu_ops

import model
import sample
import encoder
from load_dataset import load_dataset, Sampler, TextSampler, RobSamplerInterface
from accumulate import GradientAccumulator

# import memory_saving_gradients
from glob import glob
import re
import tflex
import tflex_sgdr

import pytz
from datetime import datetime, timezone

CHECKPOINT_DIR = "checkpoint"
SAMPLE_DIR = "samples"


parser = argparse.ArgumentParser(
    description="Fine-tune GPT-2 on your custom dataset.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--dataset",
    metavar="PATH",
    type=str,
    required=True,
    help="Input file, directory, or glob pattern (utf-8 text, or preencoded .npz files).",
)
parser.add_argument(
    "--model_name",
    metavar="MODEL",
    type=str,
    default="117M",
    help="Pretrained model name",
)
parser.add_argument(
    "--combine",
    metavar="CHARS",
    type=int,
    default=50000,
    help="Concatenate input files with <|endoftext|> separator into chunks of this minimum size",
)

parser.add_argument(
    "--batch_size", metavar="SIZE", type=int, default=1, help="Batch size"
)
parser.add_argument(
    "--learning_rate",
    metavar="LR",
    type=float,
    default=0.00002,
    help="Learning rate for Adam",
)
parser.add_argument(
    "--learning_rate_min", type=float, default=0.00001, help="Minimum learning rate"
)
parser.add_argument(
    "--learning_rate_cos",
    default=False,
    action="store_true",
    help="Use learn rate cosine annealing",
)
parser.add_argument(
    "--learning_rate_warmup",
    type=int,
    default=100,
    help="Learning rate warmup for cosine annealing",
)
parser.add_argument(
    "--learning_rate_period",
    type=int,
    default=100,
    help="Learning rate period for cosine annealing",
)
parser.add_argument(
    "--learning_rate_initial_step",
    type=int,
    default=0,
    help="Learning rate initial step for cosine annealing",
)
parser.add_argument(
    "--learning_rate_m_mul",
    type=float,
    default=1.0,
    help="Learning rate multiplier each reset",
)
parser.add_argument(
    "--accumulate_gradients",
    metavar="N",
    type=int,
    default=1,
    help="Accumulate gradients across N minibatches.",
)
parser.add_argument(
    "--memory_saving_gradients",
    default=False,
    action="store_true",
    help="Use gradient checkpointing to reduce vram usage.",
)
parser.add_argument(
    "--only_train_transformer_layers",
    default=False,
    action="store_true",
    help="Restrict training to the transformer blocks.",
)
parser.add_argument(
    "--optimizer", type=str, default="adam", help="Optimizer. <adam|sgd|ada>."
)
parser.add_argument(
    "--noise",
    type=float,
    default=0.0,
    help="Add noise to input training data to regularize against typos.",
)
parser.add_argument(
    "--adam_beta1", metavar="BETA1", type=float, default=0.9, help="beta1 for Adam"
)
parser.add_argument(
    "--adam_beta2", metavar="BETA2", type=float, default=0.999, help="beta2 for Adam"
)
parser.add_argument(
    "--adam_epsilon",
    metavar="EPSILON",
    type=float,
    default=1e-8,
    help="epsilon for Adam",
)
parser.add_argument("--only_train_layers_before", type=int, default=-1)
parser.add_argument("--only_train_layers_after", type=int, default=-1)

parser.add_argument("--top_k", type=int, default=40, help="K for top-k sampling.")
parser.add_argument(
    "--top_p",
    type=float,
    default=0.0,
    help="P for top-p sampling. Overrides top_k if set > 0.",
)

parser.add_argument(
    "--restore_from",
    type=str,
    default="latest",
    help='Either "latest", "fresh", or a path to a checkpoint file',
)
parser.add_argument(
    "--run_name",
    type=str,
    default="run1",
    help="Run id. Name of subdirectory in checkpoint/ and samples/",
)
parser.add_argument(
    "--sample_every",
    metavar="N",
    type=int,
    default=0,
    help="Generate samples every N steps",
)
parser.add_argument(
    "--sample_length",
    metavar="TOKENS",
    type=int,
    default=-1,
    help="Sample this many tokens",
)
parser.add_argument(
    "--sample_num", metavar="N", type=int, default=1, help="Generate this many samples"
)
parser.add_argument(
    "--save_every",
    metavar="N",
    type=int,
    default=-1,
    help="Write a checkpoint every N steps",
)
parser.add_argument(
    "--save_time",
    metavar="N",
    type=float,
    default=15.0,
    help="Write a checkpoint every N minutes",
)
parser.add_argument(
    "--max_to_keep",
    metavar="N",
    type=int,
    default=5,
    help="Only keep the last N checkpoints",
)

parser.add_argument(
    "--val_dataset",
    metavar="PATH",
    type=str,
    default=None,
    help="Dataset for validation loss, defaults to --dataset.",
)
parser.add_argument(
    "--val_batch_size",
    metavar="SIZE",
    type=int,
    default=0,
    help="Batch size for validation.",
)
parser.add_argument(
    "--val_batch_count",
    metavar="N",
    type=int,
    default=0,
    help="Number of batches for validation.",
)
parser.add_argument(
    "--val_every",
    metavar="STEPS",
    type=int,
    default=0,
    help="Calculate validation loss every STEPS steps.",
)

parser.add_argument(
    "--init_tpu", default=False, action="store_true", help="Initialize TPU session."
)

parser.add_argument(
    "--fresh_model",
    default=False,
    action="store_true",
    help="Don't load model from disk; initialize model weights to random values",
)
parser.add_argument(
    "--save_on_ctrlc",
    default=False,
    action="store_true",
    help="When execution is interrupted, should we save the model to disk?",
)
parser.add_argument(
    "--debug_on_ctrlc",
    default=False,
    action="store_true",
    help="When execution is interrupted, attach a debugger (pdb.set_trace())",
)
parser.add_argument(
    "--float16", default=False, action="store_true", help="Use float16 weights?"
)
parser.add_argument(
    "--dtype", type=str, default="float32", help="dtype. <float32|float16|bfloat16>."
)
parser.add_argument(
    "--eot_workaround", default=False, action="store_true", help="Handle EOT properly"
)
parser.add_argument(
    "--rob_sampler", default=False, action="store_true", help="Use Rob's sampler"
)
parser.add_argument(
    "--noise_scale",
    default=False,
    action="store_true",
    help="Estimate gradient noise scale",
)

parser.add_argument(
    "--n_ctx",
    type=int,
    default=-1,
    help="For a fresh model, how large should n_ctx be?",
)
parser.add_argument(
    "--n_embd",
    type=int,
    default=-1,
    help="For a fresh model, how large should n_embd be?",
)
parser.add_argument(
    "--n_head",
    type=int,
    default=-1,
    help="For a fresh model, how large should n_head be?",
)
parser.add_argument(
    "--n_layer",
    type=int,
    default=-1,
    help="For a fresh model, how large should n_layer be?",
)

parser.add_argument(
    "--sample_ctx",
    type=int,
    default=-1,
    help="Compute loss over N samples. Equal to n_ctx if set < 0.",
)

parser.add_argument(
    "--truncate_weights",
    default=False,
    action="store_true",
    help="Try loading variables from snapshots, even if those variables' shapes do not match",
)

parser.add_argument(
    "--debug_print_all_vars",
    default=False,
    action="store_true",
    help="Print all variables after running one training step",
)
parser.add_argument(
    "--debug_print_trainable_vars",
    default=False,
    action="store_true",
    help="Print trainable variables after running one training step",
)

parser.add_argument(
    "--allow_growth",
    default=False,
    action="store_true",
    help="Set config.gpu_options.allow_growth = True",
)
parser.add_argument(
    "--disable_layout_optimizer",
    default=False,
    action="store_true",
    help="Set config.graph_options.rewrite_options.layout_optimizer = rewriter_config_pb2.RewriterConfig.OFF",
)

parser.add_argument(
    "--debug_before_training",
    default=False,
    action="store_true",
    help="Drop into debugger before starting the training loop",
)

parser.add_argument("--res_dropout", type=float, default=0.0, help="res_dropout")
parser.add_argument("--attn_dropout", type=float, default=0.0, help="attn_dropout")
parser.add_argument("--acti_dropout", type=float, default=0.0, help="acti_dropout")

parser.add_argument(
    "--seed",
    type=int,
    default=-1,
    help="Deterministic seed for dataset sampler. Disabled if set < 0",
)

parser.add_argument(
    "--save_graph",
    default=False,
    action="store_true",
    help="Save TensorFlow graph to summary log (to see ops in tensorboard)",
)

parser.add_argument(
    "--cross_shard_opt",
    default=False,
    action="store_true",
    help="Use CrossShardOptimizer on TPU",
)

parser.add_argument(
    "--use_adapters", default=False, action="store_true", help="Use adapters"
)
parser.add_argument(
    "--attn_adapters", default=False, action="store_true", help="Use adapters"
)
parser.add_argument(
    "--mlp_adapters", default=False, action="store_true", help="Use adapters"
)
parser.add_argument(
    "--n_adapt_attn", type=int, default=128, help="Adapter down proj size"
)
parser.add_argument(
    "--nhead_adapt_attn", type=int, default=25, help="Adapter down proj size"
)
parser.add_argument(
    "--n_adapt_mlp", type=int, default=128, help="Adapter down proj size"
)
parser.add_argument(
    "--attn_adapters_share_proj",
    default=False,
    action="store_true",
    help="Use adapters",
)

parser.add_argument(
    "--save_optimizer", default=False, action="store_true", help="Save opt state"
)
parser.add_argument(
    "--avoid_load_optimizer", default=False, action="store_true", help="don't opt state"
)
parser.add_argument(
    "--avg_loss_beta",
    type=float,
    default=0.99,
    help="Decay param for moving avg loss printed to console",
)
parser.add_argument(
    "--gs_sync", default=False, action="store_true", help="Save checkpoints to gs"
)

PST = pytz.timezone("US/Pacific")


def timestamp(now=None, tz=None):
    if now is None:
        now = datetime.now(timezone.utc)
    if tz is None:
        tz = PST
    return "{}".format(now.astimezone(tz).isoformat())


def maketree(path):
    try:
        os.makedirs(path)
    except:
        pass


def randomize(context, hparams, p):
    if p > 0:
        mask = tf.random.uniform(shape=tf.shape(context)) < p
        noise = tf.random.uniform(
            shape=tf.shape(context), minval=0, maxval=hparams.n_vocab, dtype=tf.int32
        )
        return tf.where(mask, noise, context)
    else:
        return context


def main():
    try:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
            tpu="grpc://" + os.environ["COLAB_TPU_ADDR"]
        )
    except:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(resolver)
    topology = tf.tpu.experimental.initialize_tpu_system(resolver)

    args = parser.parse_args()
    enc = encoder.get_encoder(args.model_name, eot_workaround=args.eot_workaround)
    hparams = model.default_hparams()
    hparams.acti_dropout = args.acti_dropout
    hparams.attn_dropout = args.attn_dropout
    hparams.res_dropout = args.res_dropout
    epsilon = -1e10
    if args.dtype == "float32":
        hparams.dtype = tf.float32
    elif args.dtype == "float16":
        hparams.dtype = tf.float16
        epsilon = -65500
    elif args.dtype == "bfloat16":
        hparams.dtype = tf.bfloat16
        epsilon = -65500
    else:
        print("Unknown dtype", args.dtype)
    if args.float16:
        hparams.dtype = tf.bfloat16
        epsilon = -65500

    with open(os.path.join("models", args.model_name, "hparams.json")) as f:
        hparams.override_from_dict(json.load(f))
    if args.n_ctx >= 0:
        hparams.n_ctx = args.n_ctx
    if args.n_embd >= 0:
        hparams.n_embd = args.n_embd
    if args.n_head >= 0:
        hparams.n_head = args.n_head
    if args.n_layer >= 0:
        hparams.n_layer = args.n_layer
    hparams.use_adapters = args.use_adapters
    hparams.attn_adapters = args.attn_adapters
    hparams.mlp_adapters = args.mlp_adapters
    hparams.n_adapt_attn = args.n_adapt_attn
    hparams.nhead_adapt_attn = args.nhead_adapt_attn
    hparams.n_adapt_mlp = args.n_adapt_mlp
    hparams.attn_adapters_share_proj = args.attn_adapters_share_proj
    if hparams.use_adapters:
        hparams.orth_init = True

    if args.sample_length < 0:
        args.sample_length = hparams.n_ctx - 1
    if args.sample_length > hparams.n_ctx:
        raise ValueError(
            "Can't get samples longer than window size: %s" % hparams.n_ctx
        )
    if args.sample_ctx < 0:
        args.sample_ctx = hparams.n_ctx

    if args.model_name == "345M":
        args.memory_saving_gradients = True
        if args.optimizer == "adam":
            args.only_train_transformer_layers = True

    if args.use_adapters:
        args.only_train_transformer_layers = False

    config = tf.ConfigProto()
    if args.allow_growth:
        config.gpu_options.allow_growth = True
    if args.disable_layout_optimizer:
        config.graph_options.rewrite_options.layout_optimizer = (
            rewriter_config_pb2.RewriterConfig.OFF
        )
    with tflex.Session(config=config, init_tpu=args.init_tpu) as sess:
        context = tf.placeholder(tf.int32, [args.batch_size, None])
        context_in = randomize(context, hparams, args.noise)
        output = model.model(hparams=hparams, X=context_in)

        if args.val_every > 0:
            def tpufn_val_base(tpufn_val_in):
                output_ = model.model(hparams=hparams, X=tpufn_val_in)
                scale = 1.0 / NUM_REPLICAS
                loss_ = scale * tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=tpufn_val_in[:, 1:], logits=output_["logits"][:, :-1]
                    )
                )

                with ops.colocate_with(loss_):
                    summed_loss_ = tpu_ops.cross_replica_sum(loss_)
                print(("summed_loss_", summed_loss_))
                return summed_loss_

            tpufn_val = tpufn_val_base

        def tpufn(tpufn_in):
            output_ = model.model(hparams=hparams, X=tpufn_in)
            scale = 1.0 / NUM_REPLICAS
            loss_ = scale * tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=tpufn_in[:, 1:], logits=output_["logits"][:, :-1]
                )
            )

            grads_ = tf.gradients(loss_, train_vars)

            summed_grads_ = []
            for grad in grads_:
                if grad is None:
                    summed_grads_.append(grad)
                else:
                    with ops.colocate_with(grad):
                        summed_grads_.append(tpu_ops.cross_replica_sum(grad))

            with ops.colocate_with(loss_):
                summed_loss_ = tpu_ops.cross_replica_sum(loss_)
            return summed_grads_ + [summed_loss_]

        with tf.variable_scope(tf.get_variable_scope().name, reuse=tf.AUTO_REUSE):
            global_step = tflex.get_variable("global_step") or tf.get_variable(
                "global_step", shape=(), dtype=tf.int32, trainable=False
            )
            current_step = args.learning_rate_initial_step
            global_step.load(current_step, session=sess)
            if args.learning_rate_cos:
                lr = tflex_sgdr.sgdr_decay_with_warmup(
                    args.learning_rate,
                    global_step,
                    warmup_steps=args.learning_rate_warmup,
                    initial_period_steps=args.learning_rate_period,
                    learning_rate_min=args.learning_rate_min,
                    m_mul=args.learning_rate_m_mul,
                )
            else:
                lr = tflex.get_variable("learn_rate") or tf.get_variable(
                    "learn_rate", shape=(), dtype=tf.float32, trainable=False
                )
                lr.load(args.learning_rate, session=sess)

        def update_lr(rate=None, step=None):
            if not args.learning_rate_cos:
                if step is None:
                    step = global_step.eval(session=sess)
                if rate is None:
                    rate = args.learning_rate
                if callable(rate):
                    rate = rate(step)
                lr.load(rate, session=sess)
            return lr.eval(session=sess)

        @tflex.register_command
        def set_learning_rate():
            print("Current learn rate: %0.8f" % update_lr())
            print("New learn rate?")
            rate = input("")
            if not rate:
                print("Empty input; not changing anything.")
            else:
                try:
                    rate = float(rate)
                except:
                    print("Invalid input; must be a float")
            print("Setting learn rate to %0.8f" % rate)
            args.learning_rate = rate

        if args.optimizer == "adam":
            opt = tf.train.AdamOptimizer(
                learning_rate=lr,
                beta1=args.adam_beta1,
                beta2=args.adam_beta2,
                epsilon=args.adam_epsilon,
            )
        elif args.optimizer == "sgd":
            opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
        elif args.optimizer == "adafactor":
            from tensor2tensor.utils.adafactor import AdafactorOptimizer

            opt = AdafactorOptimizer()
        elif args.optimizer == "yellowfin":
            from yellowfin import YFOptimizer

            opt = YFOptimizer()
        elif args.optimizer == "novograd":
            from novograd import NovoGrad

            opt = NovoGrad(
                learning_rate=lr,
                beta1=args.adam_beta1,
                beta2=args.adam_beta2,
                epsilon=args.adam_epsilon,
            )
        else:
            exit("Bad optimizer:", args.optimizer)

        COMPU_SHAPE = [1, 1, 1]
        NUM_REPLICAS = 8
        device_assignment = tf.tpu.experimental.DeviceAssignment.build(
            topology,
            computation_shape=COMPU_SHAPE,
            num_replicas=NUM_REPLICAS,
        )

        min_layer = args.only_train_layers_after
        max_layer = (
            hparams.n_layer + 1
            if args.only_train_layers_before < 0
            else args.only_train_layers_before
        )

        print(args.only_train_layers_before, args.only_train_layers_after)
        print(min_layer, max_layer)

        allowed_substrings = [
            f"/h{i}/"
            for i in range(hparams.n_layer)
            if (i > min_layer) and (i < max_layer)
        ]
        if not args.only_train_transformer_layers:
            allowed_substrings.append("/wte/")

        print(allowed_substrings)

        all_vars = [v for v in tf.trainable_variables() if "model" in v.name]
        if args.use_adapters:
            train_vars = [v for v in all_vars if "adapt" in v.name]
        else:
            train_vars = [
                v for v in all_vars if any([s in v.name for s in allowed_substrings])
            ]

        layer_names = ["wte", "wpe"] + [f"/h{i}/" for i in range(hparams.n_layer)]
        layer_params = [
            [v for v in train_vars if name in v.name] for name in layer_names
        ]

        layer_param_norms = [
            (name, tf.global_norm(vars))
            for name, vars in zip(layer_names, layer_params)
            if len(vars) > 0
        ]

        layer_param_norm_names = [t[0] for t in layer_param_norms]
        layer_param_norm_vars = [t[1] for t in layer_param_norms]

        parameter_count = sum([np.prod(v.shape.as_list()) for v in train_vars])
        print(
            "This model is using %d parameters (%.2fM)"
            % (parameter_count, parameter_count / (1024.0 * 1024.0))
        )

        out_raw = tf.tpu.shard(
            tpufn,
            inputs=[
                context_in,
            ],
            num_shards=NUM_REPLICAS,
            device_assignment=device_assignment,
            outputs_from_all_shards=False,
        )
        out = out_raw[:-1]
        out_loss = out_raw[-1]

        out_reduced = out

        out_reduced_reshaped = [
            tf.reshape(o, v.shape.as_list()) for o, v in zip(out_reduced, train_vars)
        ]

        out_loss_reduced = out_loss
        opt_grads = list(zip(out_reduced_reshaped, train_vars))

        if args.accumulate_gradients > 1:
            if False:  # args.no_loss:
                raise ValueError(
                    "accumulate_gradients with no_loss not implemented yet"
                )
            # TODO: work with no_loss
            gradient_accumulator = GradientAccumulator(
                var_list=train_vars, noise_scale_estimates=args.noise_scale
            )
            opt_reset = gradient_accumulator.reset()
            opt_add_gradients = gradient_accumulator.add_gradients(
                out_loss_reduced, opt_grads
            )
            opt_apply = gradient_accumulator.apply_gradients(opt)

            print(
                (
                    "gradient_accumulator.noise_scale_estimates",
                    gradient_accumulator.noise_scale_estimates,
                )
            )
        else:
            if args.noise_scale:
                raise ValueError(
                    f"noise_scale only supported via gradient accumulation"
                )
            opt_apply = opt.apply_gradients(opt_grads)

        print(("opt_apply", opt_apply))

        if args.sample_every > 0:
            tf_sample = sample.sample_sequence(
                hparams=hparams,
                length=args.sample_length,
                context=context,
                batch_size=min(args.sample_num, args.batch_size),
                temperature=1.0,
                top_k=args.top_k,
                top_p=args.top_p,
                epsilon=epsilon,
                disable_prints=True,
            )["tokens"]

        summary_lr = tf.summary.scalar("learning_rate", lr)

        summary_log = tf.summary.FileWriter(os.path.join(CHECKPOINT_DIR, args.run_name))

        if args.save_graph:
            summary_log.add_graph(tf.get_default_graph())

        opt_vars = []
        if args.save_optimizer:
            opt_vars = opt.variables()
        saver = tflex.Saver(
            var_list=all_vars,
            max_to_keep=args.max_to_keep,
            keep_checkpoint_every_n_hours=200000,
            reshape=args.truncate_weights,
        )
        saver_base_vars_only = tflex.Saver(
            var_list=[v for v in all_vars if "adapt" not in v.name],
            max_to_keep=args.max_to_keep,
            keep_checkpoint_every_n_hours=200000,
            reshape=args.truncate_weights,
        )
        saver_train_vars_only = tflex.Saver(
            var_list=train_vars,
            max_to_keep=args.max_to_keep,
            keep_checkpoint_every_n_hours=200000,
            reshape=args.truncate_weights,
        )
        saver_opt = tflex.Saver(
            var_list=opt_vars,
            max_to_keep=args.max_to_keep,
            keep_checkpoint_every_n_hours=200000,
            reshape=args.truncate_weights,
        )
        sess.run(tf.global_variables_initializer())

        if args.restore_from == "latest":
            ckpt = tflex.latest_checkpoint(os.path.join(CHECKPOINT_DIR, args.run_name))
            if ckpt is None:
                # Get fresh GPT weights if new run.
                ckpt = tflex.latest_checkpoint(os.path.join("models", args.model_name))
        elif args.restore_from == "fresh":
            ckpt = tflex.latest_checkpoint(os.path.join("models", args.model_name))
        else:
            ckpt = tflex.latest_checkpoint(args.restore_from)
        print("Loading snapshot %s..." % ckpt)
        t0 = time.time()
        if not args.fresh_model:
            if args.use_adapters:
                base_ckpt = tflex.latest_checkpoint(
                    os.path.join("models", args.model_name)
                )
                saver_base_vars_only.restore(sess, base_ckpt)

                ckpt = tflex.latest_checkpoint(
                    os.path.join(CHECKPOINT_DIR, args.run_name)
                )
                if ckpt is not None:
                    print(ckpt)
                    saver_train_vars_only.restore(sess, ckpt)
            else:
                saver.restore(sess, ckpt)

                ckpt_opt = None if ckpt is None else ckpt.replace("model", "opt")
                if ckpt_opt is not None and not (
                    os.path.exists(ckpt_opt) or os.path.exists(ckpt_opt + ".hdf5")
                ):
                    print(
                        f"constructed ckpt_opt path {repr(ckpt_opt)}, but it doesn't exist"
                    )
                    ckpt_opt = None
                if (not args.avoid_load_optimizer) and (ckpt_opt is not None):
                    print(f"restoring optimizer from {ckpt_opt}")
                    saver_opt.restore(
                        sess,
                        ckpt_opt,
                    )
                    for _name, _v in opt._non_slot_dict.items():
                        print((_name, _v.eval(sess)))
                else:
                    print("using fresh optimizer")
        t1 = time.time()
        print("Loaded in %f seconds" % (t1 - t0))

        def make_sampler(dataset, enc, seed, combine):
            if os.path.isdir(dataset) or dataset.endswith(".npz"):
                chunks = load_dataset(enc, dataset, combine)
                if args.rob_sampler:
                    ckpt = tflex.latest_checkpoint(
                        os.path.join(CHECKPOINT_DIR, args.run_name)
                    )
                    ckpt_sampler = (
                        None
                        if ckpt is None
                        else ckpt.replace("model", "sampler") + ".json"
                    )
                    if ckpt_sampler is not None:
                        print(f"looking for ckpt_sampler {ckpt_sampler}")
                        if os.path.exists(ckpt_sampler):
                            print("found it, loading")
                            data_sampler = RobSamplerInterface.load(
                                chunks, path=ckpt_sampler
                            )
                        else:
                            print("did not find it, making fresh sampler")
                            data_sampler = RobSamplerInterface(chunks, seed=seed)
                    else:
                        print("no checkpoints, making fresh sampler")
                        data_sampler = RobSamplerInterface(chunks, seed=seed)
                else:
                    data_sampler = Sampler(chunks, seed=seed)
                print(
                    "dataset has",
                    data_sampler.total_size,
                    "tokens",
                    len(chunks),
                    "chunks",
                )
            else:
                data_sampler = TextSampler(dataset, enc, seed=seed)
            return data_sampler

        print("Loading dataset...")
        seed = None if args.seed < 0 else args.seed
        data_sampler = make_sampler(
            dataset=args.dataset, enc=enc, seed=seed, combine=args.combine
        )
        if args.val_every > 0:
            # Sample from validation set once with fixed seed to make
            # it deterministic during training as well as across runs.
            val_dataset = args.val_dataset if args.val_dataset else args.dataset
            val_data_sampler = make_sampler(
                dataset=val_dataset, enc=enc, seed=1, combine=args.combine
            )

            val_batch_size = args.val_batch_size
            if val_batch_size == 0:
                val_batch_size = args.batch_size

            val_batch_count = args.val_batch_count
            if args.val_batch_count == 0:
                val_batch_count = val_data_sampler.total_size // (
                    hparams.n_ctx * val_batch_size
                )
                print(
                    f"val_batch_count: {val_data_sampler.total_size} // ({hparams.n_ctx} * {val_batch_size}) = {val_batch_count}"
                )

            val_batches = [
                [val_data_sampler.sample(hparams.n_ctx) for _ in range(val_batch_size)]
                for _ in range(val_batch_count)
            ]

            val_loss = tf.tpu.shard(
                tpufn_val,
                inputs=[
                    context_in,
                ],
                num_shards=NUM_REPLICAS,
                device_assignment=device_assignment,
                outputs_from_all_shards=False,
            )[0]

        print("Training...")
        counter = 1
        counter_path = os.path.join(CHECKPOINT_DIR, args.run_name, "counter")
        if os.path.exists(counter_path):
            # Load the step number if we're resuming a run
            # Add 1 so we don't immediately try to save again
            with open(counter_path, "r") as fp:
                counter = int(fp.read()) + 1

        @tflex.register_command
        def save():
            maketree(os.path.join(CHECKPOINT_DIR, args.run_name))

            if args.rob_sampler:
                ckpt_sampler = (
                    os.path.join(CHECKPOINT_DIR, args.run_name, "sampler-{}").format(
                        counter
                    )
                    + ".json"
                )
                print("Saving", ckpt_sampler)
                data_sampler.save(ckpt_sampler)

            print(
                "Saving",
                os.path.join(CHECKPOINT_DIR, args.run_name, "model-{}").format(counter),
            )
            t0 = time.time()
            saver_to_use = saver_train_vars_only if args.use_adapters else saver
            saver_to_use.save(
                sess,
                os.path.join(CHECKPOINT_DIR, args.run_name, "model"),
                global_step=counter,
            )
            if args.save_optimizer:
                saver_opt.save(
                    sess,
                    os.path.join(CHECKPOINT_DIR, args.run_name, "opt"),
                    global_step=counter,
                )
            t1 = time.time()
            print("Saved in %f seconds" % (t1 - t0))
            with open(counter_path, "w") as fp:
                fp.write(str(counter) + "\n")

            if args.gs_sync:
                import subprocess

                run_dir = os.path.join(CHECKPOINT_DIR, args.run_name)
                gs_dir = f"gs://nost_ar_work/checkpoint_gs_sync/{args.run_name}/"
                subprocess.check_output(
                    f"gsutil -m rsync {run_dir} {gs_dir}", shell=True
                )
                subprocess.check_output(
                    f"gsutil -m rsync -d {run_dir} {gs_dir}", shell=True
                )

        if args.sample_every > 0:

            @tflex.register_command
            def generate_samples():
                print("Generating samples...")
                context_tokens = [
                    data_sampler.sample(1) for _ in range(args.batch_size)
                ]
                all_text = []
                index = 0
                while index < args.sample_num:
                    out = sess.run(tf_sample, feed_dict={context: context_tokens})
                    for i in range(min(args.sample_num - index, args.batch_size)):
                        text = enc.decode(out[i])
                        text = "======== SAMPLE {} ========\n{}\n".format(
                            index + 1, text
                        )
                        print(text)
                        all_text.append(text)
                        if args.eot_workaround:
                            verify_workaround = enc.encoder["<|endoftext|>"] in out[i]
                            verify_str = f"EOT in sample: {verify_workaround}"
                            print(verify_str)
                            all_text.append(verify_str)
                        index += 1
                maketree(os.path.join(SAMPLE_DIR, args.run_name))
                with open(
                    os.path.join(SAMPLE_DIR, args.run_name, "samples-{}").format(
                        counter
                    ),
                    "w",
                ) as fp:
                    fp.write("\n".join(all_text))

        @tflex.register_command
        def validation():
            if args.val_every <= 0:
                return
            print("Calculating validation loss...")
            losses = []
            batch_iter = tqdm.tqdm(val_batches)
            for batch in batch_iter:
                losses.append(sess.run(val_loss, feed_dict={context: batch}))
                batch_iter.set_postfix(l=losses[-1], lav=np.mean(losses))
            v_val_loss = np.mean(losses)
            print(
                "{stamp} [{counter} | {time:2.4f}] loss_val={loss:2.4f}".format(
                    stamp=timestamp(),
                    counter=counter,
                    time=time.time() - start_time,
                    loss=v_val_loss,
                )
            )

        start_time = time.time()

        def elapsed():
            return time.time() - start_time

        def say(msg):
            print(
                "{stamp} [{counter} | {time:2.4f}] {msg}".format(
                    counter=counter, time=elapsed(), msg=msg, stamp=timestamp()
                )
            )

        def sample_batch():
            r = []
            times = []
            for _ in range(args.batch_size):
                start = time.time()
                sample = data_sampler.sample(args.sample_ctx)
                end = time.time()
                elapsed = end - start
                r += [sample]
                times += [elapsed]
            total = sum(times)
            avg = total / len(times)
            return r

        prev_time = time.time()
        avg_loss = (0.0, 0.0)

        if args.debug_before_training:
            import pdb

            pdb.set_trace()

        last_saved_time = elapsed()
        while True:
            try:
                now = elapsed()
                if args.save_time > 0 and (
                    ((now - last_saved_time) / 60.0) >= args.save_time
                ):
                    save()
                    last_saved_time = now
                elif args.save_every > 0 and (counter % args.save_every == 0):
                    save()
                if (args.sample_every > 0) and (counter % args.sample_every) == 0:
                    generate_samples()
                if args.val_every > 0 and (
                    counter % args.val_every == 0 or counter == 1
                ):
                    validation()

                v_rate = update_lr()

                if args.accumulate_gradients > 1:
                    sess.run(opt_reset)
                    for _ in range(args.accumulate_gradients):
                        batch = sample_batch()
                        say("Running opt_add_gradients...")
                        sess.run(
                            opt_add_gradients,
                            feed_dict={context: batch},
                        )
                    say("Running opt_apply...")
                    if args.noise_scale:
                        v_loss, gn_big, gn_small = sess.run(opt_apply)

                        B_small = args.batch_size
                        B_big = args.batch_size * args.accumulate_gradients

                        G_noise = (B_big * gn_big - B_small * gn_small) / (
                            B_big - B_small
                        )
                        S_noise = (gn_small - gn_big) / (1 / B_small - 1 / B_big)
                    else:
                        v_loss = sess.run(opt_apply)
                else:
                    batch = sample_batch()
                    say("Running opt_apply...")
                    _, v_loss = sess.run(
                        (opt_apply, out_loss_reduced), feed_dict={context: batch}
                    )

                if args.float16:
                    v_loss = tf.to_float(v_loss).eval()

                avg_loss = (
                    avg_loss[0] * args.avg_loss_beta + v_loss,
                    avg_loss[1] * args.avg_loss_beta + 1.0,
                )

                now = time.time()
                extras = ""
                if args.noise_scale:
                    extras = f"gn_small={gn_small:2.7f} gn_big={gn_big:2.7f}"
                    extras += f" G_noise={G_noise:2.7f} S_noise={S_noise:2.7f}"
                print(
                    "{stamp} [{counter} | {time:2.4f} | {delta:2.2f}s | {ops:2.6f}tokens/s] {extras} loss={loss:2.4f} avg={avg:2.4f} rate={rate:0.7f} step={step}".format(
                        stamp=timestamp(),
                        counter=counter,
                        time=now - start_time,
                        delta=now - prev_time,
                        ops=args.accumulate_gradients
                        * args.sample_ctx
                        * args.batch_size
                        / (now - prev_time),
                        rate=v_rate,
                        loss=v_loss,
                        avg=avg_loss[0] / avg_loss[1],
                        step=current_step,
                        extras=extras,
                    )
                )
                if args.accumulate_gradients > 1:
                    pass
                else:
                    pass

                counter += 1
                current_step += 1
                global_step.load(current_step, session=sess)

                tflex.check_commands_with_args(
                    session=sess,
                    stamp=timestamp(),
                    counter=counter,
                    time=now - start_time,
                    delta=now - prev_time,
                    ops=args.batch_size / (now - prev_time),
                    rate=v_rate,
                    loss=v_loss,
                    avg=avg_loss[0] / avg_loss[1],
                    avg_loss=avg_loss,
                    step=current_step,
                    train_vars=train_vars,
                    all_vars=all_vars,
                    args=args,
                    data_sampler=data_sampler,
                    ckpt=ckpt,
                    saver=saver,
                )
                if tflex.should_quit():
                    break

                prev_time = now
                if args.debug_print_all_vars:
                    print("all variables:")
                    print("name/shape/parameter_count")
                    param_count = 0
                    for x in tf.all_variables():
                        shape = x.shape.as_list()
                        count = np.prod(shape)
                        print(x.name, shape, count)
                        param_count += count
                    print("Total parameters:", param_count)
                    args.debug_print_all_vars = False

                if args.debug_print_trainable_vars:
                    print("trainable variables:")
                    print("name/shape/parameter_count")
                    param_count = 0
                    for x in tf.trainable_variables():
                        shape = x.shape.as_list()
                        count = np.prod(shape)
                        print(x.name, shape, count)
                        param_count += count
                    print("Total parameters:", param_count)
                    args.debug_print_trainable_vars = False
            except KeyboardInterrupt:
                print("interrupted")
                if args.save_on_ctrlc:
                    save()
                if args.debug_on_ctrlc:
                    import pdb

                    pdb.set_trace()
                else:
                    break


if __name__ == "__main__":
    main()
