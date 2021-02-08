import sys

sys.path.append("gpt-2/src/")
import time
from datetime import datetime
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    average_precision_score,
    matthews_corrcoef,
    log_loss,
    hinge_loss,
)
import scipy.special
import scipy.stats

from tensorflow.contrib.opt import AdamWOptimizer
import tflex_sgdr
from model import model as model_fn
from accumulate import GradientAccumulator

from selector_nn import selector


ORIG_POST_CHAR_CHINESE = "翰"


def var_score(y_true, y_pred):
    return np.var(y_pred) / 0.083335


def skewness_score(y_true, y_pred):
    return scipy.stats.skew(y_pred)


def skewness_abs_score(y_true, y_pred):
    return np.abs(scipy.stats.skew(y_pred))


def skewness_zscore_score(y_true, y_pred):
    return scipy.stats.skewtest(y_pred).statistic


def skewness_pval_score(y_true, y_pred):
    return scipy.stats.skewtest(y_pred).pvalue


def hinge_loss_non_svm(y_true, y_pred):
    return hinge_loss([1 if entry else -1 for entry in y_true], y_pred - 0.5)


def make_textpost_scorer(metric, needs_proba=False):
    def _textpost_scorer(estimator, X, y, sample_weight=None):
        textpost_filter = X["prompt_finalchar"] == ORIG_POST_CHAR_CHINESE

        textpost_X = X[textpost_filter]
        y_true = y[textpost_filter]

        if needs_proba:
            y_pred = estimator.predict_proba(textpost_X)[:, 1]
        else:
            y_pred = estimator.predict(textpost_X)
        return metric(y_true, y_pred)

    return _textpost_scorer


class SelectorEstimatorFromCkpt(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        ckpt,
        layer_nums,
        res_dropout=0.0,
        attn_dropout=0.0,
        attn_dropout_before_softmax: bool = False,
        acti_dropout=0.0,
        orth_init=True,
        he_init=False,
        init_default_gain=1.0,
        init_lreg_gain=0.02,
        use_mlp=True,
        resid_mlp=True,
        mlp_ratio=3,
        use_logit_diff_basis=False,
        use_only_logit_diff=False,
        weight_decay=0.025,
        base_lr=2.5e-5,
        min_lr_frac=0.25,
        m_mul=0.5,
        warmup_ratio=0.0,
        epochs=3,
        batch_size=8,
        grad_clip=1000,
        base_hparams=hparams,
        enc=enc,
        selection_tok=SELECTION_TOK,
        length=825,
        persist_variables=True,
        cleanup_on_exception=True,
        session_override=None,
        supervise_logits=False,
        supervise_only_logit_diff=False,
        calibrate=False,
        calibration_val_size=0.15,
        calibration_split_type="ttsp",
        calibrate_prefixes_separately=False,
        calibrate_logits_separately=False,
        n_head=40,
        additional_full_blocks=0,
        show_batch_stats=False,
        stop_early=False,
        stopping_metric="ap",
        evaluate_during_training=True,
        warm_resets=False,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        optimizer="adam",
        huber=False,
        huber_delta=1.0,
        flooding=False,
        flood_level=0.0,
        accumulate_gradients=1,
    ):
        self.ckpt = ckpt
        self.layer_nums = layer_nums
        self.res_dropout = res_dropout
        self.attn_dropout = attn_dropout
        self.attn_dropout_before_softmax = attn_dropout_before_softmax
        self.acti_dropout = acti_dropout
        self.orth_init = orth_init
        self.he_init = he_init
        self.init_default_gain = init_default_gain
        self.init_lreg_gain = init_lreg_gain
        self.use_mlp = use_mlp
        self.resid_mlp = resid_mlp
        self.mlp_ratio = mlp_ratio
        self.use_logit_diff_basis = use_logit_diff_basis
        self.use_only_logit_diff = use_only_logit_diff

        self.weight_decay = weight_decay
        self.base_lr = base_lr
        self.min_lr_frac = min_lr_frac
        self.m_mul = m_mul
        self.warmup_ratio = warmup_ratio
        self.epochs = epochs
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        self.base_hparams = base_hparams
        self.enc = enc
        self.selection_tok = selection_tok
        self.length = length

        self.persist_variables = persist_variables
        self.cleanup_on_exception = cleanup_on_exception
        self.session_override = session_override

        self.supervise_logits = supervise_logits
        self.supervise_only_logit_diff = supervise_only_logit_diff
        if (
            self.supervise_logits
            and self.supervise_only_logit_diff
            and not self.use_only_logit_diff
        ):
            self.use_logit_diff_basis = True

        self.calibrate = calibrate
        self.calibration_val_size = calibration_val_size
        self.calibration_split_type = calibration_split_type
        self.calibrate_prefixes_separately = calibrate_prefixes_separately
        self.calibrate_logits_separately = calibrate_logits_separately

        self.n_head = n_head

        self.additional_full_blocks = additional_full_blocks
        self.show_batch_stats = show_batch_stats
        self.stop_early = stop_early
        self.evaluate_during_training = evaluate_during_training
        self.stopping_metric = stopping_metric

        self.warm_resets = warm_resets

        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.adam_epsilon = adam_epsilon
        self.optimizer = optimizer
        self.huber = huber
        self.huber_delta = huber_delta
        self.flooding = flooding
        self.flood_level = flood_level
        self.accumulate_gradients = accumulate_gradients

        self.uid_ = None
        self.select_scope_ = None
        self.hparams_select_train_ = None
        self.hparams_select_eval_ = None
        self.train_vars_ = None
        self.decay_vars_ = None
        self.selection_step_train_ = None
        self.selection_step_eval_ = None
        self.select_logits_train_ = None
        self.select_logits_eval_ = None
        self.select_target_ = None
        self.select_loss_ = None
        self.target_cols_ = None

        self.global_step_ = None
        self.lr_ = None
        self.opt_ = None
        self.lr_calib_ = None
        self.lr_calib_resp_ = None
        self.lr_calib_orig_ = None
        self.n_head_ = None
        self.last_best_val_metric_ = None
        self.gradient_accumulator_ = None
        self.opt_reset_ = None
        self.opt_add_gradients_ = None

        self.X_train_, self.y_train_, self.X_val_, self.y_val_ = None, None, None, None

    def _load_ckpt(self):
        load_done = False

        while not load_done:
            try:
                tf.reset_default_graph()
                self.session_ = tf.Session()

                print("entering self.session_")
                with self.session_.as_default():
                    self.context_for_h_ = tf.placeholder(
                        tf.int32, [self.batch_size, None], name="context"
                    )
                    if self.supervise_logits:
                        self.select_target_ = tf.placeholder(
                            tf.float32, [self.batch_size, 2], name="select_target"
                        )
                    else:
                        self.select_target_ = tf.placeholder(
                            tf.int32, [self.batch_size], name="select_target"
                        )
                    if self.add_prompt_cont_embs:
                        self.prompt_end_ntoks_ = tf.placeholder(
                            tf.int32, [self.batch_size], name="select_prompt_end_ntoks"
                        )
                    else:
                        self.prompt_end_ntoks_ = None

                    _ = model.model(
                        hparams=self.base_hparams,
                        X=self.context_for_h_,
                        return_activations_at=self.layer_nums,
                    )["activations"]

                    saver = tflex.Saver()
                    print(f"restoring checkpoint: {self.ckpt}")
                    saver.restore(self.session_, self.ckpt)
                load_done = True
            except Exception as e:
                print(f"encountered {e}, retrying...")

    def _setup(self, X, y):
        print("entering setup")
        print(f"fitting this estimator:\n{repr(self)}\n")

        self.uid_ = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

        self.n_head_ = self.base_hparams.n_head if self.n_head is None else self.n_head

        self.select_scope_ = "select_" + self.uid_
        self.select_scope_train_ = self.select_scope_
        self.select_scope_eval_ = self.select_scope_

        self.hparams_select_train_ = HParams(
            n_vocab=self.base_hparams.n_vocab,
            n_ctx=self.base_hparams.n_ctx,
            n_embd=self.base_hparams.n_embd,
            n_head=self.n_head_,
            n_layer=self.base_hparams.n_layer,
            res_dropout=self.res_dropout,
            attn_dropout=self.attn_dropout,
            attn_dropout_before_softmax=self.attn_dropout_before_softmax,
            acti_dropout=self.acti_dropout,
            dtype=tf.float32,
            do_resid=False,
            orth_init=self.orth_init,
            he_init=self.he_init,
            init_default_gain=self.init_default_gain,
            init_lreg_gain=self.init_lreg_gain,
            additional_full_blocks=self.additional_full_blocks,
        )

        # TODO: DRY
        self.hparams_select_eval_ = HParams(
            n_vocab=self.base_hparams.n_vocab,
            n_ctx=self.base_hparams.n_ctx,
            n_embd=self.base_hparams.n_embd,
            n_head=self.n_head_,
            n_layer=self.base_hparams.n_layer,
            res_dropout=0.0,
            attn_dropout=0.0,
            attn_dropout_before_softmax=self.attn_dropout_before_softmax,
            acti_dropout=0.0,
            dtype=tf.float32,
            do_resid=False,
            orth_init=self.orth_init,
            he_init=self.he_init,
            init_default_gain=self.init_default_gain,
            init_lreg_gain=self.init_lreg_gain,
            additional_full_blocks=self.additional_full_blocks,
        )

        print("loading ckpt")
        if self.session_override is not None:
            self.session_ = self.session_override
            self.context_for_h_ = self.session_.graph.get_tensor_by_name("context:0")
            self.select_target_ = self.session_.graph.get_tensor_by_name(
                "select_target:0"
            )
        else:
            self._load_ckpt()

        with self.session_.as_default():
            print("making selection steps")
            self.selection_step_train_ = selector(
                X=self.context_for_h_,
                hparams=self.base_hparams,
                select_scope=self.select_scope_,
                hparams_select=self.hparams_select_train_,
                layer_nums=self.layer_nums,
                use_mlp=self.use_mlp,
                resid_mlp=self.resid_mlp,
                mlp_ratio=self.mlp_ratio,
                use_logit_diff_basis=self.use_logit_diff_basis,
                use_only_logit_diff=self.use_only_logit_diff,
                batch_size=self.batch_size,
                scalar_mix_n_out=self.scalar_mix_n_out,
            )

            self.selection_step_eval_ = selector(
                X=self.context_for_h_,
                hparams=self.base_hparams,
                select_scope=self.select_scope_,
                hparams_select=self.hparams_select_eval_,
                layer_nums=self.layer_nums,
                use_mlp=self.use_mlp,
                resid_mlp=self.resid_mlp,
                mlp_ratio=self.mlp_ratio,
                use_logit_diff_basis=self.use_logit_diff_basis,
                use_only_logit_diff=self.use_only_logit_diff,
                batch_size=self.batch_size,
            )
            self.select_logits_train_ = self.selection_step_train_["logits_select"]
            self.select_logits_eval_ = self.selection_step_eval_["logits_select"]

            self.activ_norms_ = {
                k: self.selection_step_train_[k]
                for k in self.selection_step_train_.keys()
                if k.startswith("norm_")
            }

            print("making losses")

            if self.supervise_logits:
                if self.huber:

                    def _huber_loss(labels, predictions):
                        return tf.losses.huber_loss(
                            labels, predictions, delta=self.huber_delta
                        )

                    regression_loss = _huber_loss
                else:
                    regression_loss = tf.squared_difference
                if self.supervise_only_logit_diff:
                    self.select_logit_diff_train_ = self.selection_step_train_[
                        "logit_diff"
                    ]
                    self.select_target_logit_diff_ = (
                        self.select_target_[:, 1] - self.select_target_[:, 0]
                    )
                    self.select_loss_ = tf.reduce_mean(
                        regression_loss(
                            self.select_target_logit_diff_,
                            self.select_logit_diff_train_,
                        )
                    )
                else:
                    self.select_loss_ = tf.reduce_mean(
                        regression_loss(self.select_target_, self.select_logits_train_)
                    )
            else:
                loss_unreduced = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.select_target_, logits=self.select_logits_train_
                )
                self.is_flooding_ = tf.reduce_mean(
                    tf.cast(loss_unreduced < self.flood_level, tf.float32)
                )
                if self.flooding:
                    loss_unreduced = (
                        tf.abs(loss_unreduced - self.flood_level) + self.flood_level
                    )
                self.select_loss_ = tf.reduce_mean(loss_unreduced)

            print("setting up vars")
            selector_vars = [
                var
                for var in tf.trainable_variables()
                if self.select_scope_train_ in var.name
                or self.select_scope_eval_ in var.name
            ]
            # self.train_vars_ = [var for var in selector_vars if "ln_" not in var.name]
            self.train_vars_ = [var for var in selector_vars]
            self.fullb_ln_vars_ = [
                var
                for var in selector_vars
                if "ln" in var.name and "fullblock" in var.name
            ]
            self.fullb_attn_vars_ = [
                var
                for var in selector_vars
                if "attn/" in var.name and "/w" in var.name and "fullblock" in var.name
            ]
            self.fullb_mlp_vars_ = [
                var
                for var in selector_vars
                if "mlp/" in var.name and "/w" in var.name and "fullblock" in var.name
            ]

            self.ln_vars_ = [
                var
                for var in selector_vars
                if "ln" in var.name and "fullblock" not in var.name
            ]
            self.attn_vars_ = [
                var
                for var in selector_vars
                if "attn/" in var.name
                and "/w" in var.name
                and "fullblock" not in var.name
            ]
            self.mlp_vars_ = [
                var
                for var in selector_vars
                if "mlp/" in var.name
                and "/w" in var.name
                and "fullblock" not in var.name
            ]
            self.lreg_vars_ = [var for var in selector_vars if "/w_select" in var.name]
            self.mix_vars_ = [var for var in selector_vars if "scalar_mix" in var.name]
            self.softlayer_vars_ = [
                var for var in selector_vars if "soft_layer" in var.name
            ]

            self.decay_vars_ = [
                var
                for var in self.train_vars_
                if "b_select" not in var.name
                and "ln_" not in var.name
                and "_length" not in var.name
                and "down_proj" not in var.name
                and "prompt_cont" not in var.name
                and "scalar_mix" not in var.name
            ]
            parameter_count = sum(
                [np.prod(v.shape.as_list()) for v in self.train_vars_]
            )
            print(
                "This model is using %d parameters (%.2fM)"
                % (parameter_count, parameter_count / (1024.0 * 1024.0))
            )

            print("initializing vars")
            initialize_uninitialized(self.session_)

            # opt stuff
            print("setting up opt")
            self.global_step_ = tf.get_variable(
                "global_step_" + self.uid_,
                shape=(),
                dtype=tf.int32,
                trainable=False,
            )
            self.global_step_.load(0, session=self.session_)

            n_batches_per_epoch = len(X) // self.batch_size
            if self.warm_resets:
                initial_period_steps = int(n_batches_per_epoch)
            else:
                initial_period_steps = self.epochs * (int(n_batches_per_epoch) + 1)
            print(f"initial_period_steps: {initial_period_steps}")
            warmup_steps = int(self.warmup_ratio * initial_period_steps)
            print(f"warmup_steps: {warmup_steps}")

            min_lr = self.base_lr * self.min_lr_frac

            base_lr_tensor = tf.constant(self.base_lr, dtype=tf.float32)
            tflex_lr = tflex_sgdr.sgdr_decay_with_warmup(
                base_lr_tensor,
                self.global_step_,
                warmup_steps=warmup_steps,
                initial_period_steps=initial_period_steps,
                t_mul=1.0,
                m_mul=self.m_mul,
            )
            tflex_lr_floored = tf.maximum(
                tflex_lr, tf.constant(min_lr, dtype=tflex_lr.dtype)
            )
            # let it go to zero in warmup, but not in decay
            self.lr_ = tf.where(
                self.global_step_ > warmup_steps, tflex_lr_floored, tflex_lr
            )

            print("creating opt")
            self.opt_scope_ = self.select_scope_ + "/optimizer"
            with tf.variable_scope(self.opt_scope_):
                if self.optimizer == "adam":
                    self.opt_ = AdamWOptimizer(
                        weight_decay=self.weight_decay * self.lr_,
                        learning_rate=self.lr_,
                        beta1=self.adam_beta1,
                        beta2=self.adam_beta2,
                        epsilon=self.adam_epsilon,
                    )
                opt_gradients, opt_variables = zip(
                    *self.opt_.compute_gradients(self.select_loss_, self.train_vars_)
                )

                opt_gradients_fullb_ln = [
                    g
                    for g, v in zip(opt_gradients, opt_variables)
                    if v in self.fullb_ln_vars_
                ]
                opt_gradients_fullb_attn = [
                    g
                    for g, v in zip(opt_gradients, opt_variables)
                    if v in self.fullb_attn_vars_
                ]
                opt_gradients_fullb_mlp = [
                    g
                    for g, v in zip(opt_gradients, opt_variables)
                    if v in self.fullb_mlp_vars_
                ]
                opt_gradients_ln = [
                    g
                    for g, v in zip(opt_gradients, opt_variables)
                    if v in self.ln_vars_
                ]
                opt_gradients_attn = [
                    g
                    for g, v in zip(opt_gradients, opt_variables)
                    if v in self.attn_vars_
                ]
                opt_gradients_mlp = [
                    g
                    for g, v in zip(opt_gradients, opt_variables)
                    if v in self.mlp_vars_
                ]
                opt_gradients_lreg = [
                    g
                    for g, v in zip(opt_gradients, opt_variables)
                    if v in self.lreg_vars_
                ]
                opt_gradients_mix = [
                    g
                    for g, v in zip(opt_gradients, opt_variables)
                    if v in self.mix_vars_
                ]

                self.gn_ = tf.global_norm(opt_gradients)
                self.gn_fullb_ln_ = tf.global_norm(opt_gradients_fullb_ln)
                self.gn_fullb_attn_ = tf.global_norm(opt_gradients_fullb_attn)
                self.gn_fullb_mlp_ = tf.global_norm(opt_gradients_fullb_mlp)
                self.gn_ln_ = tf.global_norm(opt_gradients_ln)
                self.gn_attn_ = tf.global_norm(opt_gradients_attn)
                self.gn_mlp_ = tf.global_norm(opt_gradients_mlp)
                self.gn_lreg_ = tf.global_norm(opt_gradients_lreg)
                self.gn_mix_ = tf.global_norm(opt_gradients_mix)

                opt_gradients, _ = tf.clip_by_global_norm(opt_gradients, self.grad_clip)

                if self.accumulate_gradients > 1:
                    self.gradient_accumulator_ = GradientAccumulator(
                        var_list=self.train_vars_
                    )
                    self.opt_reset_ = self.gradient_accumulator_.reset()
                    self.opt_add_gradients_ = self.gradient_accumulator_.add_gradients(
                        self.select_loss_, list(zip(opt_gradients, opt_variables))
                    )
                    if self.optimizer == "adam":
                        self.opt_apply_ = self.gradient_accumulator_.apply_gradients(
                            self.opt_, decay_var_list=self.decay_vars_
                        )
                    else:
                        self.opt_apply_ = self.gradient_accumulator_.apply_gradients(
                            self.opt_
                        )
                else:
                    if self.optimizer == "adam":
                        self.opt_apply_ = self.opt_.apply_gradients(
                            list(zip(opt_gradients, opt_variables)),
                            decay_var_list=self.decay_vars_,
                        )
                    else:
                        self.opt_apply_ = self.opt_.apply_gradients(
                            list(zip(opt_gradients, opt_variables))
                        )
            print("initializing opt")
            re_initialize(self.session_, tf.global_variables(scope=self.opt_scope_))

            self.novograd_avgs_ = []
            if self.optimizer == "novograd":
                self.novograd_avgs_ = self.opt_._grads_ema

    def _make_batched_data(self, X, y=None):
        if y is None:
            data = X.reset_index(drop=True)
            data.index.name = "selector_internal_ix"
            data = data.reset_index()
        else:
            self.target_cols_ = y.columns if len(y.shape) > 1 else y.name
            data = pd.concat([X, y], axis=1)
        data = data.sort_values(by="n_tokens")
        data = reshuffle_batches(data)
        return data

    def _feed_from_batch(self, data_batch, scope):
        feed_dict = {}
        if self.add_prompt_cont_embs:
            batch_prompt_end_ntoks = data_batch.prompt_end_ntoks.values
        batch_context = [
            self.enc.encode(text)[: (self.length - 1)] + [self.selection_tok]
            for text in data_batch.selector_input.values
        ]
        max_tokens = min(self.length, max([len(toks) for toks in batch_context]))
        batch_context_ = [
            toks + [0 for _ in range(max_tokens - len(toks))] for toks in batch_context
        ]
        # batch_context_ = [toks[-self.length:] for toks in batch_context_]
        batch_context = batch_context_
        feed_dict[self.context_for_h_] = np.asarray(batch_context_)

        if self.add_prompt_cont_embs:
            shift = max(0, max_tokens - self.length)
            batch_prompt_end_ntoks = batch_prompt_end_ntoks - shift
            batch_prompt_end_ntoks[batch_prompt_end_ntoks < 0] = 0

            feed_dict[self.prompt_end_ntoks_] = batch_prompt_end_ntoks

        return feed_dict, max_tokens

    def _epoch(self, X, y, avg_loss_beta=0.98):
        extra_postfixes = {}
        all_losses = []
        batch_loss = None
        running_loss = None

        data = self._make_batched_data(X, y)
        steps = len(data) // self.batch_size

        row_ix = 0
        step_iter = tqdm(
            list(range(0, steps)), smoothing=0.0, miniters=1, mininterval=3
        )
        for step_ix in step_iter:
            data_batch = data.iloc[row_ix : row_ix + self.batch_size, :]

            feed_dict, batch_max_tokens = self._feed_from_batch(
                data_batch, scope=self.select_scope_train_
            )

            batch_target = (
                data_batch[self.target_cols_]
                if len(self.target_cols_) > 1
                else data_batch[self.target_cols_[0]]
            )
            feed_dict[self.select_target_.name] = batch_target.values

            with self.session_.as_default():
                try:
                    if self.accumulate_gradients > 1:
                        _, cur_lr, current_step = self.session_.run(
                            [self.opt_add_gradients_, self.lr_, self.global_step_],
                            feed_dict=feed_dict,
                        )
                        accum_count = self.gradient_accumulator_.count_loss.eval(
                            self.session_
                        )
                        if accum_count % self.accumulate_gradients != 0:
                            # TODO: DRY
                            row_ix += self.batch_size
                            self.global_step_.load(
                                current_step + 1, session=self.session_
                            )
                            extra_postfixes["aaa_accum_count"] = accum_count
                            extra_postfixes["ntok"] = batch_max_tokens
                            step_iter.set_postfix(
                                loss=batch_loss,
                                loss_avg=running_loss,
                                lr=cur_lr,
                                **extra_postfixes,
                            )
                            continue
                    if self.show_batch_stats:
                        (
                            loss_out,
                            cur_lr,
                            current_step,
                            cur_gn,
                            cur_gn_fullb_ln,
                            cur_gn_fullb_attn,
                            cur_gn_fullb_mlp,
                            cur_gn_ln,
                            cur_gn_attn,
                            cur_gn_mlp,
                            cur_gn_lreg,
                            cur_gn_mix,
                            cur_activ_norms,
                            cur_is_flooding,
                            *softlayer_vars,
                            apply_out,
                        ) = self.session_.run(
                            [
                                self.select_loss_,
                                self.lr_,
                                self.global_step_,
                                self.gn_,
                                self.gn_fullb_ln_,
                                self.gn_fullb_attn_,
                                self.gn_fullb_mlp_,
                                self.gn_ln_,
                                self.gn_attn_,
                                self.gn_mlp_,
                                self.gn_lreg_,
                                self.gn_mix_,
                                self.activ_norms_,
                                self.is_flooding_,
                                self.softlayer_vars_,
                                self.opt_apply_,
                            ],
                            feed_dict=feed_dict,
                        )
                    else:
                        loss_out, cur_lr, current_step, apply_out = self.session_.run(
                            [
                                self.select_loss_,
                                self.lr_,
                                self.global_step_,
                                self.opt_apply_,
                            ],
                            feed_dict=feed_dict,
                        )
                    if self.accumulate_gradients > 1:
                        self.session_.run(self.opt_reset_)
                except tf.errors.InvalidArgumentError:
                    continue

            batch_loss = apply_out if self.accumulate_gradients > 1 else loss_out
            all_losses.append(batch_loss)
            if running_loss is None:
                if step_ix > 3:
                    running_loss = sum(all_losses) / len(all_losses)
            else:
                running_loss = (avg_loss_beta * running_loss) + (
                    (1 - avg_loss_beta) * batch_loss
                )
            # cur_lr = self.lr_.eval(session=self.session_)
            # current_step = self.global_step_.eval(self.session_)

            extra_postfixes = {}
            if self.show_batch_stats:
                extra_postfixes["z_gn_ln"] = cur_gn_ln
                extra_postfixes["z_gn_attn"] = cur_gn_attn
                extra_postfixes["z_gn_lreg"] = cur_gn_lreg
                extra_postfixes["zz_gn_ztotal"] = cur_gn
                if self.use_scalar_mix:
                    extra_postfixes["z_gn_amix"] = cur_gn_mix
                if self.use_mlp:
                    extra_postfixes["z_gn_mlp"] = cur_gn_mlp
                if self.additional_full_blocks > 0:
                    extra_postfixes["z_gn_fullb_ln"] = cur_gn_fullb_ln
                    extra_postfixes["z_gn_fullb_attn"] = cur_gn_fullb_attn
                    extra_postfixes["z_gn_fullb_mlp"] = cur_gn_fullb_mlp
                if self.flooding:
                    extra_postfixes["z_flood%"] = cur_is_flooding
                # for i, nga in enumerate(novograd_avgs):
                #     extra_postfixes[f"aa_nvg_avg{i}"] = nga
                for k, v in cur_activ_norms.items():
                    extra_postfixes[f"zz_{k}"] = v
                for var, val in zip(self.softlayer_vars_, softlayer_vars[0]):
                    extra_postfixes[f"aa_soft_{var.name.partition('/')[2]}"] = [
                        f"{item:.4f}" for item in val
                    ]
            if self.accumulate_gradients > 1:
                extra_postfixes["aaa_accum_count"] = accum_count
            extra_postfixes["ntok"] = batch_max_tokens

            step_iter.set_postfix(
                loss=batch_loss, loss_avg=running_loss, lr=cur_lr, **extra_postfixes
            )
            row_ix += self.batch_size
            self.global_step_.load(current_step + 1, session=self.session_)

    def _val_split(self, X, y):
        if self.calibrate or self.evaluate_during_training:
            if self.calibration_split_type == "tts":
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, stratify=y, test_size=self.calibration_val_size
                )
            elif self.calibration_split_type == "ttsp":
                stratifier = X["prompt_finalchar"] + y.apply(str)
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, stratify=stratifier, test_size=self.calibration_val_size
                )
            elif self.calibration_split_type == "gss":
                gss = GroupKFold(n_splits=int(1 / self.calibration_val_size))
                train_ix, val_ix = next(gss.split(X, groups=X["prefix"]))
                X_train, X_val = X.iloc[train_ix, :], X.iloc[val_ix, :]
                y_train, y_val = y.iloc[train_ix], y.iloc[val_ix]
            else:
                raise ValueError(
                    f"calibration_split_type={self.calibration_split_type} not implemented"
                )
            print(
                f"made split:\n\tX_train {X_train.shape}, X_val {X_val.shape}, y_train {y_train.shape}, y_val {y_val.shape}"
            )
        else:
            X_train, y_train = X, y
            X_val, y_val = None, None
        return X_train, y_train, X_val, y_val

    def _display_eval_metrics(self, y_true, preds, probs, pfcs=None):
        eval_metrics_results = {}

        eval_metrics = {
            "loss": {"fn": log_loss, "greater_is_better": False, "wants_probs": True},
            "acc": {
                "fn": accuracy_score,
                "greater_is_better": True,
                "wants_probs": False,
            },
            "mcc": {
                "fn": matthews_corrcoef,
                "greater_is_better": True,
                "wants_probs": False,
            },
            "ap": {
                "fn": average_precision_score,
                "greater_is_better": True,
                "wants_probs": True,
            },
            "br": {
                "fn": brier_score_loss,
                "greater_is_better": False,
                "wants_probs": True,
            },
            "var+": {"fn": var_score, "greater_is_better": False, "wants_probs": True},
            "skew": {
                "fn": skewness_score,
                "greater_is_better": False,
                "wants_probs": True,
            },
            "skew_p": {
                "fn": skewness_pval_score,
                "greater_is_better": True,
                "wants_probs": True,
            },
            "hinge": {
                "fn": hinge_loss_non_svm,
                "greater_is_better": True,
                "wants_probs": True,
            },
        }

        for name, metric in eval_metrics.items():
            m = (
                metric["fn"](y_true, probs[:, 1])
                if metric["wants_probs"]
                else metric["fn"](y_true, preds)
            )
            eval_metrics_results[name] = m

        with tqdm(
            list(range(0, 1)), smoothing=0.0, miniters=1, mininterval=1
        ) as fake_iter:
            fake_iter.set_postfix(**eval_metrics_results)

        if pfcs is not None:
            for pfc in sorted(pfcs.unique()):
                pfc_filter = (pfcs == pfc).values
                eval_metrics_results_pfc = {"aa_type": pfc, "aa_N": pfc_filter.sum()}

                probs_pfc = probs[pfc_filter, :]
                preds_pfc = preds[pfc_filter]
                y_true_pfc = y_true.values[pfc_filter]
                for name, metric in eval_metrics.items():
                    m = (
                        metric["fn"](y_true_pfc, probs_pfc[:, 1])
                        if metric["wants_probs"]
                        else metric["fn"](y_true_pfc, preds_pfc)
                    )
                    eval_metrics_results_pfc[name] = m
                with tqdm(
                    list(range(0, 1)), smoothing=0.0, miniters=1, mininterval=1
                ) as fake_iter_pfc:
                    fake_iter_pfc.set_postfix(**eval_metrics_results_pfc)

        return eval_metrics_results

    def eval_on_val_set(self, X_val, y_val, disable_calibration=True):
        stop_early_signal = None

        probs = self._predict(X_val, key="probs", disable_calibration=True)
        preds = probs[:, 1] > 0.5

        eval_metrics_results = self._display_eval_metrics(
            y_val, preds, probs, pfcs=X_val["prompt_finalchar"]
        )

        if self.stop_early:
            criterion = eval_metrics_results[self.stopping_metric]
            if self.last_best_val_metric_ is not None:
                delta = criterion - self.last_best_val_metric_
                delta_is_okay = (
                    (delta > 0)
                    if eval_metrics[self.stopping_metric]["greater_is_better"]
                    else (delta < 0)
                )
                stop_early_signal = not delta_is_okay
            self.last_best_val_metric_ = criterion
        return stop_early_signal, eval_metrics_results

    def _calib_inputs(self, logits):
        if self.calibrate_logits_separately:
            return logits
        else:
            logit_diff = logits[:, 1:] - logits[:, :1]
            return logit_diff

    @property
    def _calib_kwargs(self):
        return (
            {"penalty": "l2", "C": 1e4}
            if self.calibrate_logits_separately
            else {"penalty": "none"}
        )

    def _fit_calibration(self, X_val, y_val):
        logits = self._predict(X_val, key="logits", disable_calibration=True)
        calib_inputs = self._calib_inputs(logits)

        probs = scipy.special.softmax(logits, axis=1)
        preds = probs[:, 1] > 0.5
        self._display_eval_metrics(y_val, preds, probs, pfcs=X_val["prompt_finalchar"])

        if self.calibrate_prefixes_separately:
            orig_filter = (X_val["prompt_finalchar"] == ORIG_POST_CHAR_CHINESE).values

            self.lr_calib_resp_ = LogisticRegression(**self._calib_kwargs)
            self.lr_calib_resp_.fit(
                calib_inputs[orig_filter == False], y_val.values[orig_filter == False]
            )

            self.lr_calib_orig_ = LogisticRegression(**self._calib_kwargs)
            self.lr_calib_orig_.fit(
                calib_inputs[orig_filter == True], y_val.values[orig_filter == True]
            )

            calib_coef_info = {
                "resp_coef": self.lr_calib_resp_.coef_.tolist(),
                "resp_intercept": self.lr_calib_resp_.intercept_,
                "orig_coef": self.lr_calib_orig_.coef_.tolist(),
                "orig_intercept": self.lr_calib_orig_.intercept_,
            }
        else:
            self.lr_calib_ = LogisticRegression(**self._calib_kwargs)
            self.lr_calib_.fit(calib_inputs, y_val)

            calib_coef_info = {
                "coef": self.lr_calib_.coef_.tolist(),
                "intercept": self.lr_calib_.intercept_,
            }

        with tqdm(
            list(range(0, 1)), smoothing=0.0, miniters=1, mininterval=1
        ) as fake_iter:
            fake_iter.set_postfix(**calib_coef_info)

        # TODO: get rid of this entirely if that doesn't break anything
        # i've never found it infomative b/c it's evaluating on (lr_calib_'s) training data
        #
        # calib_probs = self._compute_calib_probs(logits, pfcs=X_val["prompt_finalchar"])
        # calib_preds = calib_probs[:, 1]>0.5
        # self._display_eval_metrics(y_val, calib_preds, calib_probs,  pfcs=X_val['prompt_finalchar'])

    @property
    def train_val_sizes_(self):
        return {
            "train": len(self.X_train_),
            "val": 0 if self.X_val_ is None else len(self.X_val_),
        }

    def fit(self, X, y, avg_loss_beta=0.99):
        try:
            self.X_train_, self.y_train_, self.X_val_, self.y_val_ = self._val_split(
                X, y
            )
            self._setup(self.X_train_, self.y_train_)
            for epoch_ix in tqdm(list(range(self.epochs))):
                self._epoch(self.X_train_, self.y_train_, avg_loss_beta=avg_loss_beta)

                epoch_needs_val = self.stop_early or self.evaluate_during_training

                if epoch_needs_val:
                    stop_early_signal, eval_metrics_results = self.eval_on_val_set(
                        self.X_val_, self.y_val_
                    )
                    if stop_early_signal:
                        print(f"stopping early at {epoch_ix}")
                        break
            if self.calibrate:
                self._fit_calibration(self.X_val_, self.y_val_)
        except (Exception, KeyboardInterrupt) as e:
            if self.cleanup_on_exception:
                self.cleanup()
            raise e
        return self

    def _compute_calib_probs(self, logits, pfcs):
        calib_inputs = self._calib_inputs(logits)
        if self.calibrate_prefixes_separately:
            orig_filter = (pfcs == ORIG_POST_CHAR_CHINESE).values
            probs = np.zeros((len(logits), 2))

            if (orig_filter == False).any():
                probs[orig_filter == False, :] = self.lr_calib_resp_.predict_proba(
                    calib_inputs[orig_filter == False]
                )
            if (orig_filter == True).any():
                probs[orig_filter == True, :] = self.lr_calib_orig_.predict_proba(
                    calib_inputs[orig_filter == True]
                )
        else:
            probs = self.lr_calib_.predict_proba(calib_inputs)
        return probs

    def _predict_select(self, batch, threshold=0.5, disable_calibration=False):
        if len(batch) != self.batch_size:
            raise ValueError("badlength")
        feed_dict, _ = self._feed_from_batch(batch, scope=self.select_scope_eval_)

        with self.session_.as_default():
            logits = self.session_.run(self.select_logits_eval_, feed_dict=feed_dict)

        probs_raw = scipy.special.softmax(logits, axis=1)

        if self.calibrate and not disable_calibration:
            probs = self._compute_calib_probs(logits, pfcs=batch["prompt_finalchar"])
        else:
            probs = probs_raw
        results = {"logits": logits, "probs": probs, "probs_raw": probs_raw}
        results["preds"] = probs[:, 1] > threshold
        return results

    def _predict(self, X, key="preds", disable_calibration=False):
        all_pd_ixs = []
        all_preds = []
        data = self._make_batched_data(X)
        steps = len(data) // self.batch_size + 1

        row_ix = 0
        step_iter = tqdm(
            list(range(0, steps)), smoothing=0.0, miniters=1, mininterval=1
        )
        for step_ix in step_iter:
            data_batch = data.iloc[row_ix : row_ix + self.batch_size, :]
            n_needed = len(data_batch)
            if n_needed == 0:
                continue
            if n_needed < self.batch_size:
                data_batch = pd.concat(
                    [data_batch]
                    + (self.batch_size - n_needed) * [data_batch.iloc[:1, :]],
                    ignore_index=True,
                )

            results_batch = self._predict_select(
                data_batch, disable_calibration=disable_calibration
            )
            all_preds.extend(results_batch[key][:n_needed])
            all_pd_ixs.extend(data_batch["selector_internal_ix"].tolist()[:n_needed])

            row_ix += self.batch_size

        if key == "preds":
            pd_obj = pd.Series(all_preds, index=all_pd_ixs)
        else:
            pd_obj = pd.DataFrame(all_preds, index=all_pd_ixs)
        pd_obj = pd_obj.sort_index()
        return pd_obj.values

    def predict(self, X):
        # TODO: make this less of a shitty hack
        return self._predict(X, key="preds" if not self.supervise_logits else "logits")

    def predict_proba(self, X):
        return self._predict(X, key="probs")

    def decision_function(self, X):
        return self._predict(X, key="probs")

    def cleanup(self):
        if not self.persist_variables:
            print("cleanup: closing session")
            self.session_.close()
        else:
            print("cleanup: no-op (self.persist_variables is on)")
