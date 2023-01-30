# TODO:
# - better name for "prompt_finalchar"
import inspect
import joblib
import os
import json
import gc
import weakref
import random
from functools import partial
from string import ascii_lowercase

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
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

import torch
import torch.nn.functional as F

from tumblr_to_text.classic.autoresponder_static import ORIG_POST_CHAR_CHINESE
from classifier_heads.head_nn import NostARHead, NostARHeadArchitectureParams, GPT2TokenizerType, GPTModelType, prep_inputs
from classifier_heads.head_nn_utils import NostARHeadOptimizerParams, get_nost_ar_head_optimizers, get_nost_ar_head_scheduler, cross_entropy_with_flooding, make_huber_loss_from_logits
from util.util import typed_namedtuple_to_dict
from ml.kv_cache import kv_buffer_scope


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


def reshuffle_batches(train_data_for_selection, batch_size, seed=None):
    train_data_for_selection = train_data_for_selection.sort_values(by="n_tokens")
    batches = [
        train_data_for_selection.iloc[row_ix : row_ix + batch_size, :]
        for row_ix in range(0, len(train_data_for_selection), batch_size)
    ]

    if seed is not None:
        np.random.seed(seed)
    np.random.shuffle(batches)

    return pd.concat(batches, ignore_index=True)


class NostARHeadEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_model: GPTModelType,
        tokenizer: GPT2TokenizerType,
        params: NostARHeadArchitectureParams,
        opt_params: NostARHeadOptimizerParams,
        params_extras=None,
        device='cuda:0',
        length=None,
        regression_target=False,
        calibrate=True,
        calibration_val_size=0.1,
        calibration_split_type="ttsp",
        calibration_val_seed=None,
        shuffle_seed=None,
        evaluate_during_training=True,
        huber_delta=1.0,
        flooding=True,
        flood_level=0.0,
        cleanup_on_exception=True,
        show_running_loss=True,
        use_amp_training=True,
        use_amp_inference=True,
        pad_to_mult=None,
        display_interval_secs=3,
        partial_forward_type="tfu",
        use_wandb=False,
        wandb_init_args=None,
        use_galileo=False,
        galileo_separate_runs_for_epochs=False,
        blocks_inference_device_attn=None,
        blocks_inference_device_mlp=None,
        **kwargs
    ):
        self.device = device
        self.blocks_inference_device_attn = blocks_inference_device_attn or self.device
        self.blocks_inference_device_mlp = blocks_inference_device_mlp or self.device
        self._base_model = weakref.ref(base_model)
        self.tokenizer = tokenizer
        self.params = params
        self.opt_params = opt_params

        self.params_extras = {} if params_extras is None else params_extras

        self.length = length

        self.regression_target = regression_target

        self.calibrate = calibrate
        self.calibration_val_size = calibration_val_size
        self.calibration_split_type = calibration_split_type

        self.calibration_val_seed = calibration_val_seed
        self.shuffle_seed = shuffle_seed

        self.evaluate_during_training = evaluate_during_training

        self.huber_delta = huber_delta
        self.flooding = flooding
        self.flood_level = flood_level
        self.cleanup_on_exception = cleanup_on_exception
        self.show_running_loss = show_running_loss

        self.use_amp_training = use_amp_training
        self.use_amp_inference = use_amp_inference

        self.pad_to_mult = pad_to_mult
        self.display_interval_secs = display_interval_secs
        self.partial_forward_type = partial_forward_type
        self.use_wandb = use_wandb
        self.wandb_init_args = wandb_init_args
        self.use_galileo = use_galileo
        self.galileo_separate_runs_for_epochs = galileo_separate_runs_for_epochs

        self.target_cols_ = None

        self.lr_ = None
        self.opt_ = None
        self.sched_ = None
        self.scaler_ = None
        self.grad_clip_ = None

        self.lr_calib_ = None

        self.X_train_, self.y_train_, self.X_val_, self.y_val_ = None, None, None, None

        self.model_ = None

        for k in kwargs:
            print(f"\tSkipping constructor arg {k}")

    @property
    def base_model(self):
        return self._base_model()

    def _setup(self, X=None, y=None, training=True):
        print("entering setup")

        print("making model")

        self.model_ = NostARHead(
            base_model=self.base_model,
            tokenizer=self.tokenizer,
            params=self.params,
            partial_forward_type=self.partial_forward_type,
            initialize_weights=training,
            params_extras=self.params_extras,
        )
        self.model_ = self.model_.to(self.device)

        parameter_count = sum(
            [np.prod(list(v.shape)) for v in self.model_.parameters()]
        )
        print(
            "This model is using %d parameters (%.2fM)"
            % (parameter_count, parameter_count / (1024.0 * 1024.0))
        )

        if not training:
            return

        print("creating loss fn")
        if self.regression_target:
            self.loss_fn = make_huber_loss_from_logits(huber_delta=self.huber_delta)
        else:
            self.loss_fn = F.cross_entropy
            if self.flooding:
                self.loss_fn = partial(cross_entropy_with_flooding, flood_level=self.flood_level)

        # opt stuff
        print("creating opt")
        self.opt_ = get_nost_ar_head_optimizers(
            self.model_,
            self.opt_params
        )

        self.sched_ = get_nost_ar_head_scheduler(
            self.opt_,
            self.opt_params,
            len(X),
        )

        self.scaler_ = torch.cuda.amp.GradScaler(enabled=self.use_amp_training)

        self.grad_clip_ = self.params_extras.get('grad_clip', 1000.)

    def _make_batched_data(self, X, y=None):
        if y is None:
            data = X.reset_index(drop=True)
            data.index.name = "selector_internal_ix"
            data = data.reset_index()
        else:
            self.target_cols_ = y.columns if len(y.shape) > 1 else y.name
            data = pd.concat([X, y], axis=1)
        if "n_tokens" not in data.columns:
            data["n_tokens"] = data.selector_input.apply(
                lambda s: len(self.tokenizer.encode(s))
            )
        data = data.sort_values(by="n_tokens")
        data = reshuffle_batches(data, batch_size=self.opt_params.batch_size, seed=self.shuffle_seed)
        return data

    def _feed_from_batch(self, data_batch):
        input_ids, attention_mask, input_ids_with_pads = prep_inputs(
            data_batch.selector_input.values,
            self.tokenizer,
            max_length=self.length,
            device=self.base_model.device,
            pad_to_mult=self.pad_to_mult,
        )

        batch_max_tokens = input_ids.shape[1]
        return input_ids, attention_mask, input_ids_with_pads, batch_max_tokens

    def _epoch(self, X, y, avg_loss_beta=0.98):
        self.model_.train()
        for param in self.model_.parameters():
            param.requires_grad = True

        extra_postfixes = {}
        all_losses = []
        running_loss = None

        data = self._make_batched_data(X, y)
        steps = len(data) // self.opt_params.batch_size

        # data pass
        target_dtype = torch.float if self.regression_target else torch.long

        row_ix = 0
        step_iter = tqdm(
            list(range(0, steps)), smoothing=0.0, miniters=1, mininterval=self.display_interval_secs
        )
        epoch_data = []
        for step_ix in step_iter:
            data_batch = data.iloc[row_ix : row_ix + self.opt_params.batch_size, :]

            input_ids, attention_mask, input_ids_with_pads, batch_max_tokens = self._feed_from_batch(
                data_batch
            )

            batch_target = (
                data_batch[self.target_cols_].values
                if len(self.target_cols_) > 1
                else data_batch[self.target_cols_[0]].values
            )

            batch_target = torch.as_tensor(batch_target, dtype=target_dtype).pin_memory().to(self.device)

            batch_dq_id = None
            if self.use_galileo:
                batch_dq_id = data_batch['dq_id'].values

            epoch_data.append(
                {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "input_ids_with_pads": input_ids_with_pads,
                    "batch_max_tokens": batch_max_tokens,
                    "batch_target": batch_target,
                    "batch_dq_id": batch_dq_id
                }
            )
            row_ix += self.opt_params.batch_size

        # train pass
        row_ix = 0
        step_iter = tqdm(
            epoch_data, smoothing=0.0, miniters=1, mininterval=self.display_interval_secs
        )
        max_tokens_so_far = 0
        for step_ix, batch_data in enumerate(step_iter):
            input_ids = batch_data["input_ids"]
            attention_mask = batch_data["attention_mask"]
            input_ids_with_pads = batch_data["input_ids_with_pads"]
            batch_max_tokens = batch_data["batch_max_tokens"]
            batch_target = batch_data["batch_target"]
            batch_dq_id = batch_data["batch_dq_id"]

            # TODO: figure out whether we need logits in float32 explicitly
            outs = self.model_(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_ids_with_pads=input_ids_with_pads,
                autocast=self.use_amp_training,
                return_embs=self.use_galileo,
            )

            if self.use_galileo:
                logits, embs = outs
            else:
                logits = outs

            if self.use_galileo:
                import dataquality as dq

                dq.log_model_outputs(embs=embs, logits=logits, ids=batch_dq_id)

            loss = self.loss_fn(input=logits, target=batch_target)

            self.scaler_.scale(loss).backward()

            self.scaler_.unscale_(self.opt_)

            grad_norm = torch.nn.utils.clip_grad_norm_(self.model_.parameters(),
                                                       max_norm=self.grad_clip_)

            self.scaler_.step(self.opt_)
            self.scaler_.update()

            self.opt_.zero_grad()

            self.sched_.step()

            loss_float = None
            cur_lr = None
            if self.show_running_loss:
                loss_float = loss.detach().item()
                all_losses.append(loss_float)

                cur_lr = self.sched_.state_dict()['_last_lr'][0]

                grad_norm_float = grad_norm.item()

            del loss
            del logits
            del batch_data

            attn_gain, mlp_gain = None, None
            if len(self.model_.blocks) > 0 and self.model_.params.use_block_out_gain:
                attn_gain = (self.model_.blocks[0].attn_gain * self.model_.blocks[0].gain_scale).exp().item()
                mlp_gain = (self.model_.blocks[0].mlp_gain * self.model_.blocks[0].gain_scale).exp().item()

            if self.show_running_loss:
                if running_loss is None:
                    if step_ix > 3:
                        running_loss = sum(all_losses) / len(all_losses)
                else:
                    running_loss = (avg_loss_beta * running_loss) + (
                        (1 - avg_loss_beta) * loss_float
                    )

            if self.use_wandb and self.show_running_loss:
                import wandb

                wandb.log(
                    {
                        f'train/loss': float(loss_float),
                        f'train/ntok': batch_max_tokens,
                        f'train/lr': cur_lr,
                        f'train/grad_norm': grad_norm_float,
                        f'train/attn_gain': attn_gain,
                        f'train/mlp_gain': mlp_gain,
                    }
                )

            max_tokens_so_far = max(max_tokens_so_far, batch_max_tokens)
            extra_postfixes["ntok"] = batch_max_tokens
            extra_postfixes["ntok_max"] = max_tokens_so_far

            extra_postfixes["attn_gain"] = attn_gain
            extra_postfixes["mlp_gain"] = mlp_gain

            step_iter.set_postfix(
                loss=loss_float,
                loss_avg=running_loss,
                lr=cur_lr,
                gnorm=grad_norm_float,
                refresh=False,
                **extra_postfixes,
            )
            row_ix += self.opt_params.batch_size

    def _val_split(self, X, y):
        if self.calibrate or self.evaluate_during_training:
            if self.calibration_split_type == "tts":
                stratifier = (y > 0) if self.regression_target else y
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, stratify=stratifier, test_size=self.calibration_val_size,
                    random_state=self.calibration_val_seed,
                )
            elif self.calibration_split_type == "ttsp":
                y_stratifier = (y > 0) if self.regression_target else y
                stratifier = X["prompt_finalchar"] + y_stratifier.apply(str)
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y, stratify=stratifier, test_size=self.calibration_val_size,
                    random_state=self.calibration_val_seed,
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

    def _init_dq_run(self, dq_run_name):
        if self.use_galileo:
            import dataquality as dq

            dq.init(task_type="text_classification", project_name="nbar_heads", run_name=dq_run_name)

            X_train, y_train = self.X_train_, self.y_train_
            X_val, y_val = self.X_val_, self.y_val_

            X_train["dq_id"] = np.arange(len(X_train))
            dq_df_train = X_train.copy()
            if self.regression_target:
                label_for_dq = y_train
            else:
                label_for_dq = (y_train > 0).astype(int)
                dq.set_labels_for_run([0, 1])
            dq_df_train['label'] = label_for_dq
            dq.log_dataset(dq_df_train, id="dq_id", text="selector_input", split="train")

            if X_val is not None:
                X_val["dq_id"] = np.arange(len(X_train), len(X_train) + len(X_val))
                dq_df_val = X_val.copy()
                label_for_dq = (y_val > 0).astype(int) if self.regression_target else y_val
                dq_df_val['label'] = label_for_dq
                dq.log_dataset(dq_df_val, id="dq_id", text="selector_input", split="validation")

            # dq.set_labels_for_run([0, 1])

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

        all_eval_metrics_results_pfc = {}
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
                all_eval_metrics_results_pfc[pfc] = eval_metrics_results_pfc
                with tqdm(
                    list(range(0, 1)), smoothing=0.0, miniters=1, mininterval=1
                ) as fake_iter_pfc:
                    fake_iter_pfc.set_postfix(**eval_metrics_results_pfc)

        if self.use_wandb:
            import wandb

            wandb.log(
                {f"val/{k}": float(v) for k, v in eval_metrics_results.items()},
                commit=False
            )

            for pfc, metrics in all_eval_metrics_results_pfc.items():
                wandb.log(
                    {f"val/{pfc}/{k}": float(v) for k, v in metrics.items() if not isinstance(v, str)},
                    commit=False
                )

        return eval_metrics_results

    def eval_on_val_set(self, X_val, y_val, disable_calibration=True):
        stop_early_signal = None

        probs = self._predict(X_val, key="probs", disable_calibration=True, training=True)
        preds = probs[:, 1] > 0.5

        eval_metrics_results = self._display_eval_metrics(
            y_val, preds, probs, pfcs=X_val["prompt_finalchar"]
        )

        return stop_early_signal, eval_metrics_results

    def _calib_inputs(self, logits):
        logit_diff = logits[:, 1:] - logits[:, :1]
        return logit_diff

    @property
    def _calib_kwargs(self):
        return {} if self.regression_target else {"penalty": "none"}

    def _fit_calibration(self, X_val, y_val):
        logits = self._predict(X_val, key="logits", disable_calibration=True)
        calib_inputs = self._calib_inputs(logits)

        if not self.regression_target:
            probs = scipy.special.softmax(logits, axis=1)
            preds = probs[:, 1] > 0.5
            self._display_eval_metrics(y_val, preds, probs, pfcs=X_val["prompt_finalchar"])

        lr_cls = LinearRegression if self.regression_target else LogisticRegression
        self.lr_calib_ = lr_cls(**self._calib_kwargs)
        self.lr_calib_.fit(calib_inputs, y_val)

        calib_coef_info = {
            "coef": self.lr_calib_.coef_.tolist(),
            "intercept": self.lr_calib_.intercept_,
        }

        with tqdm(
            list(range(0, 1)), smoothing=0.0, miniters=1, mininterval=1
        ) as fake_iter:
            fake_iter.set_postfix(**calib_coef_info)

    @property
    def train_val_sizes_(self):
        return {
            "train": len(self.X_train_),
            "val": 0 if self.X_val_ is None else len(self.X_val_),
        }

    def fit(self, X, y, avg_loss_beta=0.99):
        if self.use_wandb:
            import wandb

            wandb_init_args = {"project": "nbar-heads"}
            if self.wandb_init_args is not None:
                wandb_init_args.update(self.wandb_init_args)

            wandb.init(**wandb_init_args)


        dq_run_name = ''.join(random.choices(ascii_lowercase, k=8))

        if self.use_galileo:
            import dataquality as dq
            dq.login()

        try:
            self.X_train_, self.y_train_, self.X_val_, self.y_val_ = self._val_split(
                X, y
            )
            if not self.galileo_separate_runs_for_epochs:
                self._init_dq_run(dq_run_name)

            self._setup(self.X_train_, self.y_train_, training=True)
            for epoch_ix in tqdm(list(range(self.opt_params.epochs))):
                if self.use_galileo:
                    if self.galileo_separate_runs_for_epochs:
                        self._init_dq_run(dq_run_name + f"_epoch{epoch_ix}")
                        dq.set_epoch(0)
                    else:
                        dq.set_epoch(epoch_ix)

                    dq.set_split('train')

                self._epoch(self.X_train_, self.y_train_, avg_loss_beta=avg_loss_beta)

                epoch_needs_val = self.evaluate_during_training

                if epoch_needs_val:
                    dq.set_split('validation')
                    stop_early_signal, eval_metrics_results = self.eval_on_val_set(
                        self.X_val_, self.y_val_
                    )
                    if stop_early_signal:
                        print(f"stopping early at {epoch_ix}")
                        break

                if self.galileo_separate_runs_for_epochs:
                    dq.finish(wait=False)
            if self.calibrate:
                self._fit_calibration(self.X_val_, self.y_val_)
        except (Exception, KeyboardInterrupt) as e:
            if self.cleanup_on_exception:
                self.cleanup()
            if self.use_galileo:
                dq.finish(wait=False)
            raise e
        if self.use_wandb:
            wandb.log({})  # commits final val
        if self.use_galileo and not self.galileo_separate_runs_for_epochs:
            dq.finish(wait=False)
        return self

    def _compute_calib_probs(self, logits, pfcs):
        calib_inputs = self._calib_inputs(logits)
        predict_method = 'predict' if self.regression_target else 'predict_proba'
        result = getattr(self.lr_calib_, predict_method)(calib_inputs)
        return result

    def _predict_select(self, batch, threshold=0.5, disable_calibration=False, autocast=True, training=False):
        self.model_.eval()
        for param in self.model_.parameters():
            param.requires_grad = False

        input_ids, attention_mask, input_ids_with_pads, _ = self._feed_from_batch(batch)

        use_galileo = self.use_galileo and training

        # TODO: figure out whether we need logits in float32 explicitly
        with kv_buffer_scope(self.base_model, False):
            outs = self.model_(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_ids_with_pads=input_ids_with_pads,
                autocast=self.use_amp_inference,
                return_embs=use_galileo,
            )

        if use_galileo:
            logits_raw, embs = outs
        else:
            logits_raw = outs

        if use_galileo:
            import dataquality as dq

            batch_dq_id = batch['dq_id'].values

            if len(np.unique(batch_dq_id)) != len(batch_dq_id):
                print(f"multiple dq ids in batch, skipping dq log: {batch_dq_id}")
            else:
                dq.log_model_outputs(embs=embs, logits=logits_raw, ids=batch_dq_id)

        logits_raw = logits_raw.cpu().detach().numpy()

        if self.regression_target and (self.calibrate and not disable_calibration):
            logits = self._compute_calib_probs(logits_raw, pfcs=batch["prompt_finalchar"])
            logits = np.stack([-logits / 2, logits / 2], axis=1)
        else:
            logits = logits_raw

        probs_raw = scipy.special.softmax(logits_raw, axis=1)

        if (not self.regression_target) and (self.calibrate and not disable_calibration):
            probs = self._compute_calib_probs(logits, pfcs=batch["prompt_finalchar"])
        else:
            probs = scipy.special.softmax(logits, axis=1)
        results = {"logits": logits, "probs": probs, "probs_raw": probs_raw}
        results["preds"] = probs[:, 1] > threshold
        return results

    def _predict(self, X, key="preds", disable_calibration=False, suppress_tqdm=False, training=False):
        if isinstance(X, list):
            X = pd.DataFrame.from_records(X)

        all_pd_ixs = []
        all_preds = []
        data = self._make_batched_data(X)
        steps = len(data) // self.opt_params.batch_size + 1

        row_ix = 0

        step_iter = (
            tqdm(list(range(0, steps)), smoothing=0.0, miniters=1, mininterval=self.display_interval_secs)
            if (steps > 1) and (not suppress_tqdm)
            else list(range(0, steps))
        )

        # move to gpu for use
        base_layers_moved = []
        offset = 1
        for block in self.model_.blocks:
            layer_num = self.model_.last_base_layer_used + offset + 1
            base_layer = self.base_model.transformer.h[layer_num]
            base_layer.to(device=self.device)
            base_layers_moved.append(layer_num)
            offset += 1

        # move an additional base layer to make room for the head part after the block(s)
        layer_num = self.model_.last_base_layer_used + offset + 1
        base_layer = self.base_model.transformer.h[layer_num]
        base_layer.to(device=self.device)
        base_layers_moved.append(layer_num)

        self.model_.cuda()

        # predict
        for step_ix in step_iter:
            data_batch = data.iloc[row_ix : row_ix + self.opt_params.batch_size, :]
            n_needed = len(data_batch)
            if n_needed == 0:
                continue
            if n_needed < self.opt_params.batch_size and not training:
                data_batch = pd.concat(
                    [data_batch]
                    + (self.opt_params.batch_size - n_needed) * [data_batch.iloc[:1, :]],
                    ignore_index=True,
                )

            results_batch = self._predict_select(
                data_batch, disable_calibration=disable_calibration, training=training,
            )
            all_preds.extend(results_batch[key][:n_needed])
            all_pd_ixs.extend(data_batch["selector_internal_ix"].tolist()[:n_needed])

            row_ix += self.opt_params.batch_size

        # move back to orig devices
        self.model_.to(device=self.device)
        for layer_num in base_layers_moved:
            base_layer = self.base_model.transformer.h[layer_num]
            base_layer.cuda()

        if key == "preds":
            pd_obj = pd.Series(all_preds, index=all_pd_ixs)
        else:
            pd_obj = pd.DataFrame(all_preds, index=all_pd_ixs)
        pd_obj = pd_obj.sort_index()
        return pd_obj.values

    def predict(self, X, suppress_tqdm=False):
        # TODO: make this less of a shitty hack
        return self._predict(X, key="preds" if not self.regression_target else "logits", suppress_tqdm=suppress_tqdm)

    def predict_proba(self, X, suppress_tqdm=False):
        return self._predict(X, key="probs", suppress_tqdm=suppress_tqdm)

    def decision_function(self, X, suppress_tqdm=False):
        return self._predict(X, key="probs", suppress_tqdm=suppress_tqdm)

    def cleanup(self):
        print("cleanup: deleting state")
        to_delete = list(self.model_.parameters())
        to_delete += list(self.opt_.state)
        for p in to_delete:
            del p
        gc.collect()

    def _make_constructor_args(self):
        # TODO: create mixin for this
        sig = inspect.signature(self.__class__.__init__)
        args = {k: getattr(self, k) for k in sig.parameters.keys() if hasattr(self, k)}
        return args

    def save(self, path: str):
        no_save_args = {"tokenizer", "base_model"}

        metadata = {
            "constructor_args": {
                name: value
                for name, value in self._make_constructor_args().items()
                if name not in no_save_args
            },
        }

        metadata["constructor_args"]["params"] = typed_namedtuple_to_dict(metadata["constructor_args"]["params"])
        metadata["constructor_args"]["opt_params"] = typed_namedtuple_to_dict(metadata["constructor_args"]["opt_params"])

        with open(os.path.join(path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=1)

        state_dict_path = os.path.join(path, "state_dict.pt")
        torch.save(self.model_.state_dict(), state_dict_path)

        joblib.dump(self.lr_calib_, os.path.join(path, "lr_calib.pkl.gz"))

    @staticmethod
    def load(path, base_model, tokenizer,
             inference_batch_size=None,
             use_amp_inference=True,
             blocks_inference_device_attn=None,
             blocks_inference_device_mlp=None,
             **kwargs) -> "NostARHeadEstimator":
        with open(os.path.join(path, "metadata.json"), "r") as f:
            metadata = json.load(f)

        constructor_args = metadata["constructor_args"]
        constructor_args["base_model"] = base_model
        constructor_args["tokenizer"] = tokenizer
        constructor_args["use_amp_inference"] = use_amp_inference
        constructor_args["blocks_inference_device_attn"] = blocks_inference_device_attn
        constructor_args["blocks_inference_device_mlp"] = blocks_inference_device_mlp

        if inference_batch_size is not None:
            constructor_args["opt_params"]["batch_size"] = inference_batch_size

        if "proj_ratio" not in constructor_args["params"]:
            constructor_args["params"]["proj_ratio"] = 1  # TODO: remove after next model save

        # using namedtuple was a mistake :(
        extras_defaults = {
            'block_lr': None,
            'decay_ratio': None,
            'no_weight_decay_in_blocks': True,
            'gain_scale_blocks_out': 1.,
            'init_gain_blocks': 1.,
            'init_gain_blocks_out': 1.,
            'mlp_only_blocks': False,
            'mlp_ratio_blocks': 4,
            'n_blocks': 0,
            'n_head_blocks': 16,
            'no_orth_init_in_final_mlp': False,
            'qk_dim_blocks': 4096,
            'qk_dim_final': 4096,
            'rotary_blocks': False,
            'rotary_dim_blocks': 32,
            'tune_base_block_attn': False,
            'tune_base_block_mlp': False,
            'use_block_out_gain': False,
            'use_final_mlp': True,
            'v_dim_final': 4096
        }

        init_args = inspect.signature(NostARHeadArchitectureParams.__new__).parameters.keys()
        for k in set(init_args).intersection(extras_defaults.keys()).difference(constructor_args["params"].keys()):
            constructor_args["params"][k] = extras_defaults[k]

        init_args = inspect.signature(NostARHeadOptimizerParams.__new__).parameters.keys()
        for k in set(init_args).intersection(extras_defaults.keys()).difference(constructor_args["opt_params"].keys()):
            constructor_args["opt_params"][k] = extras_defaults[k]

        constructor_args["params"] = NostARHeadArchitectureParams(**constructor_args["params"])
        constructor_args["opt_params"] = NostARHeadOptimizerParams(**constructor_args["opt_params"])

        constructor_args.update(**kwargs)

        est = NostARHeadEstimator(**constructor_args)
        est._setup(training=False)

        state_dict_path = os.path.join(path, "state_dict.pt")
        est.model_.load_state_dict(torch.load(state_dict_path, map_location=constructor_args['device']))

        est.lr_calib_ = joblib.load(os.path.join(path, "lr_calib.pkl.gz"))

        return est
