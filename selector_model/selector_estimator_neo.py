# TODO:
# - better name for "prompt_finalchar"
import inspect
import gc

import numpy as np
import pandas as pd
from tqdm.autonotebook import tqdm
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

import torch
import torch.nn.functional as F
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoForCausalLM

from selector_model.selector_nn_neo import NostARHead, NostARHeadArchitectureParams, GPT2TokenizerType, prep_inputs
from selector_model.selector_utils_neo import NostARHeadOptimizerParams, get_nost_ar_head_optimizers, get_nost_ar_head_scheduler

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


def reshuffle_batches(train_data_for_selection, batch_size):
    train_data_for_selection = train_data_for_selection.sort_values(by="n_tokens")
    batches = [
        train_data_for_selection.iloc[row_ix : row_ix + batch_size, :]
        for row_ix in range(0, len(train_data_for_selection), batch_size)
    ]

    np.random.shuffle(batches)

    return pd.concat(batches, ignore_index=True)


class NostARHeadEstimator(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        base_model: GPTNeoForCausalLM,  # TODO: make compat with GPTNeoModel, etc?
        tokenizer: GPT2TokenizerType,
        params: NostARHeadArchitectureParams,
        opt_params: NostARHeadOptimizerParams,
        device='cuda:0',
        use_logit_diff_basis=False,  # TODO: figure these out for sentiment
        use_only_logit_diff=False,  # TODO: figure these out for sentiment
        length=None,
        uid_override=None,
        supervise_logits=False,  # TODO: figure these out for sentiment
        supervise_only_logit_diff=False,  # TODO: figure these out for sentiment
        calibrate=True,
        calibration_val_size=0.1,
        calibration_split_type="ttsp",
        evaluate_during_training=True,
        huber=False,  # TODO: figure these out for sentiment
        huber_delta=1.0,  # TODO: figure these out for sentiment
        flooding=False,  # TODO: implement
        flood_level=0.0,  # TODO: implement
        classic_init=False,  # TODO: implement
        cleanup_on_exception=True
    ):
        self.device = device
        self.model = NostARHead(
            base_model=base_model,
            tokenizer=tokenizer,
            params=params
        )
        self.model.to(self.device)
        self.tokenizer = tokenizer
        self.opt_params = opt_params

        parameter_count = sum(
            [np.prod(list(v.shape)) for v in self.model.parameters()]
        )
        print(
            "This model is using %d parameters (%.2fM)"
            % (parameter_count, parameter_count / (1024.0 * 1024.0))
        )

        self.use_logit_diff_basis = use_logit_diff_basis
        self.use_only_logit_diff = use_only_logit_diff

        self.length = length

        self.uid_override = uid_override

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

        self.evaluate_during_training = evaluate_during_training

        self.huber = huber
        self.huber_delta = huber_delta
        self.flooding = flooding
        self.flood_level = flood_level
        self.classic_init = classic_init
        self.cleanup_on_exception = cleanup_on_exception

        self.uid_ = None
        self.train_vars_ = None
        self.decay_vars_ = None
        self.selection_step_train_ = None
        self.selection_step_eval_ = None
        self.select_logits_train_ = None
        self.select_logits_eval_ = None
        self.select_target_ = None
        self.select_loss_ = None
        self.target_cols_ = None

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

    def _setup(self, X=None, y=None, training=True):
        print("entering setup")
        print(f"estimator:\n{repr(self)}\n")

        print("making losses")

        if self.supervise_logits:
            raise NotImplementedError
        else:
            self.loss_fn = F.cross_entropy
            print("setting up vars")
            # self.train_vars_ = [var for var in selector_vars if "ln_" not in var.name]

            self.decay_vars_ = []
            self.non_decay_vars_ = []

            for name, param in self.model.named_parameters():
                if "ln.weight" in name or "logit_head.bias" in name:
                    print(f"assigning '{name}' to non_decay_vars_")
                    self.non_decay_vars_.append(param)
                else:
                    print(f"assigning '{name}' to decay_vars_")
                    self.decay_vars_.append(param)

            if not training:
                # done
                return None

            # opt stuff

            print("creating opt")
            self.opt_decay_, self.opt_no_decay_ = get_nost_ar_head_optimizers(
                self.model,
                self.opt_params
            )

            self.sched_decay_ = get_nost_ar_head_scheduler(
                self.opt_decay_,
                self.opt_params,
                len(X) // self.opt_params.batch_size
            )

            self.sched_no_decay_ = get_nost_ar_head_scheduler(
                self.opt_no_decay_,
                self.opt_params,
                len(X) // self.opt_params.batch_size
            )

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
                lambda s: len(self.enc.encode(s))
            )
        data = data.sort_values(by="n_tokens")
        data = reshuffle_batches(data, batch_size=self.opt_params.batch_size)
        return data

    def _feed_from_batch(self, data_batch):
        input_ids, attention_mask, input_ids_with_pads = prep_inputs(
            data_batch.selector_input.values,
            self.tokenizer,
            max_length=self.length,
            device=self.device
        )

        batch_max_tokens = input_ids.shape[1]
        return input_ids, attention_mask, input_ids_with_pads, batch_max_tokens

    def _epoch(self, X, y, avg_loss_beta=0.98):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        extra_postfixes = {}
        all_losses = []
        running_loss = None

        data = self._make_batched_data(X, y)
        steps = len(data) // self.opt_params.batch_size

        row_ix = 0
        step_iter = tqdm(
            list(range(0, steps)), smoothing=0.0, miniters=1, mininterval=3
        )
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

            batch_target = torch.as_tensor(batch_target).to(self.device)

            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_ids_with_pads=input_ids_with_pads,
            )

            loss = self.loss_fn(input=logits, target=batch_target)
            loss.backward()

            self.opt_decay_.step()
            self.opt_no_decay_.step()

            self.opt_decay_.zero_grad()
            self.opt_no_decay_.zero_grad()

            self.sched_decay_.step()
            self.sched_no_decay_.step()

            loss_float = loss.detach().item()

            del loss
            del logits
            del batch_target

            all_losses.append(loss_float)
            if running_loss is None:
                if step_ix > 3:
                    running_loss = sum(all_losses) / len(all_losses)
            else:
                running_loss = (avg_loss_beta * running_loss) + (
                    (1 - avg_loss_beta) * loss_float
                )

            extra_postfixes["ntok"] = batch_max_tokens

            step_iter.set_postfix(
                loss=loss_float,
                loss_avg=running_loss,
                # lr=cur_lr,
                refresh=False,
                **extra_postfixes,
            )
            row_ix += self.opt_params.batch_size

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

            self.lr_calib_resp_ = self.lr_calib_
            self.lr_calib_orig_ = self.lr_calib_

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
        try:
            self.X_train_, self.y_train_, self.X_val_, self.y_val_ = self._val_split(
                X, y
            )
            self._setup(self.X_train_, self.y_train_, training=True)
            for epoch_ix in tqdm(list(range(self.opt_params.epochs))):
                self._epoch(self.X_train_, self.y_train_, avg_loss_beta=avg_loss_beta)

                epoch_needs_val = self.evaluate_during_training

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
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        if len(batch) != self.opt_params.batch_size:
            raise ValueError("badlength")
        input_ids, attention_mask, input_ids_with_pads, _ = self._feed_from_batch(batch)

        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_ids_with_pads=input_ids_with_pads,
        )

        probs_raw = scipy.special.softmax(logits.cpu().detach().numpy(), axis=1)

        if self.calibrate and not disable_calibration:
            probs = self._compute_calib_probs(logits, pfcs=batch["prompt_finalchar"])
        else:
            probs = probs_raw
        results = {"logits": logits, "probs": probs, "probs_raw": probs_raw}
        results["preds"] = probs[:, 1] > threshold
        return results

    def _predict(self, X, key="preds", disable_calibration=False):
        if isinstance(X, list):
            X = pd.DataFrame.from_records(X)

        all_pd_ixs = []
        all_preds = []
        data = self._make_batched_data(X)
        steps = len(data) // self.opt_params.batch_size + 1

        row_ix = 0

        step_iter = (
            tqdm(list(range(0, steps)), smoothing=0.0, miniters=1, mininterval=1)
            if steps > 1
            else list(range(0, steps))
        )

        for step_ix in step_iter:
            data_batch = data.iloc[row_ix : row_ix + self.opt_params.batch_size, :]
            n_needed = len(data_batch)
            if n_needed == 0:
                continue
            if n_needed < self.opt_params.batch_size:
                data_batch = pd.concat(
                    [data_batch]
                    + (self.opt_params.batch_size - n_needed) * [data_batch.iloc[:1, :]],
                    ignore_index=True,
                )

            results_batch = self._predict_select(
                data_batch, disable_calibration=disable_calibration
            )
            all_preds.extend(results_batch[key][:n_needed])
            all_pd_ixs.extend(data_batch["selector_internal_ix"].tolist()[:n_needed])

            row_ix += self.opt_params.batch_size

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
        print("cleanup: deleting state")
        to_delete = list(self.model.parameters())
        to_delete += list(self.opt_decay_.state)
        to_delete += list(self.opt_no_decay_.state)
        for p in to_delete:
            del p
        gc.collect()

    def _make_constructor_args(self):
        # TODO: create mixin for this
        sig = inspect.signature(self.__class__.__init__)
        args = {k: getattr(self, k) for k in sig.parameters.keys() if hasattr(self, k)}
        return args

    def save(self, path: str):
        raise NotImplementedError

    @staticmethod
    def load(path, session, base_hparams, enc, **kwargs) -> "NostARHeadEstimator":
        raise NotImplementedError
