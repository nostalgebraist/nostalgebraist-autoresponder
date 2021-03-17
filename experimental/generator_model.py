from typing import NamedTuple
from textwrap import wrap

import tensorflow as tf
import tflex

import model
import sample

from autoresponder_config import *  # TODO: move elsewhere?


def typed_namedtuple_to_dict(tup: NamedTuple) -> dict:
  return {name: getattr(tup, name) for name in tup._fields}


def copy_and_update_config(cls, config, **kwargs):
    old_d = typed_namedtuple_to_dict(config)
    new_d = {k: kwargs.get(k) if k in kwargs else v for k, v in old_d.items()}
    return cls(**new_d)


def is_repeating_criterion(unique_token_frac):
    return unique_token_frac < 0.2


SamplingParams = NamedTuple(
    'SamplingParams',
    temperature=float,
    top_k=int,
    top_p=float,
    middle_p=float,
    chop_lowest=float,
    chop_highest=float,
    mirostat=bool,
    breakruns=bool,
    breakruns_tau=float,
    breakruns_decay=float
)


SamplingConfig = NamedTuple(
    'SamplingConfig',
    pre_continue_length=int,
    post_window_length=int,
    pre_continue_params=SamplingParams,
    params=SamplingParams,
    disable_prints=bool,
    max_ctx_fits_on_gpu=int,
    max_continue_steps=int,
    max_continue_tokens=int,
    mirostat_lr=float,
    mirostat_v2=bool,
    mirostat_trunc=int,
    miro_only_on_continue=bool,
)


DEFAULT_SAMPLING_CONFIG = SamplingConfig(
    pre_continue_params=SamplingParams(
        temperature=pre_continue_temperature,
        top_k=pre_continue_top_k,
        top_p=pre_continue_top_p,
        middle_p=pre_continue_middle_p,
        chop_lowest=pre_continue_chop_lowest,
        chop_highest=pre_continue_chop_highest,
        mirostat=pre_continue_mirostat,
        breakruns=BREAKRUNS,
        breakruns_tau=BREAKRUNS_TAU,
        breakruns_decay=BREAKRUNS_DECAY
    ),
    params=SamplingParams(
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        middle_p=middle_p,
        chop_lowest=chop_lowest,
        chop_highest=chop_highest,
        mirostat=MIRO,
        breakruns=BREAKRUNS,
        breakruns_tau=BREAKRUNS_TAU,
        breakruns_decay=BREAKRUNS_DECAY
    ),
    disable_prints=True,
    pre_continue_length=pre_continue_length,
    post_window_length=length,
    max_ctx_fits_on_gpu=max_ctx_fits_on_gpu,
    max_continue_steps=MAX_CONTINUE_STEPS,
    max_continue_tokens=MAX_CONTINUE_TOKENS,
    mirostat_lr=MIRO_LR,
    mirostat_v2=MIRO_V2,
    mirostat_trunc=MIRO_TRUNC,
    miro_only_on_continue=MIRO_ONLY_ON_CONTINUE,
)


class GeneratorModel:
    def __init__(
        self,
        enc,
        batch_size,
        sample_done_criterion,
        sampling_config: SamplingConfig=DEFAULT_SAMPLING_CONFIG,
        hparams=model.hparams_1558M(),
        session=None,
    ):
        self.enc = enc
        self.batch_size = batch_size
        self.sample_done_criterion = sample_done_criterion
        self.sampling_config = sampling_config
        self.hparams = hparams

        self.session = session

        # tf placeholders
        self.context = None

        self.sample_pasts = None

        self.mirostat_target = None
        self.mirostat_lr = None
        self.mirostat_mu_from_past = None

        # tf ops
        self.first_sample_op = None
        self.second_sample_op = None
        self.presents_op = None

        # cache
        self.startup_presents_for_prompt = {}

    def _setup(self, reset=False):
        if reset:
            tf.reset_default_graph()

        if self.session is None:
            self.session = tflex.Session()

        with self.session.as_default():
            self.context = tf.placeholder(tf.int32, [self.batch_size, None])
            self.sample_pasts = tf.placeholder(
                tf.float32,
                model.past_shape(hparams=self.hparams, batch_size=self.batch_size),
            )

            self.mirostat_target = tf.placeholder(shape=[], dtype=tf.float32)
            self.mirostat_lr = tf.placeholder(shape=[], dtype=tf.float32)
            self.mirostat_mu_from_past = tf.placeholder(
                tf.float32,
                [
                    self.batch_size,
                ],
            )

            sampling_args = dict(
                stop_at_EOT=True,
                eot_workaround=True,
                enc=self.enc,
                hparams=self.hparams,
                start_token=None,
                context=self.context,
                batch_size=self.batch_size,
                return_presents=True,
                pasts=self.sample_pasts,
                mirostat_surprise_target=self.mirostat_target,
                mirostat_lr=self.mirostat_lr,
                mirostat_v2=self.sampling_config.mirostat_v2,
                mirostat_trunc=self.sampling_config.mirostat_trunc,
                disable_prints=self.sampling_config.disable_prints,
            )

            self.first_sample_op = sample.sample_sequence(
                length=self.sampling_config.pre_continue_length,
                better_length=False,
                **typed_namedtuple_to_dict(self.sampling_config.pre_continue_params),
                **sampling_args,
            )
            # TODO: DRY
            self.sample_op_fill_window = sample.sample_sequence(
                length=self.sampling_config.max_ctx_fits_on_gpu,
                better_length=True,
                **typed_namedtuple_to_dict(self.sampling_config.params),
                **sampling_args,
            )
            # TODO: DRY
            self.sample_op_beyond_window = sample.sample_sequence(
                length=self.sampling_config.post_window_length,
                better_length=False,
                **typed_namedtuple_to_dict(self.sampling_config.params),
                **sampling_args,
            )
            self.presents_op = model.model(hparams=self.hparams, X=self.context)[
                "present"
            ]

    def done_writing(self, prompt: str):
        if prompt in self.startup_presents_for_prompt:
            del self.startup_presents_for_prompt[prompt]

    def write_random_prompt(
        self, prompts: list, probs: list, mirotarg: float = None, verbose=False
    ):
        if mirotarg is None:
            mirotarg = np.random.choice(MIRO_TARGET_ALL)

        prompt = np.random.choice(prompts, p=np.array(probs) / sum(probs))
        return self.write(prompt=prompt, mirotarg=mirotarg, verbose=verbose)

    def write(self, prompt: str, mirotarg: float = None, stop_repeats=True, verbose=False):
        if mirotarg is None:
            mirotarg = np.random.choice(MIRO_TARGET_ALL)

        context_tokens = self.enc.encode(prompt)

        startup_presents = self.startup_presents_for_prompt.get(prompt, None)

        max_context_size = self.sampling_config.max_ctx_fits_on_gpu - self.sampling_config.post_window_length
        if len(context_tokens) > max_context_size:
            orig_len = len(context_tokens)
            context_tokens = context_tokens[-(max_context_size):]
            print(
                f"truncated {orig_len} to {len(context_tokens)}, max_context_size={max_context_size}"
            )
        else:
            print(
                f"{len(context_tokens)} tokens can fit in max_context_size {max_context_size}"
            )

        token_start_ix = len(context_tokens)

        batch_context_tokens = [context_tokens for _ in range(self.batch_size)]
        continuations = [[prompt] for _ in batch_context_tokens]
        continuations_tokens = [[context_tokens] for _ in range(self.batch_size)]
        is_repeating = [False for _ in batch_context_tokens]
        is_not_finished = [True for _ in batch_context_tokens]
        generated = 0
        tokens_generated = 0
        this_batch_continue_steps = 0

        first_step_with_miro = 1 if self.sampling_config.miro_only_on_continue else 0

        done = False
        recompute_presents = False
        if startup_presents is None:
            print("computing startup presents")
            startup_presents = self.session.run(
                self.presents_op,
                feed_dict={self.context: [bct[:-1] for bct in batch_context_tokens]},
            )
            self.startup_presents_for_prompt[prompt] = startup_presents
        presents = startup_presents

        miromu = None
        mirosurprises, miroks, miromus, mirotoks = None, None, None, None
        mu_init_scale = 1.0 if self.sampling_config.mirostat_v2 else 2.0

        while not done:
            recompute_presents = (token_start_ix >= max_context_size) or (
                presents is None
            )
            with self.session.as_default():
                if recompute_presents:
                    print("recomputing presents")
                    presents = self.session.run(
                        self.presents_op,
                        feed_dict={
                            self.context: [bct[:-1] for bct in batch_context_tokens]
                        },
                    )
                    if this_batch_continue_steps >= first_step_with_miro:
                        if miromu is None:
                            miromu = (
                                mu_init_scale * mirotarg * np.ones((self.batch_size,))
                            )
                        print(f"miromu on entry: {miromu}")
                        sample_output_dict = self.session.run(
                            self.sample_op_beyond_window,
                            feed_dict={
                                self.context: batch_context_tokens,
                                self.mirostat_target: mirotarg,
                                self.mirostat_lr: self.sampling_config.mirostat_lr,
                                self.mirostat_mu_from_past: miromu,
                                self.sample_pasts: presents,
                            },
                        )
                    else:
                        sample_output_dict = self.session.run(
                            self.first_sample_op,  # output,
                            feed_dict={
                                self.context: batch_context_tokens,
                                self.mirostat_target: mirotarg,
                                self.mirostat_lr: self.sampling_config.mirostat_lr,
                                self.sample_pasts: presents,
                            },
                        )
                else:
                    print("using saved presents")
                    if this_batch_continue_steps >= first_step_with_miro:
                        if miromu is None:
                            miromu = (
                                mu_init_scale * mirotarg * np.ones((self.batch_size,))
                            )
                        print(f"miromu on entry: {miromu}")
                        sample_output_dict = self.session.run(
                            self.sample_op_fill_window,
                            feed_dict={
                                self.context: batch_context_tokens,
                                self.sample_pasts: presents,
                                self.mirostat_target: mirotarg,
                                self.mirostat_lr: self.sampling_config.mirostat_lr,
                                self.mirostat_mu_from_past: miromu,
                            },
                        )
                    else:
                        sample_output_dict = self.session.run(
                            self.first_sample_op,
                            feed_dict={
                                self.context: batch_context_tokens,
                                self.sample_pasts: presents,
                                self.mirostat_target: mirotarg,
                                self.mirostat_lr: self.sampling_config.mirostat_lr,
                            },
                        )
            sample_output_dict["tokens"] = sample_output_dict["tokens"][
                :, token_start_ix:
            ]
            sample_output_dict["presents"] = sample_output_dict["presents"][
                ..., -(max_context_size - 1) :, :
            ]
            out, presents = sample_output_dict["tokens"], sample_output_dict["presents"]

            if mirosurprises is None or (
                this_batch_continue_steps == first_step_with_miro
            ):
                mirosurprises = sample_output_dict["mirostat_surprises"][:, 1:]
                miroks = sample_output_dict["mirostat_ks"][:, 1:]
                miromus = sample_output_dict["mirostat_mus"][:, 1:]
                mirotoks = sample_output_dict["tokens"]
            else:
                mirosurprises = np.concatenate(
                    [mirosurprises, sample_output_dict["mirostat_surprises"][:, 1:]], axis=1
                )
                miroks = np.concatenate(
                    [miroks, sample_output_dict["mirostat_ks"][:, 1:]], axis=1
                )
                miromus = np.concatenate(
                    [miromus, sample_output_dict["mirostat_mus"][:, 1:]], axis=1
                )
                mirotoks = np.concatenate(
                    [mirotoks, sample_output_dict["tokens"]], axis=1
                )

            print(f"miromu before setting: {miromu}")
            if this_batch_continue_steps >= first_step_with_miro:
                miromu = sample_output_dict["mirostat_mus"][:, -1]
                print(f"miromu after setting: {miromu}")

            miroks = np.clip(miroks, a_min=None, a_max=self.hparams.n_vocab)

            miro_avg_surprises = np.mean(mirosurprises, axis=1)
            miro_median_ks = np.median(miroks, axis=1)
            miro_mean_ks = np.mean(miroks, axis=1)

            tokens_generated += len(out[0])
            unique_token_fracs = []
            for i in range(self.batch_size):
                generated += 1
                text = self.enc.decode(out[i])

                continuations[i].append(text)
                continuations_tokens[i].append(out[i])

                unique_token_frac = len(set(out[i])) / len(out[i])
                unique_token_fracs.append(unique_token_frac)

                is_repeating_this_time = is_repeating_criterion(unique_token_frac) if stop_repeats else False

                if (not is_repeating_this_time) and not is_repeating[i]:
                    is_repeating[i] = False
                else:
                    print(f"{i} is repeating")
                    is_repeating[i] = True

            next_prompts = ["".join(subtexts) for subtexts in continuations]
            batch_context_tokens = [
                np.concatenate(ct)[-(max_context_size):] for ct in continuations_tokens
            ]

            bct_lens = [len(bct) for bct in batch_context_tokens]
            token_start_ix = min(bct_lens)

            next_prompts_contonly = [
                "".join(subtexts[1:]) for subtexts in continuations
            ]
            is_not_finished = [
                not self.sample_done_criterion(c, unique_token_frac)
                for c, unique_token_frac in zip(
                    next_prompts_contonly, unique_token_fracs
                )
            ]
            not_finished = [
                c for c, is_nf in zip(next_prompts_contonly, is_not_finished) if is_nf
            ]
            n_not_finished = len(not_finished)
            more_needed = n_not_finished > 0
            more_permitted = (this_batch_continue_steps < self.sampling_config.max_continue_steps) and (
                tokens_generated < self.sampling_config.max_continue_tokens
            )

            show_miro_logs = self.sampling_config.params.mirostat and (
                (not self.sampling_config.params.miro_only_on_continue)
                or this_batch_continue_steps >= first_step_with_miro
            )

            if show_miro_logs:
                for i in range(self.batch_size):
                    if i == 0:
                        print("\n")
                    finished_mark = "[ ]" if is_not_finished[i] else "[x]"
                    print(
                        f"{finished_mark} {i}: targeting surprise {mirotarg:.3f}, avg surprise {miro_avg_surprises[i]:.3f}, median k {miro_median_ks[i]:.1f}, mean k {miro_mean_ks[i]:.1f}"
                    )

                    if this_batch_continue_steps == first_step_with_miro:
                        print(
                            [
                                (
                                    j,
                                    self.enc.decode([tok]),
                                    mk,
                                    f"{ms:.3f}",
                                    f"{mmu:.3f}",
                                )
                                for j, (tok, mk, ms, mmu) in enumerate(
                                    zip(
                                        out[i],
                                        miroks[i, 1:],
                                        mirosurprises[i].tolist()[1:],
                                        sample_output_dict["mirostat_mus"][i].tolist()[
                                            1:
                                        ],
                                    )
                                )
                            ]
                        )
                    if i == self.batch_size - 1:
                        print()

            done = (not more_needed) or (not more_permitted)
            if not done:
                print("continuing within batch:")
                print(f"\t{n_not_finished}/{len(next_prompts)} unfinished")
                print(
                    f"\t{this_batch_continue_steps}/{self.sampling_config.max_continue_steps} continue steps used"
                )
                print(
                    f"\t{tokens_generated}/{self.sampling_config.max_continue_tokens} continue tokens generated"
                )
                print(
                    f"\tcontext tokens sizes: {[len(ct) for ct in batch_context_tokens]}"
                )

                if verbose:
                    print("Using prompts:")
                    for nep in not_finished:
                        print("\t" + "\n\t".join(wrap(nep, width=90)) + "\n")

                this_batch_continue_steps += 1

        # cleanup
        continuations_ = []
        miro_traces = {
            "surprise": [],
            "mu": [],
            "k": [],
            'tok': [],
        }
        for subtexts, rep, ms, mmu, mk, mtk in zip(
            continuations,
            is_repeating,
            mirosurprises,
            miromus,
            miroks,
            mirotoks
        ):
            text = "".join(subtexts[1:])  # don't return prompt as part of these
            if rep:
                if GLOBAL_DEBUG:
                    print(f"skipping because repeating:\n\n{repr(text)}\n\n")
                continue
            if not text.endswith(eot_end_segment) and eot_end_segment in text:
                text = text.split(eot_end_segment)[0] + eot_end_segment
            continuations_.append(text)
            miro_traces["surprise"].append([float(x) for x in ms])
            miro_traces["mu"].append([float(x) for x in mmu])
            miro_traces["k"].append([float(x) for x in mk])
            miro_traces["tok"].append([int(x) for x in mtk])

        return {
            "continuations": continuations_,
            "side_data": {
                "prompt_for_neural": prompt,
                "mirotarg": mirotarg,
                "miro_traces": miro_traces,
            },
        }

    def restore_checkpoint(self, path, retries=False):
        enclosing_dir = path.rpartition("/")[0]
        ckpt = tflex.latest_checkpoint(enclosing_dir)
        if ckpt is None:
            raise FileNotFoundError
        saver = tflex.Saver()

        load_done = False
        while not load_done:
            try:
                with self.session.as_default():
                    print(f"restoring checkpoint: {ckpt}")
                    saver.restore(self.session, ckpt)
                    load_done = True
            except Exception as e:
                if retries:
                    print(f"encountered {e}, retrying...")
                else:
                    raise e

    def cleanup(self):
        if self.session is not None:
            self.session.close()
            self.session = None
        self.startup_presents_for_prompt = {}

    def set_sampling_config(self, sampling_config: SamplingConfig):
        self.sampling_config = sampling_config
        self._setup(reset=False)

    @staticmethod
    def load(
        path,
        enc,
        batch_size,
        sample_done_criterion,
        sampling_config: SamplingConfig=DEFAULT_SAMPLING_CONFIG,
        hparams=model.hparams_1558M(),
        retries=False,
    ) -> "GeneratorModel":

        model = GeneratorModel(
            enc=enc,
            batch_size=batch_size,
            sampling_config=sampling_config,
            sample_done_criterion=sample_done_criterion,
            hparams=hparams,
        )
        try:
            model._setup(reset=True)

            model.restore_checkpoint(path, retries=retries)
        except (Exception, KeyboardInterrupt) as e:
            model.cleanup()
            raise e
        return model
