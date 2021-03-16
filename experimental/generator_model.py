from textwrap import wrap

import tensorflow as tf
import tflex

import model
import sample

from autoresponder_config import *  # TODO: turn these into class constructor args/kwargs


def is_repeating_criterion(unique_token_frac):
    return unique_token_frac < 0.2


class GeneratorModel:
    def __init__(
        self,
        enc,
        batch_size,
        sample_done_criterion,
        hparams=model.hparams_1558M(),
        session=None,
    ):
        self.enc = enc
        self.batch_size = batch_size
        self.sample_done_criterion = sample_done_criterion
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

    def _setup(self, reset=True):
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
                # better_length=better_length,
                eot_workaround=EOT_WORKAROUND,
                enc=self.enc,
                hparams=self.hparams,
                start_token=None,
                context=self.context,
                batch_size=self.batch_size,
                return_presents=True,
                pasts=self.sample_pasts,
                mirostat_surprise_target=self.mirostat_target,
                mirostat_lr=self.mirostat_lr,
                mirostat_trunc=MIRO_TRUNC,
                mirostat_v2=MIRO_V2,
                disable_prints=True,
                breakruns=BREAKRUNS,
                breakruns_tau=BREAKRUNS_TAU,
                breakruns_decay=BREAKRUNS_DECAY,
            )

            # TODO: DRY
            self.first_sample_op = sample.sample_sequence(
                length=pre_continue_length,
                better_length=False,
                temperature=pre_continue_temperature,
                top_k=pre_continue_top_k,
                top_p=pre_continue_top_p,
                middle_p=pre_continue_middle_p,
                chop_lowest=pre_continue_chop_lowest,
                chop_highest=pre_continue_chop_highest,
                mirostat=pre_continue_mirostat,
                **sampling_args,
            )
            # TODO: DRY
            self.sample_op_fill_window = sample.sample_sequence(
                length=max_ctx_fits_on_gpu,
                better_length=True,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                middle_p=middle_p,
                chop_lowest=chop_lowest,
                chop_highest=chop_highest,
                mirostat=MIRO,
                **sampling_args,
            )
            # TODO: DRY
            self.sample_op_beyond_window = sample.sample_sequence(
                length=length,
                better_length=False,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                middle_p=middle_p,
                chop_lowest=chop_lowest,
                chop_highest=chop_highest,
                mirostat=MIRO,
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

    def write(self, prompt: str, mirotarg: float = None, verbose=False):
        if mirotarg is None:
            mirotarg = np.random.choice(MIRO_TARGET_ALL)

        context_tokens = self.enc.encode(prompt)

        startup_presents = self.startup_presents_for_prompt.get(prompt, None)

        max_context_size = max_ctx_fits_on_gpu - length
        # if better_length:
        #     max_context_size = length - required_continuation_room
        # else:
        #     max_context_size = max_ctx_fits_on_gpu - length
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

        first_step_with_miro = 1 if MIRO_ONLY_ON_CONTINUE else 0

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
        mu_init_scale = 1.0 if MIRO_V2 else 2.0

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
                                self.mirostat_lr: MIRO_LR,
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
                                self.mirostat_lr: MIRO_LR,
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
                                self.mirostat_lr: MIRO_LR,
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
                                self.mirostat_lr: MIRO_LR,
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
                mirosurprises = sample_output_dict["mirostat_surprises"]
                miroks = sample_output_dict["mirostat_ks"]
                miromus = sample_output_dict["mirostat_mus"]
                mirotoks = sample_output_dict["tokens"]
            else:
                mirosurprises = np.concatenate(
                    [mirosurprises, sample_output_dict["mirostat_surprises"]], axis=1
                )
                miroks = np.concatenate(
                    [miroks, sample_output_dict["mirostat_ks"]], axis=1
                )
                miromus = np.concatenate(
                    [miromus, sample_output_dict["mirostat_mus"]], axis=1
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

                is_repeating_this_time = is_repeating_criterion(unique_token_frac)

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
            more_permitted = (this_batch_continue_steps < MAX_CONTINUE_STEPS) and (
                tokens_generated < MAX_CONTINUE_TOKENS
            )

            show_miro_logs = MIRO and (
                (not MIRO_ONLY_ON_CONTINUE)
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
                    f"\t{this_batch_continue_steps}/{MAX_CONTINUE_STEPS} continue steps used"
                )
                print(
                    f"\t{tokens_generated}/{MAX_CONTINUE_TOKENS} continue tokens generated"
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
        self.startup_presents_for_prompt = {}

    @staticmethod
    def load(
        path,
        enc,
        batch_size,
        sample_done_criterion,
        hparams=model.hparams_1558M(),
        retries=False,
    ) -> "GeneratorModel":

        model = GeneratorModel(
            enc=enc,
            batch_size=batch_size,
            sample_done_criterion=sample_done_criterion,
            hparams=hparams,
        )
        try:
            model._setup()

            model.restore_checkpoint(path, retries=retries)
        except (Exception, KeyboardInterrupt) as e:
            model.cleanup()
            raise e
        return model
