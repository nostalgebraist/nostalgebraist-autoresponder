The code in this directory is essentially my own fork of nsheppard's finetuning fork of OpenAI's gpt-2 repo.  (Got that?)

I didn't keep a real git history during much of my early hacking away at it, so it's just here as a bunch of files.

It branched off from nsheppard's fork sometime in 2019 (?), and the two have diverged quite a bit.  (It has also diverged from OpenAI's gpt-2 repo, which was modified after nsheppard's fork split off.)

The changes I have made include:

- **Encoding**
  - parse the string `"<|endoftext|>"` to GPT-2's `<|endoftext|>` separator token
- **Sampling**
  - option to use "middle-p" sampling or the [Mirostat](https://arxiv.org/abs/2007.14966) sampling algorithm (the latter is currently used by the bot)
  - optionally printing progress during generation
  - stop generation when all members of batch contain the `<|endoftext|>` separator (or when they contain 2 such tokens, if we are beginning our prompt with one)
- **Modeling**
  - various changes to `model.py` to support building new "heads" like the selector and sentiment model, and to support some unused research code
- **Training**
  - Train script fully utilizes the 8 cores of the TPU (requires TPUv3 to fit GPT-2 1.5B)
  - Saves checkpoints to GCS
  - Samples training data without replacement during an epoch
  - Fully deterministic training:
    - Batches are sampled reproducibly
    - Optimizer state is checkpointed alongside model state
  - Gradient accumulation
  - Measures the [gradient noise scale](https://arxiv.org/abs/1812.06162) during training
