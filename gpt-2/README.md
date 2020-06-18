The code in this directory is a lightly modified version of nsheppard's finetuning fork of OpenAI's gpt-2 repo.

I didn't keep a real git history while I was hacking away at it, so it's just here as a bunch of files.  It branched off from nsheppard's fork sometime in 2019 (?), so the two have probably diverged quite a bit.  The changes I have made include:

- option to use "middle-p" sampling
- printing progress during generation
- stop generation when all members of batch contain "<|" (i.e. the first two tokens of "<|endoftext|>")
- some changes to chunk formation during dataset loading (these probably don't matter, but what was there didn't make sense to me...)
