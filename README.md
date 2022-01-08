## nostalgebraist-autoresponder

Code for the tumblr bot [nostalgebraist-autoresponder](https://nostalgebraist-autoresponder.tumblr.com/).

For some context, see:

- the bot's [About/FAQ page](https://nostalgebraist-autoresponder.tumblr.com/about)
- [my tumblr tag for posts about the bot](https://nostalgebraist.tumblr.com/tagged/nostalgebraist-autoresponder-meta)

----

This repo includes most of the code necessary to run and operate the bot.

### Disclaimers

*This is not good code!* It is a personal project for my own entertainment.

Code style varies greatly, some code is still in horrible Jupyter notebooks (or still bears the scars of its origins in horrible Jupyter notebooks), the various components are coupled together in subtle ways, etc.

*This isn't a platform for building tumblr bots or GPT-2 bots.*   This code is written to do exactly one thing: run one specific tumblr bot, in a one specific compute environment, assuming a user who knows exactly what I do.  I make it public mostly to satisfy the curiousity of people who are familiar with my bot and want to see its internals.

In principle, you could adapt this codebase to run a similar bot of your own, but I haven't taken steps to specifically support that use case.  In fact, [I don't recommend running this kind of bot *at all*](https://nostalgebraist-autoresponder.tumblr.com/about#dont-make-a-bot).

### Documentation

There is a [technical overview](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/main/docs/overview.md) under `docs/overview.md`.  It is up to date as of January 2022.

There are [some visualizations](https://github.com/nostalgebraist/nostalgebraist-autoresponder/tree/visualizations/visualizations) on a branch called `visualizations`, in a subdirectory called (you guessed it) `visualizations`.  Check these out if you want to learn more about how the selector works.
