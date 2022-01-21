## nostalgebraist-autoresponder

Code for the tumblr bot [nostalgebraist-autoresponder](https://nostalgebraist-autoresponder.tumblr.com/).

For some context, see:

- the bot's [About/FAQ page](https://nostalgebraist-autoresponder.tumblr.com/about)
- [my tumblr tag for posts about the bot](https://nostalgebraist.tumblr.com/tagged/nostalgebraist%20autoresponder%20meta)

----

This repo includes most of the code necessary to run and operate the bot.

### Documentation

There is a [technical overview](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/main/docs/overview.md) under `docs/overview.md`.  It is up to date as of January 2022.

There are [some visualizations](https://github.com/nostalgebraist/nostalgebraist-autoresponder/tree/visualizations/visualizations) on a branch called `visualizations`, in a subdirectory called (you guessed it) `visualizations`.  Check these out if you want to learn more about how the bot's selector model works.  Note that the accompanying text is somewhat out of date.

### Disclaimers

#### *The quality of the code varies widely.*

This is a hacky personal project for my own entertainment.  Its primary goals are

1. stay up and running as close to 24/7 as possible
    - ...in a context where a development/sandbox environment is essentially impossible
    - ...while handling a wide variety of unpredictable user behavior
    - ...while depending on external APIs that often lack complete contracts or guarantees
2. do a lot of cool stuff at once
3. be legible to my future self

These goals, especially #1, take precedence over code cleanliness and consistent design.  Refactors and other large-scale changes are inherently risky when all testing must happen in production, and I've often left weird code stay weird for this reason.

This project involves many layers of functionality that have accreted over time.  Code style varies greatly, and adjacent pieces of code may reflect very different phases in my personal view of the project.

#### *This isn't a platform for building tumblr bots or GPT bots.*

This code is written to do exactly one thing: run one specific tumblr bot, in a one specific compute environment, assuming a user who knows exactly what I do.  I make it public mostly to satisfy the curiousity of people who are familiar with my bot and want to see its internals.

In principle, you could adapt this codebase to run a similar bot of your own, but I haven't taken steps to specifically support that use case.  In fact, [I don't recommend running this kind of bot *at all*](https://nostalgebraist-autoresponder.tumblr.com/about#dont-make-a-bot).

If you're interested in building your own work on top of mine, I've spun off some of the work from this project into separate libraries:

- If you're making a tumblr app, check out **[pytumblr2](https://github.com/nostalgebraist/pytumblr2)**, a fork of the official tumblr API client with much fuller support for the modern tumblr API
- If you're working with transformer language models, check out **[transformer-utils](https://github.com/nostalgebraist/transformer-utils)**, a collection of utilities for the Huggingface [transformers](https://github.com/huggingface/transformers) package
