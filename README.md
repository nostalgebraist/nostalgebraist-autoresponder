## nostalgebraist-autoresponder

Code for the tumblr bot [nostalgebraist-autoresponder](https://nostalgebraist-autoresponder.tumblr.com/).

For some context, see [here](https://nostalgebraist.tumblr.com/tagged/nostalgebraist-autoresponder-meta) and particularly [here](https://nostalgebraist.tumblr.com/post/617940524224151552/i-imagine-some-people-have-been-curious-to-hear).

### Disclaimers

*This is not good code!* It is a personal project for my own entertainment.  Code style varies greatly, some code is still in horrible Jupyter notebooks (or still bears the scars of its origins in horrible Jupyter notebooks), the various components are coupled together in subtle ways, etc.

*This is not the version of the code being actively developed, or the version that is operational.*  It is a slightly cleaned-up snapshot of the project, with a fresh git history, as of 6/18/20.  All the components of the real bot are here except the serialized models and other data files, but I haven't verified that it all works the way the "live" version does.  The cleanup process may have introduced a bug here or there.

*This isn't a platform for building tumblr bots or GPT-2 bots.*.  This repo mostly exists for people familiar with my bot who are interested in how it works.  In principle, you could use this to run a similar bot of your own, but I don't expect that to be easy and haven't taken steps to specifically support that use case.

### How this code implements the bot

#### Running the bot (continuous)

When running and communicating with tumblr, the bot consists of the following processes, running simultaneously:

1. tumblr API layer
    - script `tumbl.py`
2. generator layer
    - something like the notebook `generator.ipynb`
    - ...if supplied with a GPU, an appropriately fine-tuned GPT-2, a dataset to get textpost prompts from, etc.
3. selector layer
    - script `selector.py`
4. switchboard layer
    - script `bridge_service.py`

1-3 communicate by sending requests to 4.

Layer 1 requires two sets of tumblr API keys:
  - One for the "base client" which will control the bot, make posts, etc
  - One for the "dashboard client" which will follow people "as" the bot.
    - This is done as a distinct tumblr user under the assumption that the bot is a sideblog of the operator's main account; we need a way to maintain a follow list and dashboard distinct from those attached to the user's original tumblr account.

Various config options can/should be set in `config.json` (not provided), see `bot_config.py` for what the contents should look like.

#### Updating the selector (manual step every week or so)

Layer 3, the selector, is a BERT model trained on data from user interaction with the bot over time.  This is "implemented" as a human (me) scraping data every so often and running a training script on the data.

- Code to scrape this data is in `reward_data.py`.  I do this "manually" in a python session by importing the function `scrape_new_and_save` and calling it.
- Training the model from the data happens in `train_selector.ipynb`

#### Training the generator (one-time)

The model for layer 2, the generator, should be fine-tuned on an appropriately scraped and pre-processed tumblr corpus.  This is a step which only needs to happen once, but is required for the bot to run at all.

- To scrape HTML from tumblr, use [this tool](gist.github.com/doersino/7e3e5db591e42bf543e1) or a similar one
- The notebook `prep_generator_training_dataset.ipynb` does the pre-processing, given scraped HTML from tumblr.

