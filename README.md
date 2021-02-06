## nostalgebraist-autoresponder

Code for the tumblr bot [nostalgebraist-autoresponder](https://nostalgebraist-autoresponder.tumblr.com/).

For some context, see:

- the bot's [About/FAQ page](https://nostalgebraist-autoresponder.tumblr.com/about)
- [my tumblr tag for posts about the bot](https://nostalgebraist.tumblr.com/tagged/nostalgebraist-autoresponder-meta)
- [this technical overview post](https://nostalgebraist.tumblr.com/post/617940524224151552/i-imagine-some-people-have-been-curious-to-hear) (somewhat out of date)

#### Update, 2/6/21

This README used to include a note explaining that I run the bot in practice from a separate, private rpo.

This is no longer true: since [this merge](https://github.com/nostalgebraist/nostalgebraist-autoresponder/pull/1), I have been running the bot from the `main` branch of this repo.

As of this writing, I have not yet brought the latest *training* code from my private repo into this one, only the latest *serving* code.  I will update this notice accordingly when I update the training code here.  ~~Strikethrough~~ is used below to indicate cases where code is not up to date.

### Disclaimers

*This is not good code!* It is a personal project for my own entertainment.  Code style varies greatly, some code is still in horrible Jupyter notebooks (or still bears the scars of its origins in horrible Jupyter notebooks), the various components are coupled together in subtle ways, etc.

*This isn't a platform for building tumblr bots or GPT-2 bots.*  This repo mostly exists for people familiar with my bot who are interested in how it works.  In principle, you could use this to run a similar bot of your own, but I don't expect that to be easy and haven't taken steps to specifically support that use case.

### Documentation

This README is the main documentation at this time.

Some of the machine learning functionality provided by this repo is described in a separate file [`gpt-2/README.md`](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/main/gpt-2/README.md).

I plan to provide additional documentation in the future.

### How this code implements the bot

In what follows, struck-through text indicates that a file is not up to date and needs to be replaced with an up-to-date version from my private repo.

All files I need to *continuously run* the bot are currently up to date.  Files for model training, which is done "offline" separately from the bot's continuous operation, are generally not up to date.

Code in struck-through files is not compatible with the rest of the repo, and trying to read it will confuse you.

#### Running the bot (continuous)

When running and communicating with tumblr, the bot consists of the following processes, running simultaneously:

1. tumblr API layer
    - script `tumbl.py`
2.  machine learning layer
    - ~~something like the notebook `generator.ipynb`~~
    - ...if supplied with a GPU, an appropriately fine-tuned GPT-2, a dataset to get textpost prompts from, etc.
3. switchboard layer
    - script `bridge_service.py`
4. selection layer
    - script `selector.py`
    - NOTE: the selector itself runs in the machine learning layer.  `selector.py` runs as a separate process only for historical reasons, and I intend to move its functionality into the switchboard layer in the future.

1, 2 and 4 communicate by sending requests to 3.

Layer 1 requires two sets of tumblr API keys:
  - One for the "base client" which will control the bot, make posts, etc
  - One for the "dashboard client" which will follow people "as" the bot.
    - This is done as a distinct tumblr user under the assumption that the bot is a sideblog of the operator's main account; we need a way to maintain a follow list and dashboard distinct from those attached to the user's original tumblr account.

Various config options can/should be set in `config.json` (not provided), see `bot_config.py` for what the contents should look like.

#### Updating the selector (manual step every week or so)

The selector is a model trained on data from user interaction with the bot over time.

The selector is an attention-plus-MLP neural net whose inputs are the activations in some of the generator layers, when the generator views the text it wrote.

The iterative learning of the selector is "implemented" as a human (me) scraping data every so often and running a training script on the data.

- The content of the posts should be scraped from tumblr using the same pipeline used to scrape train data for the generator (see below).
  - This means scraping tumblr using an appropriate utility, and ~~running a section in `prep_generator_training_dataset.ipynb`.~~
- Code to scrape notes, and associate them with the post content, ~~is in `reward_data.py`.  I do this "manually" in a python session by importing the function `scrape_new_and_save` and calling it.~~

Training the model from the data ~~happens in `train_generator_to_select.ipynb`~~.

#### Training the generator (one-time)

The model for layer 2, the generator, should be fine-tuned on an appropriately scraped and pre-processed tumblr corpus.  This is a step which only needs to happen once, but is required for the bot to run at all.

- To scrape HTML from tumblr, use [this tool](https://github.com/bbolli/tumblr-utils) or a similar one
  - TODO: bring my entire pipeline (including this utility) into this repo
- ~~The notebook `prep_generator_training_dataset.ipynb`~~ does the pre-processing, given scraped HTML from tumblr.
- ~~To train on the pre-processed dataset, use `train_generator.ipynb`~~

#### Training the sentiment model (one-time)

The sentiment model is the same kind of model as the selector.  Its training data does not grow with time, so it only needs to be trained once per generator model.

Training code for the sentiment model will appear in this repo when I move up-to-date selector training code here, as the two training jobs are very similar and use most of the same code.

### Repo structure

- Scripts that should run simultaneously while the bot is in operation (details above)
  - `tumbl.py`
  - `bridge_service.py`
  - `selector.py`
  - ~~`generator.ipynb`~~
- Core helper code used by the scripts
  - `bot_config.py` (loader for string constants like API keys, "bad words" to screen for, etc)
  - `bridge_shared.py` (helpers for clients of the switchboard layer)
  - Code managing our communication with the tumblr API:  
    - `pytumblr_wrapper.py` (tumblr API helper)
    - `response_cache.py` (originally tumblr API helper, has now scope creeped into being a general-purpose cache)
  - Text munging code to convert between tumblr API data and formatted text:
     - `reblogs_v5.py` (parses tumblr's pre-NPF html format to a structured text format)
    - `munging_shared.py` (utility code wrapping `reblogs_v5.py` and providing other functionality needed at the interface with the tumblr API)
    - `autoresponder_static.py` and `autoresponder_static_v8.py` (conversion between the old structured text format produced by `reblogs_v5.py` and newer structured formats used by the generator)
  - ML code that operates on formatted text:
    - `autoresponder.py` (machine learning layer; ML models run in a notebook which is a lightweight wrapper around this file)
    - `autoresponder_config.py` (config file for machine learning layer)
    - `side_judgments.py` (abstraction layer around the selector and sentiment layers, used to construct calls to these ML models and cache responses)
- Helper code for specific, less central features
  - `reply_munging.py` (responding to replies in an Xkit-like manner)
  - `sentiment.py` (wrapper around a sentiment analysis API, for the "mood" feature)
  - `mood.py` (basics of the "mood" feature)
  - `mood_dynamic.py` (evolves mood over time as an ODE system, computes the forcing term)
  - `image_analysis.py` (wrappers for a image recognition API)
  - `text_segmentation.py` (generates images from text + unused research code)
- Model training scripts/code
  - ~~`reward_data.py`~~ (scrape note counts for selector model)
  - ~~`train_generator_to_select.ipynb`~~ (train selector model, "new approach")
  - ~~`train_selector.ipynb`~~ (train selector model, "old approach")
    - TODO: mark this clearly as old/deprecated
  - ~~`prep_generator_training_dataset.ipynb`~~
  - `gpt-2/*` (implements GPT-2 for training and running the generator)
  - `gpt-2/train.py` (python train script for the generator)
  - ~~`train_generator.ipynb` (train generator on google colab, lightweight notebook wrapper around `gpt-2/train.py`)~~
    - I no longer train the generator on Google Colab.  I now use a TPUv3 on Google Compute Engine, and run `gpt-2/train.py` on that machine.

Note that all the Jupyter notebooks assume you are running them in Google Colab, with the code under `/nostalgebraist-autoresponder/`.  They expect to find serialized models and other data files in one of two places:

1. A mounted Google Drive at `/content/drive/MyDrive/` with e.g. models in `/content/drive/MyDrive/models/`
2. A Google Cloud Storage bucket specified as `BUCKET_NAME` in `config.json`

If (1) fails, as it does frequently due to Google Drive limits and flakiness, the code falls back to (2).  We try (1) first under the assumption that GCS transfer is slower than Google Drive transfer. 

These notebooks can be run in other types of environments with some straightforward modifications.
