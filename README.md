## nostalgebraist-autoresponder

Code for the tumblr bot [nostalgebraist-autoresponder](https://nostalgebraist-autoresponder.tumblr.com/).

For some context, see:

- the bot's [About/FAQ page](https://nostalgebraist-autoresponder.tumblr.com/about)
- [my tumblr tag for posts about the bot](https://nostalgebraist.tumblr.com/tagged/nostalgebraist-autoresponder-meta)
- [this technical overview post](https://nostalgebraist.tumblr.com/post/617940524224151552/i-imagine-some-people-have-been-curious-to-hear) (somewhat out of date)

#### Update, 11/19/20

It's been a while since I last synced up this repo with the "real" one I use to operate and develop the bot.  Nothing has changed dramatically, but I've made various changes that aren't reflected here, e.g.:

- Sampling with [Mirostat](https://arxiv.org/abs/2007.14966) algorithm
- Using "forum-like" natural language delimeters rather than arbitrary non-English characters
- Better tumblr API caching and ratelimit management
- Code and modeling improvements to the selector model
- Making the sentiment model an additional GPT-2 head (like the selector) rather than relying on an external service
- Shifting more code out of Jupyter into .py files, relying more on imports when notebooks are unavoidable (for Colab)

I'm working on gradually moving these changes into this repo.  My work branch for that effort is [11-28-29-syncing-with-dev-repo](https://github.com/nostalgebraist/nostalgebraist-autoresponder/tree/11-28-29-syncing-with-dev-repo).
 
When I next have time to get this repo _fully_ up to date, I'll also switch over to it for operation and development, making further divergence impossible.

### Disclaimers

*This is not good code!* It is a personal project for my own entertainment.  Code style varies greatly, some code is still in horrible Jupyter notebooks (or still bears the scars of its origins in horrible Jupyter notebooks), the various components are coupled together in subtle ways, etc.

*This is not the version of the code being actively developed, or the version that is operational.*  It is a slightly cleaned-up snapshot of the project, with a fresh git history, synced up to the operational code only occasionally.  All the components of the real bot are here except the serialized models and other data files, but I haven't verified that it all works the way the "live" version does.  The cleanup process may have introduced a bug here or there.

*This isn't a platform for building tumblr bots or GPT-2 bots.*  This repo mostly exists for people familiar with my bot who are interested in how it works.  In principle, you could use this to run a similar bot of your own, but I don't expect that to be easy and haven't taken steps to specifically support that use case.

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
      - NOTE: in the latest version of this project, the selector also runs in `generator.ipynb`, and `selector.py` is only responsible for some "plumbing" steps unrelated to the selector ML model
4. switchboard layer
    - script `bridge_service.py`

1-3 communicate by sending requests to 4.

Layer 1 requires two sets of tumblr API keys:
  - One for the "base client" which will control the bot, make posts, etc
  - One for the "dashboard client" which will follow people "as" the bot.
    - This is done as a distinct tumblr user under the assumption that the bot is a sideblog of the operator's main account; we need a way to maintain a follow list and dashboard distinct from those attached to the user's original tumblr account.

Various config options can/should be set in `config.json` (not provided), see `bot_config.py` for what the contents should look like.

#### Updating the selector (manual step every week or so)

Layer 3, the selector, is a model trained on data from user interaction with the bot over time.

The iterative learning of the selector is "implemented" as a human (me) scraping data every so often and running a training script on the data.

- The content of the posts should be scraped from tumblr using the same pipeline used to scrape train data for the generator (see below).
  - This means scraping tumblr using an appropriate utility, and running a section in `prep_generator_training_dataset.ipynb`.
- Code to scrape notes, and associate them with the post content, is in `reward_data.py`.  I do this "manually" in a python session by importing the function `scrape_new_and_save` and calling it.

Training the selector depends on which version of the model you are training.

##### New selector approach

This describes the approach used after sometime in June 2020.

The selector is an attention-plus-MLP neural net whose inputs are the activations in some of the generator layers, when the generator views the text it wrote.

Training the model from the data happens in `train_generator_to_select.ipynb`.

##### Old selector approach

This describes the approach used before sometime in June 2020.

The selector is a BERT model whose inputs are text.

Training the model from the data happens in `train_selector.ipynb`.

#### Training the generator (one-time)

The model for layer 2, the generator, should be fine-tuned on an appropriately scraped and pre-processed tumblr corpus.  This is a step which only needs to happen once, but is required for the bot to run at all.

- To scrape HTML from tumblr, use [this tool](https://github.com/bbolli/tumblr-utils) or a similar one
- The notebook `prep_generator_training_dataset.ipynb` does the pre-processing, given scraped HTML from tumblr.
- To train on the pre-processed dataset, use `train_generator.ipynb`

### Repo structure

- Scripts that should run simultaneously while the bot is in operation (details above)
  - `tumbl.py`
  - `bridge_service.py`
  - `selector.py`
  - `generator.ipynb`
- Core helper code used by the scripts
  - `bot_config.py` (loader for string constants like API keys, "bad words" to screen for, etc)
  - `ratelimit_util.py` (tumblr API helper)
  - `response_cache.py` (originally tumblr API helper, has now scope creeped into being a general-purpose cache)
  - `reblogs_v5.py` (text munger to convert structured tumblr data into text for GPT-2)
- Helper code for specific, less central features
  - `reply_munging.py` (responding to replies in an Xkit-like manner)
  - `sentiment.py` (wrapper around a sentiment analysis API, for the "mood" feature)
  - `mood.py` (basics of the "mood" feature)
  - `mood_dynamic.py` (evolves mood over time as an ODE system, computes the forcing term)
  - `image_analysis.py` (wrappers for a image recognition API)
- Model training scripts/code
  - `reward_data.py` (scrape note counts for selector model)
  - `train_generator_to_select.ipynb` (train selector model, "new approach")
  - `train_selector.ipynb` (train selector model, "old approach")
  - `prep_generator_training_dataset.ipynb`
  - `gpt-2/*` (implements GPT-2 for training and running the generator)
  - `gpt-2/train.py` (python train script for the generator)
  - `train_generator.ipynb` (train generator on google colab, lightweight notebook wrapper around `gpt-2/train.py`)

Note that all the Jupyter notebooks assume you are running them in Google Colab, with the code under `gpt-2/` in a Google Drive associated with the same account, and (when applicable) models or data in sub-directories `gpt-2/data/`, `gpt-2/models/`, or `gpt-2/reward/`.  If you have used OpenAI's gpt-2 repo on Colab before, this setup should be familiar.  These notebooks can be run in other types of environments with some straightforward modifications.
