## nostalgebraist-autoresponder

Code for the tumblr bot [nostalgebraist-autoresponder](https://nostalgebraist-autoresponder.tumblr.com/).

This is not good code!  It is a personal project for my own entertainment.

This repo mostly exists for people familiar with the bot who are interested in how it works.  In principle, you could use this to run a similar bot of your own, but I don't expect that to be easy and haven't taken steps to specifically support that use case.

### Running the bot (continuous)

The bot consists of the following processes, running simultaneously:

1. tumblr API layer
  - script `tumbl.py`
2. generator layer
  - something like the notebook `generator.ipynb`, if supplied with a GPU, an appropriately fine-tuned GPT-2, a dataset to get textpost prompts from, etc.
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

### Updating the selector (manual step every week or so)

Layer 3 is a BERT model trained on data from user interaction with the bot over time.
  - Code to scrape this data is in `reward_data.py`
  - Training the model from the data happens in `train_selector.ipynb`

### Training the generator (one-time)

The model for 2 should be fine-tuned on an appropriately scraped and pre-processed tumblr corpus.
   - The notebook `prep_generator_training_dataset.ipynb` does the pre-processing, given scraped HTML from tumblr.
   - To scrape HTML from tumblr, use [this tool](gist.github.com/doersino/7e3e5db591e42bf543e1) or a similar one
