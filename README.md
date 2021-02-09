## nostalgebraist-autoresponder

Code for the tumblr bot [nostalgebraist-autoresponder](https://nostalgebraist-autoresponder.tumblr.com/).

For some context, see:

- the bot's [About/FAQ page](https://nostalgebraist-autoresponder.tumblr.com/about)
- [my tumblr tag for posts about the bot](https://nostalgebraist.tumblr.com/tagged/nostalgebraist-autoresponder-meta)
- [this technical overview post](https://nostalgebraist.tumblr.com/post/617940524224151552/i-imagine-some-people-have-been-curious-to-hear) (somewhat out of date)

### Status of the repo (last updated 2/9/21)

This README used to include a note explaining that I run the bot in practice from a separate, private repo.

This is no longer true: since [this merge](https://github.com/nostalgebraist/nostalgebraist-autoresponder/pull/1), I have been running the bot from the `main` branch of this repo.

This repo includes code to run the bot and code to train the generator, selector, sentiment models.  It is currently missing the following pieces:

- Tumblr scraping scripts used in generator and selector training
- Data prep script used in generator training

### Disclaimers

*This is not good code!* It is a personal project for my own entertainment.  Code style varies greatly, some code is still in horrible Jupyter notebooks (or still bears the scars of its origins in horrible Jupyter notebooks), the various components are coupled together in subtle ways, etc.

*This isn't a platform for building tumblr bots or GPT-2 bots.*  This repo mostly exists for people familiar with my bot who are interested in how it works.  In principle, you could use this to run a similar bot of your own, but I don't expect that to be easy and haven't taken steps to specifically support that use case.

### Documentation

This README is the main documentation at this time.

Some of the machine learning functionality provided by this repo is described in a separate file [`gpt-2/README.md`](https://github.com/nostalgebraist/nostalgebraist-autoresponder/blob/main/gpt-2/README.md).

I plan to provide additional documentation in the future.

### How this code implements the bot

#### Running the bot (continuous)

When running and communicating with tumblr, the bot consists of the following processes, running simultaneously:

1. tumblr API layer
    - script `tumbl.py`
2.  machine learning layer
    - `autoresponder_wrapper.ipynb` (a lightweight wrapper around `autoresponder.py`)
    - Running this layer requires:
      - a GPU with ~16GB memory
      - a GCS bucket containing the following data files:
        - trained generator, selector and sentiment models
        - a data corpus (TODO: remove this requirement, as it's no longer used for anything nontrivial)
      - a particular GCS directory structure in the bucket (see `autoresponder_config.py`)
      - optionally, a copy of same files in Google Drive (loads faster but less reliably)
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

To train the selector:

- Scrape the bot's tumblr
  - _How to do this_:  For historical reasons, I currently do this with a [third-party utility](https://github.com/bbolli/tumblr-utils) and this pipeline is coupled to the output data format of that utility.  (TODO: include the whole pipeline in the repo)
- Preprocess the data
  - _How to do this_: Run the script `selector_model/data_prep.py`
- Train the model
  - _How to do this_: run the notebook `train_side_judgments.ipynb` (needs GPU)

#### Training the generator (one-time)

The model for layer 2, the generator, should be fine-tuned on an appropriately scraped and pre-processed tumblr corpus.  This is a step which only needs to happen once, but is required for the bot to run at all.

To train the generator:

- Scrape one or more tumblr(s) you want the bot to imitate
  - _How to do this_: see comments about scraping in ["Updating the selector"](#updating-the-selector-manual-step-every-week-or-so) above
- Preprocess the data
  - _How to do this_: The code for this is not in this repo yet.  It's broadly similar to `selector_model/data_prep.py` but has some additional steps.
- Download a GPT-2 1.5B pretrained model to finetune
  - _How to do this_: [OpenAI's download script](https://github.com/openai/gpt-2/blob/master/download_model.py)
- Fine-tune the model
  - _How to do this_:  and run the script `gpt-2/train.py`
    - This step requires a TPUv3 (because we need ~16GB memory to store GPT-2 1.5B's params + the Adam accumulators for them).
    - This means it won't run on Colab, which only provides TPUv2.  I run it on Google Cloud Engine.  (Running it on GCE requires some one-time setup of the VM instance not yet documented here.)
 
An example invocation of `gpt-2/train.py` on an appropriately configured GCE VM:
 
```bash
python3.7 -u gpt-2/train.py --model_name 1558M --run_name YOUR_RUN_NAME --noise_scale --dataset PATH_TO_TRAIN_DATA --val_dataset PATH_TO_VAL_DATA --val_every 200 --val_batch_size 24 --batch_size 24 --accumulate_gradients 5 --eot_workaround --rob_sampler --save_every 100 --sample_every 10000000 --save_time 300000  --max_to_keep 1 --acti_dropout 0 --attn_dropout 0 --res_dropout 0 --adam_beta1 0.9 --adam_beta2 0.999 --adam_epsilon 1e-8 --learning_rate 4.76e-5 --learning_rate_cos --learning_rate_warmup 100 --learning_rate_period 5425 --learning_rate_min 1.5e-6 --learning_rate_m_mul 0.1 --avg_loss_beta 0.99 --only_train_transformer_layers --seed 5 --save_optimizer --init_tpu --learning_rate_initial_step 1 | tee -a runlog.txt
```

I recommend setting `learning_rate_period` to 5 times the number of batches per epoch, where batches per epoch is `batch_size * accumulate_gradients * (token count of train dataset)`.  You may want to tune `learning_rate_period`, `learning_rate`, `learning_rate_warmup`, etc. based on validation performance.

The pipe to `tee -a runlog.txt` produces a log `runlog.txt`, which can be used to review losses, gradient noise scale, and other stats across training.  Utilities for this are in `util/loss_plots.py`.

For my bot, I fine-tune on a [large tumblr corpus](https://nostalgebraist.tumblr.com/post/637364984645664768/helping-my-bot-understand-tumblr-the-colossal), then additionally fine-tune the result on a specific blog.

#### Training the sentiment model (one-time)

The sentiment model is the same kind of model as the selector.  Its training data does not grow with time, so it only needs to be trained once per generator model.

To train the sentiment model

- Obtain the training data.  (TO DO: make this data public, since it's based on historical bot data and is not possible to generate from scratch without having a bot running first)
- Run `train_side_judgments.ipynb` with the `TRAIN_SENTIMENT` flag set to `True`

### Repo structure

- Scripts that should run simultaneously while the bot is in operation (details above)
  - `tumbl.py`
  - `bridge_service.py`
  - `selector.py`
  - `autoresponder_wrapper.ipynb`
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
    - `autoresponder.py` (machine learning layer; ML models run in a notebook `autoresponder_wrapper.ipynb` which is a lightweight wrapper around this file)
    - `autoresponder_config.py` (config file for machine learning layer)
    - `side_judgments.py` (abstraction layer around the selector and sentiment layers, used to construct calls to these ML models and cache responses)
- Helper code for specific, less central features
  - `reply_munging.py` (responding to replies in an Xkit-like manner)
  - `sentiment.py` (wrapper around a sentiment analysis API, for the "mood" feature)
  - `mood.py` (basics of the "mood" feature)
  - `mood_dynamic.py` (evolves mood over time as an ODE system, computes the forcing term)
  - `image_analysis.py` (wrappers for a image recognition API)
  - `text_segmentation.py` (generates images from text + unused research code)
  - `traceability.py` (logs API responses and other metadata for each post the bot makes)
- Model training scripts/code
  - `gpt-2/*` (implements GPT-2 for training and running the generator)
  - `gpt-2/train.py` (python train script for the generator)
  - `selector_model/data_prep.py` (scrape note counts for selector model)
  - `train_side_judgments.ipynb` (train selector and sentiment models)

Note that all the Jupyter notebooks assume you are running them in Google Colab, with the code under `/nostalgebraist-autoresponder/`.  They expect to find serialized models and other data files in one of two places:

1. A mounted Google Drive at `/content/drive/MyDrive/` with e.g. models in `/content/drive/MyDrive/models/`
2. A Google Cloud Storage bucket specified as `BUCKET_NAME` in `config.json`

If (1) fails, as it does frequently due to Google Drive limits and flakiness, the code falls back to (2).  We try (1) first under the assumption that GCS transfer is slower than Google Drive transfer.

These notebooks can be run in other types of environments with some straightforward modifications.
