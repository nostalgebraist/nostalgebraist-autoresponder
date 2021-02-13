## Visualizations

Directory for sharing visualizations.

Exists only on a non-default branch `visualizations`, because transfering images can be slow and I want cloning from `main` to be fast.

### Subdirectories

#### `selector_attention/`

These images show where the attention block in the selector model is looking.  Many different versions of the selector model are visualized here.

##### Reading the plots

The selector model has this structure:

- It is a transfer learning model uses activations from the (frozen) generator model as inputs to a task-specific model head.
  - That is, it is a "feature extraction" model in the sense of [Peters et al, "To Tune or Not to Tune?"](https://arxiv.org/pdf/1903.05987.pdf).
  - Transfer learning with ELMo is probably the most widely familiar instance of this type of model in NLP.  Here, the generator (finetuned GPT-2) plays the role of the pretrained BiLSTM in ELMo.
- The features extracted are the hidden states of the generator at layers 8 and 24 (of 48 layers total).
- TODO: finish

##### File names

File names follow the pattern

`example_{NAME}_ar_{GENERATOR_VERSION}_selector_{SELECTOR_VERSION}.png`

where

- `NAME` is a name identifying the post being visualized.
- `GENERATOR_VERSION` is the version number of the generator model, which the selector "sits on top of."
  - As of this writing, `v10` is the latest version.
- `SELECTOR_VERSION` is the version number of the selector model.
  - These numbers reset when `GENERATOR_VERSION` increments.   For example:
    - `ar_v9_1_selector_v1` is the first selector trained on top of generator `v9_1`.  
    - `ar_v10_selector_v1` is the first selector trained on top of generator `v10`.  

### TODO

- Add train/val loss and noise scale plots from generator training
- Add selector evaluation plots
- Add long-term / counterfactual mood graphs
