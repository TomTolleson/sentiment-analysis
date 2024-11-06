---
library_name: transformers
license: apache-2.0
base_model: distilbert-base-uncased
tags:
- generated_from_trainer
datasets:
- financial_phrasebank
model-index:
- name: sentiment-analysis
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# sentiment-analysis

This model is a fine-tuned version of [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) on the financial_phrasebank dataset.

## Model description

Model is hosted on huggingface at tangohoteltango/sentimaent-analysis
https://huggingface.co/tangohoteltango/sentiment-analysis

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 3

### Training results



### Framework versions

- Transformers 4.45.2
- Pytorch 2.4.1
- Datasets 3.0.1
- Tokenizers 0.20.1
