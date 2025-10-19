---
layout: blog
author: cohlem
---

## Pre-training

### Document packing

while pretraining, different documents could be packed inside a sequence. For instance, a model with context_length 1024 can have 256 tokens from one doc and rest from the other. Demilited by EOS token.

The samples may contaminate the attention, for which cross sample attention masking is used.
But, it isn't used by DeepSeek v3, lets not use it.

while packing documents. we simply pack them as they appear in order and then add EOS token (used by GPT-2,3). But DeekSeek also uses FIM (Fill in middle) strategy using this Prefix-Suffix-Middle (PSM) framework.

`<|fim_begin|> 𝑓pre <|fim_hole|> 𝑓suf <|fim_end|> 𝑓middle <|eos_token|>.`

adopted for 0.1% of data, generally used for overfitting or limiting the model from using the same general method.

- Do vibe check once in a while

commands

change num_proc in process.py

python process.py --tokenizer_path /model/tokenizer

training run
torchrun --standalone --nproc_per_node=2 pretrain.py
