---
layout: blog
author: cohlem
---

### Gradient Accumulation

When we want to train a neural network with some predefined set of tokens, but don't have enough GPU resources, what do we do?

- Gradient Accumulation

We simply accumulate the gradients. For instance, in order to reproduce GPT-2 124B, we need to train the model with 0.5 Million tokens in a single run with 1024 context length, we would need 0.5e6/ 1024 = 488 batches i.e B,T = (488,1024) to calculate the gradients and update them.

We don't have the enough resources, to fit those batches in our GPU, now what we do is divide that 488 into multiple batches and then do the forward pass and calculate gradients and accumulate (+) the gradients but we don't update the gradients until we reach the end of the the desired batch size. After that, we update the parameters once.
