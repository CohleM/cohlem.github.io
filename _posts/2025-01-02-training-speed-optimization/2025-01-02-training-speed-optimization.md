---
layout: blog
author: cohlem
---

### Precision

The more the precision point the less operation (TFLOPS) is performed.

- FP64 used for scientific research purposes, where precision is a must.
- TF32 and BFLOAT16 are mostly used in NN training.
- INT8 is used for inference.

Picture below shows specifications of A100 GPU.

![GPU precision](/assets/images/2025-01-02-training-speed-optimization/fig1.png)

Using these precision points may have some difference in code.
See pytorch's docs

### torch.compile

It works in a similar fashion like the GCC compiler. It works by reducing overheads introduced by the python interpreter and optimizing the GPU read and writes.

For instance

![gpu memory](/assets/images/2025-01-02-training-speed-optimization/fig2.png)

```python
def gelu(x):
    """
    Applies the GELU activation function to the input.
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
```

First this operation resides in GPU's HBM memory, and this part of calculation "torch.pow(x, 3)" is passed to GPU and it performs the operations, one by one, the instructions are sent from HBM to GPU cores and transferred back to HBM one by one. But torch.compiles evaluates that the code is simply operation on input x and some +,\* and transfers the code to GPU once and does all the operation and send it back to HBM, in this way it optimizes the training process.

### Flash attention

It is somewhat similar to torch.compile's process but torch.compile itself cannot comprehend our code(shown below) to perform the optimization.

```python
aw = (Q @ torch.transpose(K, -2,-1)) # for matmul dim of q should be B,T,C and k should be B,C,T
aw = aw/(K.shape[-1] **0.5)
mask = self.tril[:,:,:T,:T] == 0 # generate mask
aw = aw.masked_fill_(mask, float('-inf')) # apply mask i.e fill true values with -inf
aw = torch.softmax(aw,dim=-1) # -inf values are converted to 0 and then each row is normalized
cv = aw @ V # context vector


```

We have to call **`torch.nn.functional.scaled_dot_product_attention`** combined with torch.compile to use FlashAttention in our code.

### Remove ugly numbers.

Always include numbers in our code that have powers of 2 in it.

for instance 16,32,64 work best.

**Improvements**

for instance, while training GPT-2 our vocab_size is 50257

if we factorize it has divisors

```
1 | 29 | 1733 50257
```

None of it have powers of 2, so the GPU performs operation on that matrix by truncating til the last powers of 2 and then doing the operation on the remaining parts, which is inefficient. We can simply increase that number to be a closed number that has powers of 2 such as 50304 = 2^7 × 3 x 131 which has high number of power of 2.

We can simply increase the training speed by making our numbers in our code have more powers of 2.
