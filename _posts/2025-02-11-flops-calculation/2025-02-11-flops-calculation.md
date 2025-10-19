---
layout: blog
author: cohlem
---

#### Calculation of FLOPs

- multiply accumulate cost: 2FLOPS i.e 1 for multiplication and 1 for accumulation (addition)
- if we multiply two matrices with sizes (a x b) and (b x c), the flops involved is b Multiply-add operation per the output size (a x c) i.e 2 x b x (a x c)

##### Embedding lookup

we initially have tokens with (seq_len,vocab_size) one-hot representation and embedding lookup matrix is (vocab_size, d_model), it will take

FLOPs = 2 x ( vocab_size x (seq_len x d_model))

##### Attention

**Q,K,V projections**
X @ (Wq or Wk or Wv)
i.e 2 x (seq_len x d_model x key_size x num_heads)

**attention matrix**
Q @ K.T
i.e 2\* (seq_len x seq_len x key_size x num_heads)

**softmax**

- 1 for exponential calculation (e^x).
- seq_len - 1 sum for each row. so if we divide it per row, its basically 1 FLOPs per elements.
- 1 for division
  so it becomes, 2 x num_heads x seq_len x seq_len

**Softmax @ query reductions**
2 × seq_len × seq_len × (key_size × num_heads)

**Final Linear**
2 × seq_len × (key_size × num_heads) × d_model

**Dense Block** (per layer)
2×seq_len×(d_model×ffw_size+d_model×ffw_size) (ignoring FLOPs for actions here,)

**Final Logits**
2×seq_len×d_model×vocab_size

so total FLOPs: embeddings+num_layers×(total_attention+dense_block) + logits

For backward, it takes 2 times the flops taken in backward.

```python
def calculate_transformer_flops(
    seq_len: int,
    vocab_size: int,
    d_model: int,
    key_size: int,
    num_heads: int,
    ffw_size: int,
    num_layers: int,
) -> dict:
    """
    Calculate FLOPs for each component of a transformer model including forward and backward passes.

    Args:
        seq_len: Sequence length
        vocab_size: Vocabulary size
        d_model: Model dimension
        key_size: Key dimension
        num_heads: Number of attention heads
        ffw_size: Feed-forward layer size
        num_layers: Number of transformer layers

    Returns:
        Dictionary containing FLOPs for each component and total forward/backward passes
    """

    # Embeddings
    embedding_flops = 2 * seq_len * vocab_size * d_model

    # Single Attention Layer
    key_query_value_proj = 2 * 3 * seq_len * d_model * (key_size * num_heads)
    key_query_logits = 2 * seq_len * seq_len * (key_size * num_heads)
    softmax_ops = 3 * num_heads * seq_len * seq_len
    softmax_query_reduction = 2 * seq_len * seq_len * (key_size * num_heads)
    final_linear = 2 * seq_len * (key_size * num_heads) * d_model

    total_attention_flops = (
        key_query_value_proj
        + key_query_logits
        + softmax_ops
        + softmax_query_reduction
        + final_linear
    )

    # Single Dense Block
    dense_block_flops = 2 * seq_len * (d_model * ffw_size + d_model * ffw_size)

    # Final Logits
    final_logits_flops = 2 * seq_len * d_model * vocab_size

    # Total forward pass
    total_forward_pass = (
        embedding_flops
        + num_layers * (total_attention_flops + dense_block_flops)
        + final_logits_flops
    )

    # Backward pass is approximately 2x forward pass
    total_backward_pass = 2 * total_forward_pass

    # Total forward + backward
    total_flops = total_forward_pass + total_backward_pass

    return total_flops


# Example usage
params = {
    "seq_len": 512,
    "vocab_size": 50000,
    "d_model": 640,
    "key_size": 64,
    "num_heads": 10,
    "ffw_size": 2560,
    "num_layers": 10,
}


flops = calculate_transformer_flops(**params)


print(flops)
```

So this is flops required for our model per step with one batch.
