---
layout: blog
author: cohlem
tags: [architecture]
---

## Scaled-dot product Attention

### Q1

Given the attention equation

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(xWq)(xWk)^\top}{\sqrt{d_k}}\right)(xWv)W_O
$$

Why don't we train by combining $WqWk^\top$ and $WvWo$? because mathematically they seem equivalent

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{x(WqWk^\top)x^\top}{\sqrt{d_k}}\right)x(WvW_O)
$$

I initially thought if we could combine those weights, we don't need to calculate $Q,K,V$ meaning there will be less number of matrix multiplication.

#### Answer

We lose the objective of $Q,K,V,O$, they are meant to operate independently. In analogy,
$Q$ represents the notion of "what we have", $K$ whats available to us and so on. BUT, if we combine them into a giant matrix $WqWk^\top$ during backpropagation weight updates are mixed. We no longer update them separately so we end up loosing their significance.

### Q2

So, if we can't use them during training, can we still mix those weights during inference now that we aren't updating weights? since the mathematical equivalence is the same, can we use it to optimize inference performance?

#### Answer

We decrease the number of matrix multiplication, BUT we end up increasing the actual element wise multiplications inside those matrix multiplication.

We end up decreasing the speed rather than increasing it.

Let's see this by comparison.

**NOTE: Given matrix A with size (2,3) and B with size (3,4), the total number of element wise matrix multiplication between is (2 _ 3 _ 4).**

$n$ = number of tokens
$d_{\text{model}}$: embedding dimension
$nh$: number of heads
$hd$: number of head*dimension
$d*{\text{k}}$: $nh$ x $hd$

#### CASE I: Original Attention

Compute $Q = X W_Q$: $\mathcal{O}(n \cdot d_{\text{model}} \cdot d_k)$

Compute $K = X W_K$: $\mathcal{O}(n \cdot d_{\text{model}} \cdot d_k)$

Compute $QK^T$: $\mathcal{O}(n^2 \cdot d_k)$

#### CASE II: Combined

Compute $X W_{QK}$: $\mathcal{O}(n \cdot d_{\text{model}}^2)$

Compute $(X W_{QK}) X^T$: $\mathcal{O}(n^2 \cdot d_{\text{model}})$

If $d_k \ll d_{\text{model}}$ (e.g., $d_k = 128$, $d_{\text{model}} = 512$):

Original: $\mathcal{O}( n \cdot 512 \cdot 128)+ \mathcal{O}( n \cdot 512 \cdot 128) + \mathcal{O}(n^2 \cdot 128)$

Combined: $\mathcal{O}(n \cdot 512^2) + \mathcal{O}(n^2 \cdot 512)$

As you can see the number of matrix multiplication is 3, but the total elementwise multiplication is very large.

## Multi-head Latent Attention

The main reason behind using the variants of **Attention** is that we always to increase our inference speed and we are always bottlenecked by [KV cache](https://cohlem.github.io/sub-notes/kv-cache-gqa/) The KV cache needed in original Multi-head attention is $2\cdot nh\cdot hd\cdot l$
for one token, as the tokens get large during inference, the memory needed for storing this case also increases.

Deepseek propose the use of latent dimension to compress the dimension.

As we know $K,V$ both come from the same x i.e $K=xWk$ and $V=xWv$ but the different weights $Wk, Wv$

how about we make an intermediate compressed version of x, from which we can decompress it into K and V, and only store that compressed version of x. This is what they use for multi-head latent attention.

$W_{\text{dkv}}$: compression matrix of size ($d_{\text{model}}$, $d_{\text{c}}$)
$L_{\text{kv}}$= $xW_{\text{dkv}}$ which is the compressed version of x

We decompress $L_{\text{kv}}$ into K, V using $Wuk, Wuv$ i.e

$Q=xWq$

$Kc=L_{\text{kv}} \cdot Wuk$ ($Wuk$ size = ($d_{\text{c}}, nh \cdot hd$))

$Vc=L_{\text{kv}} \cdot Wuv$ ($Wuv$ size = ($d_{\text{c}}, nh \cdot hd$))

Similarly, we can substitute those in our original attention equation

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(Q)(Kc)^\top}{\sqrt{d_k}}\right)(Vc)W_O
$$

As you can see, we've increased the number of matrix multiplication (i.e the computation of $L_{\text{kv}}$= $xW_{\text{dkv}}$), but the total number of elementwise multiplication can be made comparable with the right choice of compression dimension $d_{\text{c}}$

But, our main goal was to reduce the number of KV cache, but if we store only the $L_{\text{kv}}$ only, we still would would need to perform $Kc=L_{\text{kv}} \cdot Wuk$ and $Vc=L_{\text{kv}} \cdot Wuv$ to calculate attention. So whats the point of this compression?

Well there's a trick to still store only the $L_{\text{kv}}$ and use it without calculating Kc and Vc, we do weight combination like in our Q2 but still end up with less number of elementwise matrix multiplication. The equation above can also we written as

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(xWq)(L_{\text{kv}} \cdot Wuk)^\top}{\sqrt{d_k}}\right)(L_{\text{kv}} \cdot Wuv)W_O
$$

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{(x(Wq Wuk^\top) (L_{\text{kv}})^\top}{\sqrt{d_k}}\right)L_{\text{kv}}(WuvW_O)
$$

After combining $(Wq Wuk^\top)$ and $(WuvW_O)$ once, we can save $L_{\text{kv}}$ in our cache and then directly multiply with $(Wq Wuk^\top)$ to get the attention, without needing to calculate $Kc$ and $Vc$. Remember the issue we had while combining weights in Q2, this fades away because of the compression dimension because it strictly has to be less than $nh \cdot hd$ i.e ($d_{\text{c}} \ll nh \cdot hd$)

### Decoupled RoPE

There are still some parts I feel like I don't understand completely. But, here's what I've understood till now.

First thing to keep in mind, its the clever weight absorption design and caching only $L_{\text{kv}}$ that helps MLA to retain its fast speed. But, we have yet to apply positional information to our Q and K i.e

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{R1(xWq)(R2(L_{\text{kv}} \cdot Wuk))^\top}{\sqrt{d_k}}\right)(L_{\text{kv}} \cdot Wuv)W_O
$$

In the paper, they say the RoPE matrix gets in the middle but I don't know what i am missing here cause $({R1(xWq)Wuk^\top)L_{\text{kv}}^\top R2^\top}$ so weights can still be combined? I think there something that I don't understand here. I'll correct my understand in future.

lets resume from the point that RoPE matrix gets in the middle. They again design a clever thing here. Add two new matrices $W_{QR}\in \mathbb{R}^{(d,nh \cdot d^R)}$ and $W_{KR} \in \mathbb{R}^{(d,d^R)}$

$Q^{R}=RoPE(xW_{QR})$ and $K^{R}=RoPE(xW_{KR})$

and also cache the $K^R$

add then concatenate the new two matrices to the original Q and K

$Q = [Q, Q^R]$

$K = [K, K^R]$

and then perform our original attention.

$$
Q \cdot K^\top = [Q, Q^R] \cdot [K, K^R]^\top
$$

$$
Q \cdot K^\top = [Q \cdot K + Q^R \cdot K^R]
$$

as you can see the original Q and K are still preserved meaning we can still absorb the weights.

Total cached elements will be $K^R$ and $L_{kv}$ so the total saved cache will be $(d^R + dc)l$

#### Question

Why is second dimension of $W_{KR}$ is $d^R$ and not $(nh \cdot d^R)$
Meaning $d^R$ will be broadcasted across all the heads.

My guess is that they found that keeping only $d^R$ would produce decent result and would also save the cache memory requirement.

#### References

https://arxiv.org/abs/2405.04434
https://liorsinai.github.io/machine-learning/2025/02/22/mla.html#multi-head-latent-attention
