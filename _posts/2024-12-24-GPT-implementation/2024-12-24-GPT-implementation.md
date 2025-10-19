---
layout: blog
author: cohlem
tags: [architecture]
---

```python
# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

```python



with open('input.txt', 'r', encoding='utf-8') as f:
    data = f.read()
```

```python
from torch import nn
import torch
```

```python
vocab = sorted(list(set(data)))
len(data)

stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for s,i in stoi.items()}


encode = lambda x: [stoi[i] for i in x]
decode = lambda x: ''.join([itos[i] for i in x])
```

```python
type(data)
```

    str

```python
Xtr = data[:int(0.9*len(data))]
Xval = data[int(0.9*len(data)):]
```

```python
block_size = 8
batch_size = 32

def get_split(X):
    idx = torch.randint(0,len(X) - block_size, (batch_size,)) # we subtract block_size from total len of X, because w'll be taking next characters starting from the idx to the total len of block_size
    Xb =  torch.tensor([encode(X[i:i+block_size]) for i in idx]) # now our d should be 32,8
    Yb = torch.tensor([encode(X[i+1:i+1+block_size]) for i in idx])

    return Xb,Yb
```

```python

```

## A simple bigram language model with only embedding parameters

```python


n_vocab = len(stoi)
# emb_dim = 64

class BigramLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_layer = nn.Embedding(n_vocab, n_vocab)

    def forward(self,x,targets=None):
        loss = None
        logits = self.emb_layer(x)
#         logits.view(emb_dim)


        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits,targets)

        return logits,loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits, _ = self(idx) # idx is shape (B,T), logits is B,T,C
            probs = logits[:,-1,:] #probs is shape (B,C)
            probs = F.softmax(probs, dim = 1)
            idx_new = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_new), dim = 1)

        return idx


model = BigramLM()
```

#### Mini-batch gradient descent

```python
for idx in range(10000):
    Xb,Yb = get_split(Xtr)
    logits,loss = model(Xb,Yb)

    for p in model.parameters():
        p.grad = None
    # backprop
    loss.backward()

    #update the parameters
    lr = 0.1
    # mini-batch gradient descent
    for p in model.parameters():
        p.data += -lr*p.grad
print(loss)

```

    tensor(2.8175, grad_fn=<NllLossBackward0>)

![adam](https://builtin.com/sites/www.builtin.com/files/styles/ckeditor_optimize/public/inline-images/adam-optimization-5.png)

#### Adam optimizer Manually

```python
m = {idx: torch.zeros_like(p) for idx,p in enumerate(model.parameters())}
v = {idx: torch.zeros_like(p) for idx, p in enumerate(model.parameters())}

b1,b2 = 0.9, 0.999
e = 1e-8

```

```python


for idx in range(10000):
    Xb,Yb = get_split(Xtr)
    logits,loss = model(Xb,Yb)

    for p in model.parameters():
        p.grad = None
    # backprop
    loss.backward()

    #update the parameters
    lr = 0.1

    # Adam optimizer
    for i,p in enumerate(model.parameters()):
        m[i] = b1*m[i] + (1-b1)*(p.grad)
        v[i] = b2*v[i] + (1-b2)*(p.grad**2)

        m_corrected = m[i]/ (1-b1**(idx+1))
        v_corrected = v[i]/ (1-b2**(idx+1))


        p.data += (-lr*m_corrected)/ ((v_corrected + e)**0.5)


print(loss)

```

    tensor(2.5091, grad_fn=<NllLossBackward0>)

#### Adam from pytorch

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
```

```python

```

```python


for idx in range(10000):
    Xb,Yb = get_split(Xtr)
    logits,loss = model(Xb,Yb)

    optimizer.zero_grad(set_to_none=True)
    # backprop
    loss.backward()
    optimizer.step()

print(loss)

```

    tensor(2.5577, grad_fn=<NllLossBackward0>)

```python
import torch.nn.functional as F


print(decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))
```

    zDe her

    C, pinG:
    bluiqQPZmaJhe bQZ;jarEW;t fOLtoul, tkYSvJu melmad my myoJDtLjz3ag cat haslZfbspJtour3vkO;NK
    BhQr
    CZQouObZf?L-QV&OJGW
    Ad a O t.
    ZMDJJ'ncARFxS thean,
    FFsqARICpmedUuvWShoureenure ckDn'qJDkhaha:xRbZQouIZ.

    ven ha woure ise aloEWLKme.QMCsMXXtheaGrilkEYjQSvehourVpCin wateesVy N.B'Z.Hse u'. -y pRClisto wher hiTTRL-!, hoRinout n emeaHmarorne ilRVA,
    IOpmZot&TAlDYLqppJ'it&Zitr s pligGWGJt gMn b:


    x, t al:
    I tIO!Ic, Qf tinE:RriFfsWfs?Bvhou ss -Ej:

    FxHPhe ingeuredve th$Drbe, t Ox'e sthem.
    Cs thitoFnWBu o se sTTQxRrahera:
    T!

    I l oFFDADP
    Briter: mouureIN-?CU3 tXceiW$g o ithen;bAql$, g txpr w.
    u SOq-Nawhbinded pr, ;gpr-'cWDno wRun wiead3lveeLZf pIShadOLven w?., tha hisedt NCs stVMG. o3SRPhie3thaYJWQIsthou ngrREJ-tpe;WMXpdeeatLreditFCXPer'Phe thOqL-, t, hJWqrscQ3toubeOxQ3y'ZHIvA3
    Hry
    VIfandm&et cGWbeN-HNarelQver3p


    as t ir.-YK:

    ILHN?ly n ZLYCMh l aClle oungjAMHY,de hef amiRikn sc?c,

    KI QwP, ??e 3SSUl bSIYS, t, fO;SKLg m lktror ffiriMherrour D ll ORt m
    ar ast E-d mee ely hoing

### Attention experimentation

```python
Xb.shape
```

    torch.Size([32, 8])

```python
v = torch.ones(5,4)
v
```

    tensor([[1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.],
            [1., 1., 1., 1.]])

```python
v = torch.softmax(v, dim =1)
```

```python
v
```

    tensor([[0.2500, 0.2500, 0.2500, 0.2500],
            [0.2500, 0.2500, 0.2500, 0.2500],
            [0.2500, 0.2500, 0.2500, 0.2500],
            [0.2500, 0.2500, 0.2500, 0.2500],
            [0.2500, 0.2500, 0.2500, 0.2500]])

lets say we have v vector with 5 characters and each character has 3 feature

now we need to also include the relationship of one character with it's previous characters. How do we do it?

- find the average of one with it's previous characters.

we need to make sure that a character doesn't see it's future characters and only see the previous characters.

For instance, char at ind 0 can only look at itself and char at 1 can only look at char at ind 0 and itself, and so on.

```python
v = torch.randn(5,3)
v
```

    tensor([[-0.1586, -0.5878, -1.0289],
            [ 0.1123,  2.1602,  1.1508],
            [-0.7969, -2.1239,  1.4866],
            [ 1.0644,  1.1567, -0.5879],
            [-0.2015, -1.6920, -0.0972]])

```
1 0 0 0 0     [-0.5134, -0.3769, -0.6881]
1 1 0 0 0     [ 0.1477,  0.1931, -0.4826]
1 1 1 0 0   X [ 1.0117,  0.4637, -0.9426]
1 1 1 1 0     [-0.0454, -0.7803,  0.0046]
1 1 1 1 1     [-0.3021, -0.0271,  1.1680]
```

What do we get? sum across each columns with limited to its previous rows.

```python
i = torch.ones(v.shape[0], v.shape[0])
i
```

    tensor([[1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.],
            [1., 1., 1., 1., 1.]])

```python
i = torch.tril(i)
i
```

    tensor([[1., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.]])

```python
i = i/i.sum(1, keepdim= True)
i
```

    tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000],
            [0.5000, 0.5000, 0.0000, 0.0000, 0.0000],
            [0.3333, 0.3333, 0.3333, 0.0000, 0.0000],
            [0.2500, 0.2500, 0.2500, 0.2500, 0.0000],
            [0.2000, 0.2000, 0.2000, 0.2000, 0.2000]])

```python
z = i @ v
z
```

    tensor([[-0.1586, -0.5878, -1.0289],
            [-0.0232,  0.7862,  0.0610],
            [-0.2811, -0.1838,  0.5362],
            [ 0.0553,  0.1513,  0.2551],
            [ 0.0039, -0.2174,  0.1847]])

so this i is similar to what we have in transformer i.e attention weights (dot product of Q and K). In i we can see equal weights are given to all the all the elements, but in attention weights the weights are different which is intuitive for instance some specific words have strong relations with specific words and weak realtions with others. The weights represent how much of a focus should be given to specific words (character in our case)

```python
v
```

    tensor([[-0.1586, -0.5878, -1.0289],
            [ 0.1123,  2.1602,  1.1508],
            [-0.7969, -2.1239,  1.4866],
            [ 1.0644,  1.1567, -0.5879],
            [-0.2015, -1.6920, -0.0972]])

```python
v.T
```

    tensor([[-0.1586,  0.1123, -0.7969,  1.0644, -0.2015],
            [-0.5878,  2.1602, -2.1239,  1.1567, -1.6920],
            [-1.0289,  1.1508,  1.4866, -0.5879, -0.0972]])

```python
aw = v @ v.T
aw
```

    tensor([[ 1.4293, -2.4717, -0.1546, -0.2439,  1.1266],
            [-2.4717,  6.0035, -2.9667,  1.9416, -3.7895],
            [-0.1546, -2.9667,  7.3556, -4.1787,  3.6097],
            [-0.2439,  1.9416, -4.1787,  2.8163, -2.1144],
            [ 1.1266, -3.7895,  3.6097, -2.1144,  2.9129]])

```python

```

masking was be done using torch.tril but for normalization, we can't simply call softmax on the above aw becuase exp(0) = some value.

we need to replace those zeros with some values that when exponetiated becomes 0. and that is -infinity

```python
block_size = 8
```

```python
tril = torch.tril(torch.ones(aw.shape[0], aw.shape[0]))
```

```python
tril
```

    tensor([[1., 0., 0., 0., 0.],
            [1., 1., 0., 0., 0.],
            [1., 1., 1., 0., 0.],
            [1., 1., 1., 1., 0.],
            [1., 1., 1., 1., 1.]])

```python
aw
```

    tensor([[ 1.4293, -2.4717, -0.1546, -0.2439,  1.1266],
            [-2.4717,  6.0035, -2.9667,  1.9416, -3.7895],
            [-0.1546, -2.9667,  7.3556, -4.1787,  3.6097],
            [-0.2439,  1.9416, -4.1787,  2.8163, -2.1144],
            [ 1.1266, -3.7895,  3.6097, -2.1144,  2.9129]])

```python
mask = tril[:block_size, :block_size]
```

```python
mask == 0
```

    tensor([[False,  True,  True,  True,  True],
            [False, False,  True,  True,  True],
            [False, False, False,  True,  True],
            [False, False, False, False,  True],
            [False, False, False, False, False]])

```python
aw = aw.masked_fill(mask == 0, float('-inf'))
```

```python
torch.softmax(aw, dim= 1)
```

    tensor([[1.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [2.0852e-04, 9.9979e-01, 0.0000e+00, 0.0000e+00, 0.0000e+00],
            [5.4715e-04, 3.2873e-05, 9.9942e-01, 0.0000e+00, 0.0000e+00],
            [3.2004e-02, 2.8466e-01, 6.2566e-04, 6.8271e-01, 0.0000e+00],
            [5.2653e-02, 3.8584e-04, 6.3070e-01, 2.0601e-03, 3.1420e-01]])

```python

```

# Head

```python
from torch import nn
```

**Scaling after Q.K**

We suspect that for large values of dk, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients 4. To counteract this effect, we scale the dot products by √1 .
dk

```python
emb_dim = 128
block_size = 8


class Head(nn.Module):
    def __init__(self,h_dim):
        super().__init__()
        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self,x):
        B,T,C = x.shape
        Q,K,V = self.wq(x), self.wk(x), self.wv(x)


        # comment out if using multi head attention
        ### ------ multi-head ----------------
        n_heads = emb_dim // h_dim
        Q = Q.view(B,T,n_heads, h_dim)
        K = K.view(B,T,n_heads, h_dim)
        V = V.view(B,T,n_heads, h_dim)

        Q = torch.transpose(Q, 1,2) # transposing (n_head, block_size) cause we'll do matmul operation on block_size and h_dim
        K = torch.transpose(K, 1,2) # transposing (n_head, block_size) cause we'll do matmul operation on block_size and h_dim
        V = torch.transpose(V, 1,2) # transposing (n_head, block_size) cause we'll do matmul operation on block_size and h_dim

        ### ------ multi-head ----------------
        aw = Q @ torch.transpose(K, -2,-1) # for matmul dim of q should be B,T,C and k should be B,C,T

        aw = aw/(emb_dim **0.5)

        mask = self.tril[:T,:T] == 0 # generate mask
        aw = aw.masked_fill_(mask, float('-inf')) # apply mask i.e fill true values with -inf


        aw = torch.softmax(aw,dim=-1) # -inf values are converted to 0 and then each row is normalized

        cv = aw @ V # context vector

        cv = torch.transpose(cv, 1,2) # bring it back to (B,T,n_heads, h_dim)

        cv = cv.contiguous().view(B,T,-1)

        return cv


```

```python
ans.shape
```

    torch.Size([32, 8, 128])

```python
heads = inp.view(inp.shape[0],inp.shape[1], n_heads, h_dim).transpose(-2,-3)
```

```python
another = inp.view(inp.shape[0],inp.shape[1], n_heads, h_dim).transpose(-2,-3)
```

```python
heads.shape
```

    torch.Size([32, 4, 8, 32])

```python
another.shape
```

    torch.Size([32, 4, 8, 32])

```python
heads @ another.transpose(-2,-1)
```

    tensor([[[[ 32.1742,   3.5751,  -3.5662,  ...,  -2.2413,  -1.7000,   3.9770],
              [  3.5751,  32.8265,  -3.9191,  ...,   1.1454,   0.2817,   1.4899],
              [ -3.5662,  -3.9191,  49.9453,  ...,   4.7897,   9.8889,  -3.4950],
              ...,
              [ -2.2413,   1.1454,   4.7897,  ...,  24.6098,  10.0202,   5.2172],
              [ -1.7000,   0.2817,   9.8889,  ...,  10.0202,  28.6793,   1.9070],
              [  3.9770,   1.4899,  -3.4950,  ...,   5.2172,   1.9070,  36.0254]],

             [[ 22.7656,  -2.3078,   9.4733,  ...,   6.2143,   4.7908,   4.0378],
              [ -2.3078,  28.7329,  -6.9455,  ...,   3.3861,   1.6332,  -2.1739],
              [  9.4733,  -6.9455,  31.9591,  ...,   5.3679,  -0.4923,   1.1153],
              ...,
              [  6.2143,   3.3861,   5.3679,  ...,  37.6356,   5.0559,   0.7349],
              [  4.7908,   1.6332,  -0.4923,  ...,   5.0559,  24.6002,   2.4460],
              [  4.0378,  -2.1739,   1.1153,  ...,   0.7349,   2.4460,  18.2817]],

             [[ 31.6518,   5.8290,   6.3662,  ...,   0.6282,  -5.5420,   2.3825],
              [  5.8290,  32.2843,  -3.0601,  ...,   2.2674,   2.9054,  -2.9447],
              [  6.3662,  -3.0601,  30.9779,  ...,   7.3485,  -2.3613,  -2.9850],
              ...,
              [  0.6282,   2.2674,   7.3485,  ...,  17.7225,   1.8768,  -7.2604],
              [ -5.5420,   2.9054,  -2.3613,  ...,   1.8768,  28.1906,  -4.7192],
              [  2.3825,  -2.9447,  -2.9850,  ...,  -7.2604,  -4.7192,  21.8654]],

             [[ 37.2365,  -3.7616,  -9.2002,  ...,  -7.2032,  -4.0573,  10.2151],
              [ -3.7616,  21.1687,   0.4588,  ...,  -2.2633,   3.6384,   1.9324],
              [ -9.2002,   0.4588,  33.7734,  ...,  -2.4673,   1.9866,  -5.8346],
              ...,
              [ -7.2032,  -2.2633,  -2.4673,  ...,  31.7197,   0.5054,  -5.8715],
              [ -4.0573,   3.6384,   1.9866,  ...,   0.5054,  35.4172,  -6.8851],
              [ 10.2151,   1.9324,  -5.8346,  ...,  -5.8715,  -6.8851,  44.4438]]],


            [[[ 45.6578,   3.4671,   9.5396,  ...,   3.6376,   5.9042,   5.0866],
              [  3.4671,  31.2397,  -1.2889,  ...,  -3.2443,  -1.7427,   6.0298],
              [  9.5396,  -1.2889,  18.8146,  ...,   0.3659,   0.9351,  -0.3261],
              ...,
              [  3.6376,  -3.2443,   0.3659,  ...,  41.2272,   5.0229,   9.9165],
              [  5.9042,  -1.7427,   0.9351,  ...,   5.0229,  47.7995,   4.8620],
              [  5.0866,   6.0298,  -0.3261,  ...,   9.9165,   4.8620,  29.9559]],

             [[ 34.8406,  -1.0729,  -2.4909,  ..., -15.5392,   2.4406,   3.9956],
              [ -1.0729,  50.9441,  -1.6156,  ...,  -0.8506,   6.1251,  -0.7462],
              [ -2.4909,  -1.6156,  29.7794,  ...,  17.3859,   5.3271,  -0.5394],
              ...,
              [-15.5392,  -0.8506,  17.3859,  ...,  50.3442,   1.7252,  -5.7926],
              [  2.4406,   6.1251,   5.3271,  ...,   1.7252,  30.1659,   7.9424],
              [  3.9956,  -0.7462,  -0.5394,  ...,  -5.7926,   7.9424,  24.4478]],

             [[ 32.2363,   4.5509,   0.6994,  ...,   3.8885,   3.2419,  -3.5590],
              [  4.5509,  44.1102,   0.2451,  ...,   8.3753,   9.8859,  10.8134],
              [  0.6994,   0.2451,  18.9293,  ...,   3.8037,  -4.0057,  -0.4459],
              ...,
              [  3.8885,   8.3753,   3.8037,  ...,  30.4812,   9.0369,  -0.4821],
              [  3.2419,   9.8859,  -4.0057,  ...,   9.0369,  32.8905,   6.6835],
              [ -3.5590,  10.8134,  -0.4459,  ...,  -0.4821,   6.6835,  25.8820]],

             [[ 34.1055,  -2.9306,   0.6626,  ...,  11.1533,  -4.0041,   9.7521],
              [ -2.9306,  27.5158,  -3.4747,  ...,   0.1160,  11.5869,  -6.7454],
              [  0.6626,  -3.4747,  28.3060,  ...,  -3.1335,   0.3503,   5.5252],
              ...,
              [ 11.1533,   0.1160,  -3.1335,  ...,  33.5075,  -4.5291,   2.0836],
              [ -4.0041,  11.5869,   0.3503,  ...,  -4.5291,  32.7881,   2.4404],
              [  9.7521,  -6.7454,   5.5252,  ...,   2.0836,   2.4404,  40.9715]]],


            [[[ 32.0416,   0.0788,   6.5413,  ...,  -6.5551,  -3.2799,  -7.8908],
              [  0.0788,  31.7799,   2.8320,  ...,   0.6807,   0.5974,  -5.8561],
              [  6.5413,   2.8320,  36.3656,  ...,   5.4645,   1.2012,  -7.6004],
              ...,
              [ -6.5551,   0.6807,   5.4645,  ...,  30.7091,  -5.8358,  -2.6483],
              [ -3.2799,   0.5974,   1.2012,  ...,  -5.8358,  42.6208,   3.1208],
              [ -7.8908,  -5.8561,  -7.6004,  ...,  -2.6483,   3.1208,  38.9232]],

             [[ 37.5842,   4.2069,   6.1104,  ...,  -5.6760,   3.6003,   2.9112],
              [  4.2069,  25.2379,  12.2942,  ...,  -5.3360,  -9.7890,  -9.9670],
              [  6.1104,  12.2942,  41.9160,  ...,  -6.0786,   1.5327, -12.3278],
              ...,
              [ -5.6760,  -5.3360,  -6.0786,  ...,  35.6687,  -3.0921,   2.0084],
              [  3.6003,  -9.7890,   1.5327,  ...,  -3.0921,  39.4154,   8.4038],
              [  2.9112,  -9.9670, -12.3278,  ...,   2.0084,   8.4038,  36.2056]],

             [[ 46.8421,  -0.4574,   1.4663,  ..., -11.0569,   4.3132,   2.6288],
              [ -0.4574,  29.3338,   2.6641,  ...,  -4.7041,  -6.4938,  -0.8643],
              [  1.4663,   2.6641,  32.9588,  ...,  -4.2076,  -7.0425,  -1.0215],
              ...,
              [-11.0569,  -4.7041,  -4.2076,  ...,  20.5085,   3.6718,   6.4799],
              [  4.3132,  -6.4938,  -7.0425,  ...,   3.6718,  29.9442,   4.7719],
              [  2.6288,  -0.8643,  -1.0215,  ...,   6.4799,   4.7719,  39.3404]],

             [[ 30.1868, -18.2841,  -3.8556,  ...,   0.1748, -11.6281,   7.9357],
              [-18.2841,  33.2663,  -0.5568,  ...,   4.7933,   7.4713,  -1.5922],
              [ -3.8556,  -0.5568,  33.3571,  ...,  -5.5591,  -1.8302,  -2.3288],
              ...,
              [  0.1748,   4.7933,  -5.5591,  ...,  42.2914,  -3.1656,  -0.0794],
              [-11.6281,   7.4713,  -1.8302,  ...,  -3.1656,  37.9287,  -2.7775],
              [  7.9357,  -1.5922,  -2.3288,  ...,  -0.0794,  -2.7775,  20.1431]]],


            ...,


            [[[ 27.1805,   1.7525,   1.2874,  ...,  -0.7088,  -5.4087,  -2.1454],
              [  1.7525,  37.2680,  -7.7314,  ...,  -5.4847,  -0.3849,  10.4835],
              [  1.2874,  -7.7314,  35.9060,  ...,   5.6001,   3.8431,  -0.8432],
              ...,
              [ -0.7088,  -5.4847,   5.6001,  ...,  21.4171,   4.0599,   1.9034],
              [ -5.4087,  -0.3849,   3.8431,  ...,   4.0599,  38.6673, -15.6571],
              [ -2.1454,  10.4835,  -0.8432,  ...,   1.9034, -15.6571,  46.7569]],

             [[ 22.2746,  -1.5875,   2.0462,  ...,  -4.3275,   1.6363,   3.8162],
              [ -1.5875,  24.4395,  -6.1820,  ...,   6.4587,  -4.6774,   0.1828],
              [  2.0462,  -6.1820,  30.9477,  ...,  -0.0719,   5.0252,   2.4537],
              ...,
              [ -4.3275,   6.4587,  -0.0719,  ...,  28.9538,  -1.4443,  -3.5094],
              [  1.6363,  -4.6774,   5.0252,  ...,  -1.4443,  26.9600,   4.2082],
              [  3.8162,   0.1828,   2.4537,  ...,  -3.5094,   4.2082,  20.0763]],

             [[ 40.1019,   1.1669,  -7.5864,  ...,   4.5362,   4.3235,  -0.7608],
              [  1.1669,  38.2663,   2.0250,  ...,   3.9493,  -2.9794,   3.8610],
              [ -7.5864,   2.0250,  26.2510,  ...,   4.9898,  -2.2455,   4.8609],
              ...,
              [  4.5362,   3.9493,   4.9898,  ...,  25.6067,   0.0972,   1.2879],
              [  4.3235,  -2.9794,  -2.2455,  ...,   0.0972,  15.2263,   0.2495],
              [ -0.7608,   3.8610,   4.8609,  ...,   1.2879,   0.2495,  24.2671]],

             [[ 43.5244,  -6.5812,   6.9048,  ..., -16.0361,   4.4655, -10.7278],
              [ -6.5812,  30.5437,  -2.1458,  ...,   6.1033,   5.0552,  -4.5733],
              [  6.9048,  -2.1458,  29.7066,  ...,  -8.8169,  -0.4333, -11.8848],
              ...,
              [-16.0361,   6.1033,  -8.8169,  ...,  41.8999,  10.9827,  11.8192],
              [  4.4655,   5.0552,  -0.4333,  ...,  10.9827,  35.2686,   6.1599],
              [-10.7278,  -4.5733, -11.8848,  ...,  11.8192,   6.1599,  33.8409]]],


            [[[ 32.0467,   7.0052,  -3.2416,  ...,   8.1532,  -0.2888,  -6.5871],
              [  7.0052,  25.7955,  -8.1634,  ...,   7.4437,  -1.6669,  -2.2903],
              [ -3.2416,  -8.1634,  23.1504,  ...,  -9.1981,  -1.7050,   3.3145],
              ...,
              [  8.1532,   7.4437,  -9.1981,  ...,  40.4675,   1.8017,   2.0262],
              [ -0.2888,  -1.6669,  -1.7050,  ...,   1.8017,  23.5782,  -6.3969],
              [ -6.5871,  -2.2903,   3.3145,  ...,   2.0262,  -6.3969,  28.7649]],

             [[ 21.0464,   0.6068,  -1.7797,  ...,   2.1086,  -2.0139,  -3.8878],
              [  0.6068,  27.2985,   3.3299,  ...,  -4.8720,   0.4252,  -3.0846],
              [ -1.7797,   3.3299,  30.7588,  ...,   3.5386,  -0.9481,  -9.6544],
              ...,
              [  2.1086,  -4.8720,   3.5386,  ...,  35.5470,   6.3743,   0.3064],
              [ -2.0139,   0.4252,  -0.9481,  ...,   6.3743,  41.8936,  14.2506],
              [ -3.8878,  -3.0846,  -9.6544,  ...,   0.3064,  14.2506,  40.4118]],

             [[ 26.6901,  -1.6078,  -0.9821,  ...,  -0.6358,  -7.5112,   1.0814],
              [ -1.6078,  33.0559,   6.3802,  ...,   4.5554,   5.6757,  -3.5736],
              [ -0.9821,   6.3802,  22.2404,  ...,  -3.5796,   4.5527,  -4.9432],
              ...,
              [ -0.6358,   4.5554,  -3.5796,  ...,  20.0463,   4.5129,   1.3352],
              [ -7.5112,   5.6757,   4.5527,  ...,   4.5129,  30.3458,   2.0281],
              [  1.0814,  -3.5736,  -4.9432,  ...,   1.3352,   2.0281,  26.2497]],

             [[ 34.7860,   1.1120,   1.4674,  ...,   2.8627,  -3.8121,  -4.6312],
              [  1.1120,  27.1577,  -8.0268,  ...,  -4.4698,   1.6233,  -9.6746],
              [  1.4674,  -8.0268,  37.0896,  ...,   6.1201,   2.8779,   4.9796],
              ...,
              [  2.8627,  -4.4698,   6.1201,  ...,  17.8634,   3.2803,   0.4458],
              [ -3.8121,   1.6233,   2.8779,  ...,   3.2803,  20.7591,  -2.5509],
              [ -4.6312,  -9.6746,   4.9796,  ...,   0.4458,  -2.5509,  23.5079]]],


            [[[ 28.4154,   2.7645, -10.1010,  ...,   0.0741, -13.8531,   1.3159],
              [  2.7645,  21.8306,  -6.2272,  ...,  -2.9116,  -3.2048,  -3.4533],
              [-10.1010,  -6.2272,  31.9452,  ...,  -1.5767,  10.7117,   6.3105],
              ...,
              [  0.0741,  -2.9116,  -1.5767,  ...,  25.0870,   2.7461,   1.8960],
              [-13.8531,  -3.2048,  10.7117,  ...,   2.7461,  26.7191,  -0.1613],
              [  1.3159,  -3.4533,   6.3105,  ...,   1.8960,  -0.1613,  33.3621]],

             [[ 25.6422,  -3.1636,  -2.0446,  ...,   7.4708,   2.9027,   1.4148],
              [ -3.1636,  20.2049,   0.1776,  ...,  -4.4680,  -3.0735,   2.2424],
              [ -2.0446,   0.1776,  26.3722,  ..., -13.9194,  -3.2235,  -1.9986],
              ...,
              [  7.4708,  -4.4680, -13.9194,  ...,  32.6086,  -2.0325,  -4.2793],
              [  2.9027,  -3.0735,  -3.2235,  ...,  -2.0325,  26.0867,   4.6897],
              [  1.4148,   2.2424,  -1.9986,  ...,  -4.2793,   4.6897,  22.4440]],

             [[ 42.5938,  -7.6218,  -9.7269,  ...,  -2.8314, -10.9695,   7.8923],
              [ -7.6218,  28.2443,   0.2591,  ...,  -5.1364,   4.3846,  -3.9063],
              [ -9.7269,   0.2591,  32.0604,  ...,   0.5306,  10.6320,  -1.2622],
              ...,
              [ -2.8314,  -5.1364,   0.5306,  ...,  35.8231,  -1.7779,  -6.9608],
              [-10.9695,   4.3846,  10.6320,  ...,  -1.7779,  28.1199,   3.4098],
              [  7.8923,  -3.9063,  -1.2622,  ...,  -6.9608,   3.4098,  34.7994]],

             [[ 45.7010,  -4.8081,   6.9660,  ...,  12.1433,  -0.8634,   8.6209],
              [ -4.8081,  33.8269,  -2.8159,  ...,   0.0864,   5.1836,  -3.6970],
              [  6.9660,  -2.8159,  48.0178,  ...,   4.8133,  -4.5906,   8.6223],
              ...,
              [ 12.1433,   0.0864,   4.8133,  ...,  34.4357,  -9.2983,  -4.1290],
              [ -0.8634,   5.1836,  -4.5906,  ...,  -9.2983,  25.1852, -10.2649],
              [  8.6209,  -3.6970,   8.6223,  ...,  -4.1290, -10.2649,  39.4607]]]])

### Combining our BigramLM with our heads

```python


n_vocab = len(stoi)
emb_dim = 128

class BigramLM(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.emb_layer = nn.Embedding(n_vocab, emb_dim)
        self.mha  = Head(h_dim)
        self.proj = nn.Linear(emb_dim, n_vocab, bias = False)

    def forward(self,x,targets=None):
        loss = None
        x_embed = self.emb_layer(x)
#         print('embed', x_embed)

        x_attn = self.mha(x_embed)
#         print('attn', x_attn)

        logits = self.proj(x_attn)
#         print('logits', logits)
#         logits.view(emb_dim)

        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits,targets)

        return logits,loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits, _ = self(idx[:,-block_size]) # idx is shape (B,T), logits is B,T,C
            probs = logits[:,-1,:] #probs is shape (B,C)
            probs = F.softmax(probs, dim = 1)
            idx_new = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_new), dim = 1)

        return idx


model = BigramLM(32)
```

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for idx in range(10000):
    Xb,Yb = get_split(Xtr)
    logits,loss = model(Xb,Yb)

    optimizer.zero_grad(set_to_none=True)
    # backprop
    loss.backward()
    optimizer.step()

print(loss)

```

    tensor(2.2740, grad_fn=<NllLossBackward0>)

```python
print(decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))
```

    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[466], line 1
    ----> 1 print(decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=1000)[0].tolist()))


    Cell In[464], line 33, in BigramLM.generate(self, idx, max_new_tokens)
         31 def generate(self, idx, max_new_tokens):
         32     for i in range(max_new_tokens):
    ---> 33         logits, _ = self(idx) # idx is shape (B,T), logits is B,T,C
         34         probs = logits[:,-1,:] #probs is shape (B,C)
         35         probs = F.softmax(probs, dim = 1)


    File ~/anaconda3/envs/deep_learning/lib/python3.12/site-packages/torch/nn/modules/module.py:1736, in Module._wrapped_call_impl(self, *args, **kwargs)
       1734     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1735 else:
    -> 1736     return self._call_impl(*args, **kwargs)


    File ~/anaconda3/envs/deep_learning/lib/python3.12/site-packages/torch/nn/modules/module.py:1747, in Module._call_impl(self, *args, **kwargs)
       1742 # If we don't have any hooks, we want to skip the rest of the logic in
       1743 # this function, and just call forward.
       1744 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1745         or _global_backward_pre_hooks or _global_backward_hooks
       1746         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1747     return forward_call(*args, **kwargs)
       1749 result = None
       1750 called_always_called_hooks = set()


    Cell In[464], line 16, in BigramLM.forward(self, x, targets)
         13         x_embed = self.emb_layer(x)
         14 #         print('embed', x_embed)
    ---> 16         x_attn = self.mha(x_embed)
         17 #         print('attn', x_attn)
         19         logits = self.proj(x_attn)


    File ~/anaconda3/envs/deep_learning/lib/python3.12/site-packages/torch/nn/modules/module.py:1736, in Module._wrapped_call_impl(self, *args, **kwargs)
       1734     return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
       1735 else:
    -> 1736     return self._call_impl(*args, **kwargs)


    File ~/anaconda3/envs/deep_learning/lib/python3.12/site-packages/torch/nn/modules/module.py:1747, in Module._call_impl(self, *args, **kwargs)
       1742 # If we don't have any hooks, we want to skip the rest of the logic in
       1743 # this function, and just call forward.
       1744 if not (self._backward_hooks or self._backward_pre_hooks or self._forward_hooks or self._forward_pre_hooks
       1745         or _global_backward_pre_hooks or _global_backward_hooks
       1746         or _global_forward_hooks or _global_forward_pre_hooks):
    -> 1747     return forward_call(*args, **kwargs)
       1749 result = None
       1750 called_always_called_hooks = set()


    Cell In[463], line 35, in Head.forward(self, x)
         32 aw = aw/(emb_dim **0.5)
         34 mask = self.tril[:T,:T] == 0 # generate mask
    ---> 35 aw = aw.masked_fill_(mask, float('-inf')) # apply mask i.e fill true values with -inf
         38 aw = torch.softmax(aw,dim=-1) # -inf values are converted to 0 and then each row is normalized
         40 cv = aw @ V # context vector


    RuntimeError: The size of tensor a (9) must match the size of tensor b (8) at non-singleton dimension 3

### Combining the previous model with Feedforward network

```python
torch.relu(torch.tensor(0))
```

    tensor(0)

```python
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(emb_dim, 4*emb_dim, bias= True)
        self.layer2 = nn.Linear(4*emb_dim, emb_dim, bias = True)

    def forward(self,x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        return x



```

```python


n_vocab = len(stoi)
emb_dim = 128

class BigramLM(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.emb_layer = nn.Embedding(n_vocab, emb_dim)
        self.mha  = Head(h_dim)
        self.FFN = FFN()
        self.proj = nn.Linear(emb_dim, n_vocab, bias = False)

    def forward(self,x,targets=None):
        loss = None
        x_embed = self.emb_layer(x)
#         print('embed', x_embed)

        x_attn = self.mha(x_embed)
#         print('attn', x_attn)
        x_ffn = self.FFN(x_attn)
        logits = self.proj(x_ffn)

#         print('logits', logits)
#         logits.view(emb_dim)

        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits,targets)

        return logits,loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
            logits, _ = self(idx[:,-block_size]) # idx is shape (B,T), logits is B,T,C
            probs = logits[:,-1,:] #probs is shape (B,C)
            probs = F.softmax(probs, dim = 1)
            idx_new = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_new), dim = 1)

        return idx


model = BigramLM(32)
```

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for idx in range(10000):
    Xb,Yb = get_split(Xtr)
    logits,loss = model(Xb,Yb)

    optimizer.zero_grad(set_to_none=True)
    # backprop
    loss.backward()
    optimizer.step()

print(loss)

```

    tensor(1.9205, grad_fn=<NllLossBackward0>)

### Layernormalization

```python
class LayerNormalization(nn.Module):
    def __init__(self,emb_dim, eps= 1e-5, mom=0.1):
        super().__init__()
        self.bngain = nn.Parameter(torch.ones(emb_dim))
        self.bnbias = nn.Parameter(torch.zeros(emb_dim))
        self.out = None

        self.eps = eps

    def forward(self,x):
        meani = x.mean(-1, keepdim = True)
        vari = x.var(-1, keepdim = True)
        self.out = self.bngain *((x - meani)/ torch.sqrt(vari + self.eps)) + self.bnbias
        return self.out

```

```python

```

```python
ln = LayerNormalization(emb_dim)
len(list(ln.parameters()))
```

    2

```python
ans = ln(torch.randn(32,8,128))
```

```python
ans[-1,-1,:].std(), ans[-1,-1,:].mean()
```

    (tensor(1.0000), tensor(0.))

### combine previous model with layer normalization and skip connections + positional embedding

```python
class Block(nn.Module):
    def __init__(self,h_dim):
        super().__init__()
        self.mha = Head(h_dim)
        self.FFN = FFN()
        self.ln1 = LayerNormalization(emb_dim)
        self.ln2 = LayerNormalization(emb_dim)

    def forward(self,x):
        x = self.mha(self.ln1(x)) + x
        x = self.FFN(self.ln2(x)) + x

        return x

```

```python
block_size
```

    8

```python


n_vocab = len(stoi)
emb_dim = 128

class BigramLM(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.emb_layer = nn.Embedding(n_vocab, emb_dim)
        self.pos_emb = nn.Embedding(block_size, emb_dim)
        self.ln = LayerNormalization(emb_dim)
        self.proj = nn.Linear(emb_dim, n_vocab, bias = False)

        ## NEW
        self.block = Block(h_dim)


    def forward(self,x,targets=None):
        loss = None

        x_embed = self.emb_layer(x)
        x_pos = self.pos_emb(torch.ones_like(x) * torch.arange(x.shape[1]))

        x_block = self.block(x_embed + x_pos)
        x_ln = self.ln(x_block)

        logits = self.proj(x_ln)

#         print('logits', logits)
#         logits.view(emb_dim)

        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits,targets)

        return logits,loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
#             print('idx', idx.shape)
            logits, _ = self(idx[:,-block_size:]) # idx is shape (B,T), logits is B,T,C
#             print('logits', logits.shape)
            probs = logits[:,-1,:] #probs is shape (B,C)
            probs = F.softmax(probs, dim = 1)
            idx_new = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_new), dim = 1)

        return idx

```

```python

```

```python
model = BigramLM(32)
```

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for idx in range(10000):
    Xb,Yb = get_split(Xtr)
    logits,loss = model(Xb,Yb)

    optimizer.zero_grad(set_to_none=True)
    # backprop
    loss.backward()
    optimizer.step()

print(loss)

```

    tensor(1.7881, grad_fn=<NllLossBackward0>)

```python
print(decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=10000)[0].tolist()))
```

    To the delies.

    BRUTUS:
    Prifting king head!
    And neved
    somes drelose, been of his cannot lord, our you know;
    Now!
    A provoses;
    But king, out in that them! Lord well the eath this acan that I beaunture will to a cher forsely,
    that lord, of Julietteruse.

    VOLIO:
    Nor most. Ret.
    The now, sir: our thing he whom
    Tybarn for your head Tis many amply cutdendereizen: bear, buldown'd to so world crain as room meet, what heave misprus im; That blowisdoms finour to.

    LADUKE VINCENTIO:
    If Juchamfect better's Mayour Tray have prite warre. I ploy:

    RICHARD:
    When not now:
    Shall
    Than whoming will of heave goant dart? be is from lies
    with we that to in her your ful us;
    Sheechy, my
    with the most beaute you,
    With that you way this ma dare showed,--it held:
    An him ne'er.

    DUMERCENTIO:
    At Too curse,--but in crown needy?

    RIVERCUTIO:
    An would have chie,
    For courable he uncoth!

    GLOUCESTER:
    Now you ear threw wost play, as muciefuied
    to kind of gods naskly Eaductione, and and grow thite I cawares, as a son,. Gentlemy lieger is rail.

    JULIET:
    If thire torm: I our How not the his: grace have pun in soues of this turnclelaime, that sain wor good, look!

    And gold ward I; wilcompelse letwen dition
    Wors;
    For when your a the cisee of as me, my gived.

    AUTOLYCUS:
    The him true me kilgood us.

    KING HENRY Duke:
    Abanifemself.

    POLIXENES:
    Kore his curagese way their vowlyss,
    And the county my most thy be grace, feful advice you his traimsself.

    DUKE VINGH:
    Unto do,
    Is hath may grate: him you are drowning Help Tyband to have is shall and rums be as to his true of thee, let then heads. My look suments
    Cown: he well bedumfor wife;
    If not thought rly at siguers:
    To with:
    Hear-profes wift our the ento knew! say, no him of have I Cluld banish wefore in babiend full plieve on hip neverd, as touccretor.

    FOLAURENCENTIO:
    Which them all melecton mydel pegame, traed gold bling to it in confall what, 'erd her nore in bid in these is to the feely praily Cruay no lesty pries, why drould wrut beauntonessy good been prince an to hate goddeity were besir,
    Get in besters
    Of gate by crue, wear sorrow:
    Cher you licks.

    NORTESBY:
    God,
    Feath the straugh faity,
    To greep and word,
    A her;
    I love tearly yourses and his his ever ERWICK been this murds mine end an you to panit their plaimself.

    For break her, so been mest before,
    If Herm is nothing againe a what we quading by the const again, leave you ware Eork, thee; we it?
    The was that have to dow! were womaniught sorraw our noughtly hath thunnituded, swouldie, I that smoth, with lossion
    Of to crownce
    On
    with your shion;
    For Loncetters drown,
    Auf: that, Laster, my charink,
    He is being that for thoughtiviles abace lost towarchemself
    All not a is curs: your lathere, and spawn
    why the fries, sire wiserved, sweek to and not a'es Balmaties up woman of no law no him that himself life.

    YORK:
    He paintames it.

    PARIS:
    Richar life,
    And prop. I dring of noble too live swey couse;
    And on, to to grave clows.

    SICINIUS:
    Warwick'd at an and man
    Rome stry notion beat, and hath and guee.

    Fith od the lrover,
    And too! what it hoarrate, birt.

    LADY CAPULET:
    And end.

    Seconce;
    Did of him a trity
    To dedue exts.

    LUCIO:
    Ay, gentlemany the sway.

    WARD IV:
    His fools the heart!

    AUTOLYCUSHXENENIUS:
    Let dream!

    ANTIO:
    The be to death:
    My fresir;
    The dotwy
    Freath such ecry seem of tune why couse thee: he city
    To breath The him.
    Fhretly that be suph show.

    FRIAR TCompt
    seeceitacking Here.

    NORTIUS:
    Nor Lord thy his are I lo:
    My theeisue
    Warwice headly.
    The entone:
    This pretebeardy;
    How his coung lovery thand inshries
    To my knal awe him!

    NORTENSIO:
    Now true, limes answelcous?

    ROMEO:
    Husin!
    You cosise,
    With us,
    And pare she swifter's to him, your king of the
    an old for you purmptrese gently
    Will Lieve your bread of with am this Yoint litted.

    She honours:
    Whereign, And to sattereith it you
    Snothird noble taabbegitle by blew, we live but than train.

    LUMNIE:
    NencAnger intake bries, dead wear her:
    And for of your the Will to grace shall not ilse would custoly caperfell nry: 't,
    Thou world tone bloody upon they
    Lord friecto his me the ily? I stand one, our again'd fice,
    What not.

    LADY CARIANA:
    Tas
    For have irituation my not her made wrrange a now,
    An I way
    Capiece? Pray, your reaking my Will dowrongifformenten all at boot there all'd ara were headagentlemany libeand wron, ror made handraw heat, Causurpaciouth grace a what, doth his unperfeecome blead accame, our sward on hath, safe.

    MENE:
    There they fal straitagen.

    BUCHII:
    The the mine with for whod not upon yet plighform of the her licessighte we Straighted, a hastomberate, daught,
    These not are all every lord not dean that your creeemember'st living
    That feath.
    Now My Rome.

    For to no my has but Rive me no day, as
    Wrom the intriber gipful handGause previngbrood's we pleave thy king for heave we house have you nut and chould Dearis; and I ward in off this of unands exceedied life shing morcauself,
    As to they welcome I will for Hunbry of thy be slay? howbread, and grace a befied banise deaute is befrom cation.

    CLAUDIO:
    When, I am, I wear'd peoply tongue to Pet:

    Bounmen a before gook known:
    By we office,
    We to his clam: the king liff.

    HY:
    What?
    If You knew loved muctand you maded him, perouse thou beare not cretory we the not unjusty he reterefore is do well to her die down left me our much:
    Had b'd thy as ban abour will Anclose fair by to may of Qaint you had and this deed
    Eveth triatureing our'd and But show that hand,
    And inster?

    GLOUCESTENSIO:
    A whom come than even,
    You are thy brothery youty, I a gentle should's was way turoung most, ast all ie not in hey
    To Lordad no soul, servantale thy be narly;
    And good agaffect, colse I may is king?
    If crown'd ear Volsun
    ender'd life
    An hearts pedine thou blang!
    I mism pare saidf all in face? Was and melcower audio.
    Hand infalse?

    First we; in Serving be to her:
    No,tis stommone: when way,
    BusHe Where cure in eltal much will my prever alm in to-myself spear ing beg and in serving cong in all wighter me you deful of thy biends
    Boing it wave wronBy are none:
    Then this way holy.

    GLOUCESTER:
    Hare!
    Now'd his nonour she hous sham of a die, up make would hought heir ha,--O lose to refitsed for prom a grave
    To cannot lips.
    But bettence,
    To most put dead; thou wrong is unto
    truld Lance to it tabmand with to where jut in if have I couse,
    As to pue deson.

    CLIFFORTEGBROKE:
    If Lord do.
    The gads I was peacefore womber must your councie again. Wepty you vike ear swence a name too lady as honour great him as nread?

    GLOUCESTER:
    Pauly, Kneet, wither:
    Angelo;
    Wher your shion.

    KINGHA:
    The duspire me an ay, if turetory wron abound,
    And to casoland hall confaithe body their chood grouble!
    Our Nay, we devile him;
    And have of our raim.

    ESCALUS:
    Nor yow
    To perd Frand mee lords, tell
    Which his can fair a mand kead Ke too live the lice;
    I her sceecond be duke?
    To pretbeen hit love,
    No Let him ly know:
    Histend 'twas I love: my down the mertia; trunt, him breard goold to Wack the sajes. Comfor horse:
    I henose amany of they waunt,
    And I can in they out weep of I'll tirrought,
    I Plonds;
    Whinher,' bate fight ent is a scept to my talk see long
    As I know word;
    Of you, by why are pail the but man
    Than on us mall betry
    Rage re to out surp, my Duke I queen the battly:
    Iven thy banishould life, gentlize:
    In gethat my will to not Benouuts and Laint were wife well gume head
    at eart, and Ge: hy came none, if I landlay this part my bodanced: we what now.

    BRAMENE:
    You holy very; life
    No, uncady
    have us,
    Aways threak out,
    Nor grains
    Dears, hearewe of herd!
    A like enour us?

    LADY CARIANA:
    The us, boot thy hong Warwice asquested me and Messervy.

    Frothungent Behalt.

    HASAR:
    And for thank God Kenews, trofes, incies, shall so tell: know.

    GLOUCESTER:
    Thout, am name swear his,
    Nay, somes, if aughter, whence,
    And prence to see joing one,' delish a eximieT;
    For But hand.

    FRIAR TGAUNT:
    For labeing presence.

    JULIET:
    The grow.

    BISHARD GAUNT:
    You they do his perfee
    I shall prater cled
    With in at my leads
    Whamontway, up mase gh; untard to begs.

    QUEEN RICHAM: I will book his how Parisors;
    For I bleave I prepeding Romeo her down.

    Seconfore, as breams' dear of you man's will peecemes, Auffacenty all consmialt accould, what hall will what me how, I'll better chade and there would sure; Very noble
    He I week to no slay all Richarison?

    Seces.

    ROMEO:
    We
    Pault Tune all bear of thes hast they lish you with thy forturisign,
    And his war abin,
    Servan:
    Anborn pile as to Puscause?

    BUCHOTCESTIO:
    It such fercy,
    And tune your bold,
    The for they homined
    Gentleman, bein the boy!
    Speadowar, and you
    And ty?

    LUCIO:
    No end?

    Por man
    Off your loftious,
    Thou good Cepray your the deady!
    Of OF YORK:
    My from me dreaty powere with to kinship wronger:
    Whom bears; be all Clainsman:
    What veinst.

    VOLIO:
    Romfore,
    And traught were flamo's life,
    And her Some let comperch, that straise?

    FLORIOLANUS:
    And wea, I been this feart her all perouse all goo-me!

    MARGARET:
    Untor younsed but conums death sleave as be, weetaitold
    Of amost
    And be noble not suffer:
    How at I'll huson or have her were trould were an with debook!
    As lorderst what do, I hace an what him!
    So of heaven erein him should not
    where tim:
    I be which should Leadies, to at a be tot the home, if you; her I for else the not the ruitest majesty sunce;
    For it.

    See do the entestan of in godly ow Had--owisdies,
    And shear posity, Here lendly.

    MENIUS:
    Affatter beggard die,
    Our breed, and
    Mon's persmand Goding brraation:
    And such'd aff in to cours,
    But life; and for the lance garewell did them to Madam this nable not holk to despet ailse earies,
    Would mreadfing Most, me true have
    Or Corion? she enou harm in.

    QUEEN:
    Ay?
    Ha's with woman's son. OVERDITA:
    Abaservanty not there him trains
    An I the city too is ghood leign canners, poof more that on advici him? their perful of gualf, acd to your bayou:
    And without, in of where'er writy.
    For'd by ha; and not mine.

    BURY:
    Gresty let's
    IUS:
    O leep-bawful be; alse our dead this disceet speak his the guilt to

    ANGELO:
    Tulaties sealn forge p

as we can see there's high quality output after the addition of positional embedding

```python

```

```python
sample_emb = nn.Embedding(block_size, n_vocab)
```

```python
a = torch.randn(32,8)
```

```python
b = torch.ones_like(a)*torch.arange(8)
```

```python
b.shape
```

    torch.Size([32, 8])

```python

```

```python
sample_emb
```

B,T each T'th dimension should have the numbers between 0, block_sizem

```python

```

```python

```

```python

```

### Log of all losses

**initial using adamW 10K iter**

2.601

**After multi-head-attentnion 10k iter**

2.316

**After FFN 10k iter**

1.9205

**After LayerNormalization**
1.9252

**After skip-connections**
2.0718

**After positional embedding**
1.7881

```python

```

### Putting it all together

```python
# We always start with a dataset to train on. Let's download the tiny shakespeare dataset
!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

```python
with open('input.txt', 'r', encoding='utf-8') as f:
    data = f.read()
```

```python
from torch import nn
import torch
```

```python
vocab = sorted(list(set(data)))
len(data)

stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for s,i in stoi.items()}


encode = lambda x: [stoi[i] for i in x]
decode = lambda x: ''.join([itos[i] for i in x])
```

```python
Xtr = data[:int(0.9*len(data))]
Xval = data[int(0.9*len(data)):]
```

```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

```python
device
```

    'cpu'

```python

batch_size = 32

def get_split(X):
    idx = torch.randint(0,len(X) - block_size, (batch_size,)) # we subtract block_size from total len of X, because w'll be taking next characters starting from the idx to the total len of block_size
    Xb =  torch.tensor([encode(X[i:i+block_size]) for i in idx]) # now our d should be 32,8
    Yb = torch.tensor([encode(X[i+1:i+1+block_size]) for i in idx])

    return Xb.to(device),Yb.to(device)
```

```python
eval_iter = 200

@torch.no_grad()
def evaluate_loss():
    out = dict()

    model.eval()
    for item in ['train', 'val']:
        if item == 'train':
            losses = torch.zeros(eval_iter)
            for k in range(eval_iter):

                Xb,Yb = get_split(Xtr)
                _, loss = model(Xb,Yb)
                losses[k] = loss
            out[item] = losses.mean()

        if item == 'val':
            losses = torch.zeros(eval_iter)
            for k in range(eval_iter):

                Xb,Yb = get_split(Xval)
                _, loss = model(Xb,Yb)
                losses[k] = loss
            out[item] = losses.mean()

    model.train()
    return out


```

```python
emb_dim = 128
block_size = 8


class Head(nn.Module):
    def __init__(self,h_dim):
        super().__init__()
        self.wq = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=False)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self,x):
        B,T,C = x.shape
        Q,K,V = self.wq(x), self.wk(x), self.wv(x)


        # comment out if using multi head attention
        ### ------ multi-head ----------------
        n_heads = emb_dim // h_dim
        Q = Q.view(B,T,n_heads, h_dim)
        K = K.view(B,T,n_heads, h_dim)
        V = V.view(B,T,n_heads, h_dim)

        Q = torch.transpose(Q, 1,2) # transposing (n_head, block_size) cause we'll do matmul operation on block_size and h_dim
        K = torch.transpose(K, 1,2) # transposing (n_head, block_size) cause we'll do matmul operation on block_size and h_dim
        V = torch.transpose(V, 1,2) # transposing (n_head, block_size) cause we'll do matmul operation on block_size and h_dim

        ### ------ multi-head ----------------
        aw = Q @ torch.transpose(K, -2,-1) # for matmul dim of q should be B,T,C and k should be B,C,T
        aw = aw/(emb_dim **0.5)
        mask = self.tril[:T,:T] == 0 # generate mask
        aw = aw.masked_fill_(mask, float('-inf')) # apply mask i.e fill true values with -inf
        aw = torch.softmax(aw,dim=-1) # -inf values are converted to 0 and then each row is normalized

        cv = aw @ V # context vector
        cv = torch.transpose(cv, 1,2) # bring it back to (B,T,n_heads, h_dim)
        cv = cv.contiguous().view(B,T,-1)

        return cv


```

```python
class FFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(emb_dim, 4*emb_dim, bias= True)
        self.layer2 = nn.Linear(4*emb_dim, emb_dim, bias = True)

    def forward(self,x):
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.layer2(x)
        x = torch.relu(x)
        return x

```

```python
class LayerNormalization(nn.Module):
    def __init__(self,emb_dim, eps= 1e-5, mom=0.1):
        super().__init__()
        self.bngain = nn.Parameter(torch.ones(emb_dim))
        self.bnbias = nn.Parameter(torch.zeros(emb_dim))
        self.out = None

        self.eps = eps

    def forward(self,x):
        meani = x.mean(-1, keepdim = True)
        vari = x.var(-1, keepdim = True)
        self.out = self.bngain *((x - meani)/ torch.sqrt(vari + self.eps)) + self.bnbias
        return self.out

```

```python
class Block(nn.Module):
    def __init__(self,h_dim):
        super().__init__()
        self.mha = Head(h_dim)
        self.FFN = FFN()
        self.ln1 = LayerNormalization(emb_dim)
        self.ln2 = LayerNormalization(emb_dim)

    def forward(self,x):
        x = self.mha(self.ln1(x)) + x
        x = self.FFN(self.ln2(x)) + x

        return x

```

```python


n_vocab = len(stoi)
emb_dim = 128
block_size = 16
h_dim = 32
n_blocks = 4

class BigramLM(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.emb_layer = nn.Embedding(n_vocab, emb_dim)
        self.pos_emb = nn.Embedding(block_size, emb_dim)
        self.ln = LayerNormalization(emb_dim)
        self.proj = nn.Linear(emb_dim, n_vocab, bias = False)

        ## NEW
        self.blocks = nn.Sequential(*[Block(h_dim) for _ in range(4)])


    def forward(self,x,targets=None):
        loss = None

        x_embed = self.emb_layer(x)
        x_pos = self.pos_emb(torch.ones_like(x) * torch.arange(x.shape[1]))

        x_block = self.blocks(x_embed + x_pos)
        x_ln = self.ln(x_block)

        logits = self.proj(x_ln)

#         print('logits', logits)
#         logits.view(emb_dim)

        if targets is not None:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = nn.functional.cross_entropy(logits,targets)

        return logits,loss

    def generate(self, idx, max_new_tokens):
        for i in range(max_new_tokens):
#             print('idx', idx.shape)
            logits, _ = self(idx[:,-block_size:]) # idx is shape (B,T), logits is B,T,C
#             print('logits', logits.shape)
            probs = logits[:,-1,:] #probs is shape (B,C)
            probs = F.softmax(probs, dim = 1)
            idx_new = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx,idx_new), dim = 1)

        return idx

```

```python
model = BigramLM(h_dim).to(device)
```

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
all_losses = {'train' : [], 'val' : []}

total_iter = 10000
for ind in range(total_iter):
    Xb,Yb = get_split(Xtr)
    logits,loss = model(Xb,Yb)
    if ind % eval_iter == 0 or ind == total_iter - 1:
        with torch.no_grad():
            eloss = evaluate_loss()
            all_losses['train'].append(eloss['train'].item())
            all_losses['val'].append(eloss['val'].item())
            print(f' step {ind}: losses: {eloss}')

    optimizer.zero_grad(set_to_none=True)
    # backprop
    loss.backward()
    optimizer.step()

print(loss)

```

     step 0: losses: {'train': tensor(4.0128), 'val': tensor(4.0197)}
     step 200: losses: {'train': tensor(2.2937), 'val': tensor(2.3103)}
     step 400: losses: {'train': tensor(2.1003), 'val': tensor(2.1448)}
     step 600: losses: {'train': tensor(1.9851), 'val': tensor(2.0530)}
     step 800: losses: {'train': tensor(1.9177), 'val': tensor(2.0016)}
     step 1000: losses: {'train': tensor(1.8552), 'val': tensor(1.9664)}
     step 1200: losses: {'train': tensor(1.8357), 'val': tensor(1.9464)}
     step 1400: losses: {'train': tensor(1.7834), 'val': tensor(1.9312)}
     step 1600: losses: {'train': tensor(1.7782), 'val': tensor(1.8990)}
     step 1800: losses: {'train': tensor(1.7337), 'val': tensor(1.8853)}
     step 2000: losses: {'train': tensor(1.7283), 'val': tensor(1.8673)}
     step 2200: losses: {'train': tensor(1.7080), 'val': tensor(1.8764)}
     step 2400: losses: {'train': tensor(1.6945), 'val': tensor(1.8577)}
     step 2600: losses: {'train': tensor(1.6876), 'val': tensor(1.8383)}
     step 2800: losses: {'train': tensor(1.6763), 'val': tensor(1.8354)}
     step 3000: losses: {'train': tensor(1.6648), 'val': tensor(1.8294)}
     step 3200: losses: {'train': tensor(1.6585), 'val': tensor(1.8239)}
     step 3400: losses: {'train': tensor(1.6514), 'val': tensor(1.8005)}
     step 3600: losses: {'train': tensor(1.6405), 'val': tensor(1.8000)}
     step 3800: losses: {'train': tensor(1.6208), 'val': tensor(1.7985)}
     step 4000: losses: {'train': tensor(1.6374), 'val': tensor(1.7847)}
     step 4200: losses: {'train': tensor(1.6226), 'val': tensor(1.7880)}
     step 4400: losses: {'train': tensor(1.6164), 'val': tensor(1.7778)}
     step 4600: losses: {'train': tensor(1.6057), 'val': tensor(1.7920)}
     step 4800: losses: {'train': tensor(1.6025), 'val': tensor(1.7803)}
     step 5000: losses: {'train': tensor(1.6009), 'val': tensor(1.7762)}
     step 5200: losses: {'train': tensor(1.5881), 'val': tensor(1.7756)}
     step 5400: losses: {'train': tensor(1.5792), 'val': tensor(1.7609)}
     step 5600: losses: {'train': tensor(1.5851), 'val': tensor(1.7565)}
     step 5800: losses: {'train': tensor(1.5715), 'val': tensor(1.7503)}
     step 6000: losses: {'train': tensor(1.5762), 'val': tensor(1.7453)}
     step 6200: losses: {'train': tensor(1.5695), 'val': tensor(1.7463)}
     step 6400: losses: {'train': tensor(1.5738), 'val': tensor(1.7371)}
     step 6600: losses: {'train': tensor(1.5639), 'val': tensor(1.7313)}
     step 6800: losses: {'train': tensor(1.5496), 'val': tensor(1.7253)}
     step 7000: losses: {'train': tensor(1.5621), 'val': tensor(1.7312)}
     step 7200: losses: {'train': tensor(1.5579), 'val': tensor(1.7246)}
     step 7400: losses: {'train': tensor(1.5555), 'val': tensor(1.7399)}
     step 7600: losses: {'train': tensor(1.5465), 'val': tensor(1.7323)}
     step 7800: losses: {'train': tensor(1.5550), 'val': tensor(1.7437)}
     step 8000: losses: {'train': tensor(1.5515), 'val': tensor(1.7444)}
     step 8200: losses: {'train': tensor(1.5386), 'val': tensor(1.7312)}
     step 8400: losses: {'train': tensor(1.5342), 'val': tensor(1.7294)}
     step 8600: losses: {'train': tensor(1.5440), 'val': tensor(1.7240)}
     step 8800: losses: {'train': tensor(1.5454), 'val': tensor(1.7259)}
     step 9000: losses: {'train': tensor(1.5388), 'val': tensor(1.7214)}
     step 9200: losses: {'train': tensor(1.5282), 'val': tensor(1.7161)}
     step 9400: losses: {'train': tensor(1.5255), 'val': tensor(1.7225)}
     step 9600: losses: {'train': tensor(1.5300), 'val': tensor(1.7152)}
     step 9800: losses: {'train': tensor(1.5304), 'val': tensor(1.7066)}
     step 9999: losses: {'train': tensor(1.5280), 'val': tensor(1.6964)}
    tensor(1.6668, grad_fn=<NllLossBackward0>)

```python
print(decode(model.generate(torch.zeros((1,1), dtype=torch.long), max_new_tokens=10000)[0].tolist()))
```

    Is you rathed in love as not.

    HENRY HENRY VI

    QUEEN ELIZABETH:
    By
    Till inevice the told,
    And one gall be so chaperted
    A friender'd misi' the time earth and put to land, he counting to pray.
    'Tis man's ma attemp
    From your seatter's this verienge:
    His mothers her hate, till do
    The lorkwin the forwards
    what his quarrel piece:
    The father's, comfort to be love that more speak one on courts living worthy, and in this of rumpts and inque against distroting.

    ROMEO:
    As Risichion: my power by my right,
    And me then we do the duke.

    POLIXENES:
    And surelvey did life
    once abelitter but of houseing cause
    As that he to tell plobb'd from the heart,
    And Did bearthy cousin here?

    ISABELLA:
    Sweet so bist gaze together, I would seem on Aumerle.
    Let think iI'd a goodness of frail from our revonaminate him curse maid, in this suit, for all askites
    As that a death.
    I'll never warshed our truth;
    O! offect late intercure
    Of seem upon hy gates.
    'Text God unreisold
    Which mine holds no monfaciars! say, damneta leave.
    I'll make thou
    calledHed are the title than theirs
    Gill dame follower.
    I'll know the worl,
    Did fortune and paucil'ht thou must be do not a Captes:
    If resign the hand thy hands
    Regentle, now, that cry thou
    shalt as crown, it without Rutland in the shouts it of my langue of strivinger with sweets the fasterporate summer's pance,
    And as flowers as even he spremd
    Say bloody labour graves? will stabb'd
    Speak no keep to comfort, as a cure, his that flouded,
    And fear now our mother, hy gose
    Not in good herds it.

    Clown:
    When-changed up the queen! lost circutio more beseech.

    AUFIBALT:
    I'll not, I table.

    PAULINA:
    Not to your poor hours be a brother's most with his fallow death this? his is turn your batter
    His both sad, be shall die.

    MENENIUS:
    Let's this hihe hath conque time for hwoford.

    GLOUCESTER:
    Go treach of:
    But, thou king, which the cit,
    You wonder'st coasters,
    Tigue so her thinks with your blood of the duke, of the son-nonegly for war
    Both, Richard whose are we not hath replain,
    She's hat true throus must you being woivern, in, sir, but let him for, death.
    Long hath forth water
    Where not; as the atches as one, Lordon, our giirly.

    Nurse:
    Besides, innocence
    extremainted follow me of late Romeo, whither from his that draid
    Which death, on, death stir.

    GLOUCESTER:
    Duke of Norford

    NORFITZWARD IV:
    Come, do he hath been suffe prever, 'tis a randers for tLuckinfarencation?

    CAMILLO:
    It is not the sads and well!

    DORSET:
    What
    But so the forth bile blot die very come,
    And we livise all visit
    With by the unclusure both, poor I think
    His very nothinging some someetness!

    KING RICHARD II:
    It what lose them
    You wasque
    Whiles ye here for a foul tallige tempt:
    Reggarding words
    Thee bun
    the seizes and it with to hither solume done: good for Paris! in you
    if all be but I have.

    TUS:
    You obey, In Thank your sood lovy; and, and come, that we revenged agree done, to speak: pall.

    Clown:
    My mooning-hative thus
    Is cause,
    To find as I task.

    GREMINGS:
    It is, thy cornater,
    As years s'That I am suit pave by curginary
    Keptry thee, that true times to us dire
    But if that before the pite of his bares her rounded. Romeo, English! on' that for a very's harm,

    KING RICHARD II:
    Thou now to the enemies;
    But not peace, to see that we say rimorally and raumer with brave in none;
    And allower's master, sir, encle breathing lust, home
    Is for soul dreath; if why, and, think themself, yet we, at name sorry vengeance camition an upon his kin myself!
    Yon rathe. You have tried help no father scolds him,
    Thou having enemies, let me, my your praise
    And that beenefit evocliate for this majesty?
    My hateful voice
    To perish the Romeo will
    Will die a gaint in anothing whither than sudden the trouble,
    women die, not who, I speak, pardon thou hast never come
    Made your dam libund in,
    That lestrice ble Rome die For I am in a summanch?
     be good stood is.

    BEPLANUS:
    Doy, sir? Tell me unright, go youth
    promised to the valmouragy: but ble thou but he, sweet their pardon
    You have he mes the virds will'd here-flowise.

    CORIOLANUS:
    Office,' death hill mine of labour;
    Pastes to revious
    And, not put upon's bloodight.

    ABHOMENES:
    I'll Montague: you no, deathe
    all day prince, pickingham my torsest makes, Rome newd is down; Werping and go as here,
    There'll wist go what laid with me,
    Madam.

    LUCIO:
    O you not.
    For not to shame resolume that fair you. God
    But given to do a day ois graving 'twith me unto that you.

    First Murderer:
    And, my haste, my soul slick, committed, in set's shook which your struck and yea do not be your deave me:
    Look's honour of your brave be rieven-with you beg, not mere, as?
    We having to the holy that I spoke for the win joint of your honouration; and wouldst eiterity death she will thensuited!
    For it is his head,
    Coriolanus, and hath proclaccamaster means, lin for my daying together.

    FRIAR LAURENCE:
    Your God's haste, as realm
    When I bear build
    Shall your hated,
    When all so this trust poss were
    A blothy soul the curse maties are as hereather of my dayage,
    And moving, but in your house,
    Cenry with him thinker her tent.' lay, thou lives a putish the feaster-should hither his life;
    Not sir; he's suddenly
    Whose seo scoves be no butely home, came this airs, that founds in his for the helds came?

    Nurse:
    And this me sorrow, bid a hestress
    That naturacles my late, conscail his love it down in that should is by this, holy tride a gyour story protected a wretched but do seat able,
    Agoing Angelowninges, as thy to thou art effrem. That is
    Lay'st give himself, as now,
    Till it long but speak I thy voice,
    If they slew my trainhern;
    And whose statute' the neudest, most all go fault our flier me;
    And I drate again plame;
    If, him we could like, in the Veion's?
    Has the tell for this athing goodness! he buried
    Of hear the mean Pilause with's all.
    It was not man can Richard,
    And with but prock on
    your ringelo go.

    ROMEO:
    I bed; and, I it would but And sell the ear as
    Jaughter's horse book
    For these cieving
    Before conscience was.
    To petty are od vauquin his acting rourse
    Hight is as her?

    KIVUTUS:
    As your, fail
    shall we do cormby nothing: that overse fatable before
    The warried may slinhame, herAth thou rights king:
    Up he shall ame love herd whence
    That's hither oher to heat.
    Not our puities,
    Proclaimins but is moved to be
    Sisters o'
    For that serviced me honour?

    GLOUCESTER:
    This is command mates? I
    dward thousand 'tis to tthy blood
    Which is your quincule,
    And let me honourable wenchequired by thy till steel win 'tis my our awry contiruis live my love?
    Time eason!
    Give no for your
    And the grave me? your comfortly crows to such which shath what would all the war
    nall bastacquied it for his mumber'd, madam.

    KATHAN ELTHANUS:
    No, fair I as or him weed
    excred in your dams, and myne?
    Hie, as they shall stogether, My darged againsting.
    Entake my stade and main
    True: his your hate; and pardonator:
    Mar lifthat slow
    Corance myself, it
    speak but a door
    namelly hither times ble good faster's the king.

    KINARWILL:
    The Roman of his king instand thence.

    Clown:
    It were a wound blost thou husb loards of fear; what siit
    Citizens! hrew enswordour,
    it kindly hat hat see
    Because him on his atten,
    And as the itself,' her, and losure her sir
    Infect lieging her!

    CLARENCE:
    'Tis for this virtuous soundaint one are one in olut
    Makes' suppur two shrifl'?

    BUCKINGHAM:
    I know not for safe-but him for enemy but a fourth her, on thinkrible?
    For thou see him
    That speak them sorrow
    Cheeks, we once this hatice
    Thregrictions would
    Heaven shall bit.-
    Villain and my hand lead
    Having but upon a brother matter.

    POMPEY:
    Disphat cannot give theirs,
    And lip to be great corlier; and, in thou set loves: upon
    Thou droop mistraction. Warwice,
    Thinks all the venure he low-thousame as one stir broke myself;
    For you're wench, but 'tward Cullor all thy reans the grans,
    He will in Lord Angelo.

    LEONTES:
    The court:
    O, he, he ha'! shall; but the insteach matter-grace from
    Lay her will at you array not where
    Thou hasting he shall never side them my orator
    This is hat I for slips note weeder'd in questeriancica many bold them'd; not shall hate she maugest thus lave.

    KING RICHARD III:
    We'll that therefordman begin us,
    And life, him I ne'er nammish.

    BAGOTHASAR:
    I shall he London alter
    On the rough blood
    hear lying shall
    far and that I know shall to dry I love.
    Courage Marshal and be are not warm
    nauresty pate, my put him we hoodes look to now: think i' the gods it seal do royalter:
    A, spoke look royalting I come on, come some
    One pray itself in concer,
    To the duyse my aid
    To enterupt of a struchia
    leturn their churchars? gonery. You have see to strange.
    O, against of Earl itis with firtushiour;
    And called after,
    He letdes on his prison.

    LEONTES:
    Nice on and my legs who take and die the hearts;
    If any backled and me?

    CORKERNES:
    And again that disported true, which, what, you beast our bring
    So.

    CORIOLANUS:
    You stand to be cred.

    FLORIZEL:
    No resternight nime; spareport, this to sincel.

    QUEEN ELIZABETH:
    O, will he car;
    Hus. Lewit your crown over with an yet,
    Or your own seaters-asideathesed woe;
    Right? 'Tis my brain.
    She's now, too, past will your eye bastacks mine than royalties up she's a word in.

    First King Henry's back as the world
    The constancingA:
    And they whose lays; I'll for purgafe:
    Peace, thoughts him
    Redeems his were by be hand thou, daughter, you see them
    that mocked to sent I have.

    First Lord:
    And the comfort harity them; sister on this, good strengefore his darkle foreign warrate a tries'd
    by. Traze
    Shate it would the Derbutiones, marry vault? O, the Polixenes that would longelo our hand you king? the seat, and villain,
    pite and to go what Noture to thee
    so proud me, ready upon and plick me, somethingerous home
    In her hat thus sentrengthen
    My news?

    MUREWILIA::
    But day not me but agree we'll teern's upond Servingman:
    O thou would to king's portyour
    lay never lack as the mind:
    He more, consequence.

    HOR OF YORK:
    And, for, sir, under Norfolknow
    It sir, and who's scure imploathing of dark, call'd you might, I'll guest

```python

```
