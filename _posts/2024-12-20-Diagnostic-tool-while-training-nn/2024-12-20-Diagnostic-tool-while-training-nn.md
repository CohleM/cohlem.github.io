---
layout: blog
author: cohlem
tags: [optimization]
---

source: [ Building makemore Part 3: Activations & Gradients, BatchNorm](https://www.youtube.com/watch?v=P6sfmUTpUmc&t=3677s)

## Things to look out for while training NN

Take a look at [previous notes](http://cohlem.github.io/sub-notes/batchnormalization/) to understand this note better

consider we have this simple 6 layer NN

```python
# Linear Layer
g = torch.Generator().manual_seed(2147483647) # for reproducibility


class Layer:
    def __init__(self,fan_in, fan_out, bias=False):
        self.w = torch.randn((fan_in, fan_out),generator = g) / (fan_in)**(0.5) # applying kaiming init
        self.bias = bias
        if bias:
            self.b = torch.zeros(fan_out)

    def __call__(self, x):
        y = x @ self.w
        self.out = y + self.b if self.bias else y
        return self.out


    def parameters(self):

        return [self.w] + [self.b] if self.bias else [self.w]

class Tanh:

    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out

    def parameters(self):
        return []



class BatchNormalization1:
    def __init__(self,nf, eps= 1e-5, mom=0.1):
        self.bngain = torch.ones(nf)
        self.bnbias = torch.zeros(nf)
        self.out = None
        self.mom = mom
        self.training = True
        self.running_mean = torch.ones(nf)
        self.running_var = torch.zeros(nf)
        self.eps = eps

    def __call__(self,x):

        if self.training:
            meani = x.mean(0, keepdim = True)
            vari = x.var(0, keepdim = True)

        else:
            meani = self.running_mean
            vari = self.running_var

        if self.training:
            with torch.no_grad():
                self.running_mean = (1-self.mom)*self.running_mean + self.mom*meani
                self.running_var = (1-self.mom)*self.running_var + self.mom*vari

        self.out = self.bngain *((x - meani)/ torch.sqrt(vari + self.eps)) + self.bnbias

        return self.out

    def parameters(self):
        return [self.bngain, self.bnbias]

```

**Structure**

```python
import torch.nn.functional as F


x = torch.randn(32, 30, generator = g)
y = torch.tensor([random.randint(0,26) for _ in range(32)] )

# Embedding layer,
n_embd = 10
n_vocab = 27
n_dim = 100
batch_size = 32
C = torch.randn((n_vocab,n_embd))


st = [
    # x shape = 32, 30
    Layer(n_embd*block_size,n_dim), Tanh(),
    Layer(n_dim, n_dim), Tanh(),
    Layer(n_dim, n_dim) , Tanh(),
    Layer(n_dim, n_dim), Tanh(),
    Layer(n_dim, n_dim), Tanh(),
    Layer(n_dim, n_vocab),BatchNormalization1(n_vocab)
]


with torch.no_grad():
    st[-1].bngain *= 0.1

    for layer in st[:-1]:
        if isinstance(layer, Layer):
            layer.w *= 5/3



parameters = [C] + [p for l in st for p in l.parameters()]
for p in parameters:
    p.requires_grad = True
```

**Training Loop**

```python

for iteration in range(200000):

    # for iteration in range(2000):
    idx = torch.randint(0,Xtr.shape[0], (batch_size,))
    x_emb = C[Xtr[idx]].view(-1, block_size * n_embd)
    x = x_emb
    for idx,item in enumerate(st):
#         print(idx)
        x = item(x)

    loss = F.cross_entropy(x,y)

    for layer in st:
        layer.out.retain_grad()

    for p in parameters:
        p.grad = None

    loss.backward()

    lr = 0.1 if iteration < 150000 else 0.01
    for p in parameters:

        p.data += -lr*p.grad

    if iteration % 10000 ==0:
        print(loss.data)

#     if iteration >= 1000:
#         break
```

let's look at our activations before initializing weights using kaiming init.

```python
# these are just part of modified code from the code that's given above.
class Layer:
    def __init__(self,fan_in, fan_out, bias=False):
        self.w = torch.randn((fan_in, fan_out),generator = g) # / (fan_in)**(0.5) # commenting our the kaiming init

# part of code
with torch.no_grad():
    st[-1].bngain *= 0.1

    for layer in st[:-1]:
        if isinstance(layer, Layer):
            layer.w *= 1.0 # setting gains to 1.0 (no gain)


```

### Activation plot

![fig1](/assets/images/2024-12-20-Diagnostic-tool-while-training-nn/fig1.png)

As you can see almost all the pre activations are saturated, this is because our weight is initialized in such a way that after applying tanh, most of our output values lie in -1 and 1, which will stop gradient propagation.

Now applying kaiming init with with no gain.

```python
# these are just part of modified code from the code that's given above.
class Layer:
    def __init__(self,fan_in, fan_out, bias=False):
        self.w = torch.randn((fan_in, fan_out),generator = g) / (fan_in)**(0.5) # applying the kaiming init

# part of code
with torch.no_grad():
    st[-1].bngain *= 0.1

    for layer in st[:-1]:
        if isinstance(layer, Layer):
            layer.w *= 1.0 # setting gains to 1.0 (no gain)


```

![fig2](/assets/images/2024-12-20-Diagnostic-tool-while-training-nn/fig2.png)

The plot is starting to look nicer, because there is less saturation, because now values don't lie in the extreme values of tanh, and gradient will be propagated. But we still have issue, as we can see the standard deviation is decreasing this is because of the property of tanh, i.e it squashes the values, initially (blue plot) the output was decent but in later layers, the distribution is being shrinked that because of the property of tanh.

now let's apply kaiming init with gain too, for tanh the gain is 5/3.

![fig3](/assets/images/2024-12-20-Diagnostic-tool-while-training-nn/fig3.png)
Now the values are being evenly distributed, and the standard deviation is stable (doesn't decrease with iteration).

We have to precisely measure the gains to have a stable training. But the introduction of batch normalization changes the case, and we don't have to be that much aware for precisely initializing weights.

Let's now apply the batch normalization but without kaiming init and see the same plot.

```
st = [
    # x shape = 32, 30
    Layer(n_embd*block_size,n_dim), BatchNormalization1(n_dim), Tanh(),
    Layer(n_dim, n_dim), BatchNormalization1(n_dim), Tanh(),
    Layer(n_dim, n_dim), BatchNormalization1(n_dim), Tanh(),
    Layer(n_dim, n_dim), BatchNormalization1(n_dim), Tanh(),
    Layer(n_dim, n_dim), BatchNormalization1(n_dim), Tanh(),
    Layer(n_dim, n_vocab),BatchNormalization1(n_vocab)
]
```

![fig4](/assets/images/2024-12-20-Diagnostic-tool-while-training-nn/fig4.png)
The output values are properly distributed, with very less saturation and a constant standard deviation.

### Gradient plot

The gradient distribution at each layers would look like this when the pre activations are batch normalized.
![fig5](/assets/images/2024-12-20-Diagnostic-tool-while-training-nn/fig5.png)

### Gradient to data ratio plot

![fig6](/assets/images/2024-12-20-Diagnostic-tool-while-training-nn/fig6.png)
This is what the ratio of gradient (calculated after backprop) to data plot looks like.
x-axis represent iterations, y represent the exponents. Ideally, 1e-3 is suitable and that ratio should lie around that line. If the ratio is below that line it means, we need to step up our learning rate, and if it is higher than that line we need to lower our learning rate.

The gain that we add during kaiming init has direct correlation with this plot.

```
with torch.no_grad():
  # last layer: make less confident
  layers[-1].gamma *= 0.1
  #layers[-1].weight *= 0.1
  # all other layers: apply gain
  for layer in layers[:-1]:
    if isinstance(layer, Linear):
      layer.weight *= 0.3
```

![fig7](/assets/images/2024-12-20-Diagnostic-tool-while-training-nn/fig7.png)
as you can see, when I make gain to 0.3 the ratio significantly varies, i.e ratio for later layers are around 1e-1.5, which mean we would have to lower our learning rate because of this gain change.

So the gain significantly affects our learning rate, but it doesn't affect other plots that we plot above, because it's controlled by batch normalization.

So we don't get a free pass to assign these gains arbitrarily, because it affects our gradients (as seen from the ratio plot). If we don't worry about these gains, we have to tune these learning rates properly (by increasing or decreasing the learning rate).

These data is analyzed throughout the training of NN

### NOTE to myself

after any operation look out for how the output's standard deviation changes, we should always maintain the std of 1

for instance while doing the dot production attention,

Q @ K.T

the output's std grows by sqrt of last embedding or head dimension, which is the reason why we scale it by the sqrt of that last embedding dimension.

Similarly, in skip connections too the addition of x back to the output introduces increase in std, we should scale that down too as i've mentioned [here](https://cohlem.github.io/sub-notes/optimizing-loss/)
