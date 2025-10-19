---
layout: blog
author: cohlem
tags: [optimization]
---

As we saw in our [previous note](https://cohlem.github.io/sub-notes/optimizing-loss/) how important it is to have the pre-activation values to be roughly gaussian (0 mean, and unit std). We saw how we can initialize our weights that make our pre-activation roughly gaussian by using Kaiming init. But, how do we always maintain our pre-activations to be roughly gaussian?

Answer: BatchNormalization

**Benefits**

- stable training
- preserves vanishing gradients

### BatchNormalization

As the name suggests, batches are normalized (across batches), by normalizing across batches we preserve the gaussian property of our pre-activations.

This is our initial code

```
# same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

    # forward pass
    emb = C[Xb] # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
    # Linear layer
    hpreact = (embcat @ W1 + b1) # hidden layer pre-activation

    # Non-linearity
    h = torch.tanh(hpreact) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass
    for p in parameters:
        p.grad = None

    hpreact.retain_grad()
    logits.retain_grad()
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())


```

We add batch normalization just before the all activation layers. Right now, we only have one linear layer so only one tanh activation is applied, in a big NN, we have to add those batch normalization before applying activations.

Applying batch normalization is quite simple.

```
hpreact = (hpreact - hpreact.mean(0, keepdim = True)) / hpreact.std(0, keepdim =True)
```

but by doing this we are forcing it to lie is a particular space. To add a little bit of entropy (randomness) and let to model learn itself (by backpropagating) where it wants to go, we introduce scaling and shifting. The model will learn itself the direction it wants to go, by back propagating because these scaling and shifting are differentiable.

```
# BatchNorm parameters
bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

hpreact = bngain*(hpreact - hpreact.mean(0, keepdim = True)) / hpreact.std(0, keepdim =True) + bnbias
```

by introducing batchNormalization, we are making our pre-activation of one input depend on all the other input, this is because we are subtracting the mean from it and this mean is the depended on all the other inputs.

Now that we have introduced BatchNormalization, inorder to do inference, we would need the mean and std of the whole dataset, which we can keep track of in running manner, and it is used while doing inference.

```
# same optimization as last time
max_steps = 200000
batch_size = 32
lossi = []

for i in range(max_steps):

    # minibatch construct
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X,Y

    # forward pass
    emb = C[Xb] # embed the characters into vectors
    embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
    # Linear layer
    hpreact = (embcat @ W1 + b1)  # hidden layer pre-activation
    bnmeani = hpreact.mean(0, keepdim = True)
    bnstdi =  hpreact.std(0, keepdim =True)
    hpreact = bngain*((hpreact - bnmeani) / bnstdi) + bnbias

## --------------------- keeping track of whole mean and std ------------
    with torch.no_grad():
        bnmean_running = 0.999*bnmean_running + 0.001*bnmeani
        bnstd_running = 0.999*bnstd_running + 0.001*bnstdi
# ------------------------------------------------------------------------
    # Non-linearity
    h = torch.tanh(hpreact) # hidden layer
    logits = h @ W2 + b2 # output layer
    loss = F.cross_entropy(logits, Yb) # loss function

    # backward pass
    for p in parameters:
        p.grad = None

    hpreact.retain_grad()
    logits.retain_grad()
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 10000 == 0: # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())


```
