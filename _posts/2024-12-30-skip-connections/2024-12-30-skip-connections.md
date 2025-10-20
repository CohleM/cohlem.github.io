---
layout: blog
author: cohlem
tags: [optimization]
---

Skip connections are simply skipping the layers by adding the identity of input to it's output as shown in the figure below.

![fig1](/assets/images/2024-12-30-skip-connections/fig1.png)

Why add the identity of input x to the output ?
![fig2](/assets/images/2024-12-30-skip-connections/fig2.png)

We calculate the gradients of parameters using chain rule, as shown in figure above. For deeper layers the gradient start to become close to 0 and the gradient stops propagating, which is a vanishing gradient problem in a deep neural networks.

When we add the identity of input to it's output like this
hl+1​=F(hl​)+hl​

and when we backpropagate, we get this type of equation.
![fig3](/assets/images/2024-12-30-skip-connections/fig3.png)

So even when d(F(h1))/dh1 becomes close to 0 the previous gradient dL/dhl+1 is propagated as it is multiplied with 1 which is the outcome of adding the identity of input to its output.

Eventually, skip connections stop vanishing gradient problem, and the other thing they help with is that, when going to a residual block(could be attention block or feedforward block) neural network may loose previous information when going through that transformation, so adding identity of input to its output will take into consideration the new learned features + the previous features.
