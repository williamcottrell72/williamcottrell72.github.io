---
layout: post
title: Machine Learning and Spin Glasses 2
description: What is a glass and what does this have to do with learning?
---

In our last post we were discussing the relationship between machine learning and spin glasses.  Now, I'd like to focus more on deep learning.  

The statistical mechanics of deep learning models is very difficult since the loss function is a highly non-linear function of many variables.  For a general deep learning model we cannot make any precise statements since there are just too many architectures to consider.   Thus, we'll make some simplifying assumption to get started.  

<p id="return_note1">First and foremost, I want to assume that the neural network is built by doing the 'same thing' when we go from one layer to the next.  More precisely, I'm going to assume that there are \(d\)-dimensional layers, and that we are performing an equivalent operation in going from the \(n'th\) layer to the \( (n+1)'th\) layer for all \(n\).  I'll schematically denote this operation as \(\mathbf{X}\rightarrow \sigma(\mathbf{W}\circ \mathbf{X})\).  This means that we multiply the input by some weights and then apply some activation function <a href="#note1"><sup>[1]</sup></a>.  More generally, the neural network could be built by repeating the same sequence of layers, including dropout or a mix of different activation functions.</p>

<p> This assumption of self-similarity is critical for making any progress. Since the operation described above is generally also a scale transformation (e.g., in convolutional neural nets) the self-similarity at different scales allows us to use ideas and techniques from renormalization group theory.  Moreover, for very deep networks after many similar transformations, we would expect that the statistics of the weights reaches a fixed point.  We've learned from quantum field theory that classifying theories is tantamount to classifying fixed points plus the deviations on top of these.  Perhaps these ideas can be applied here too? </p>

<p> Let's see if any of these thoughts are actually useful for evaluating some of the nasty integrals we encountered in our last post.  We encountered things like:</p>  


$$
P_{cg}(\vec{W})\sim \int \mathcal{D} \mathbf{W}\int \mathcal{D}\mathbf{X} P_{0}(\mathbf{X}) e^{-\beta H(\mathbf{W},\mathbf{X})}\delta\left(\langle \mathbf{W}\rangle_{L} - \vec{W}\right)
$$

<p>Here, \(P_{cg}(\vec{W})\) is the probability that the neural network assumes the coarse-grained value \(\vec{W}\) and \(P_{0}\) is the prior distribution of data on the input layer, i.e,. layer 0.  Note that this is an annealed quantity (we are computing the average of \(e^{-\beta H}\)) rather than a quenched quantity (computing the average of a log).  However, this is the sort of integral we encounter after applying the replica trick so we may as well practice here.  In any case this integral is intrinsically useful and will accurately describe certain phases of the system.  </p>

<p> In order to proceed, let's write the Hamiltonian schematically as:</p>

$$
H_{h}(\mathbf{W},\mathbf{X})=\frac{1}{|\mathbf{X}|}\sum_{1\le i\le |\mathbf{X}|} \left(x^{i}_{0}-\left(\sigma(\mathbf{W} \circ)\right)^{h}\vec{x}^{i}\right)^{2}
$$

<p> where \(h\) is the number of layers and I've broken up the training data as \(X=(x_{0},\vec{x})\) where \(x_{0}\) plays the role of the label.  I've used a very schematic notation to describe the action of a feed forward network and I'm assuming a MSE loss.  I'm just writing this in order to have a more concrete story, the main ideas I'd like to discuss are tangential to the actual choice of loss function.  </p>

Let's look more closely at the feed-forward network.  

$$
\left(\sigma(\mathbf{W} \circ)\right)^{H}\vec{x}^{i} =\sigma\left(\mathbf{W}^{H}\vec{\sigma}(\mathbf{W}^{H-1} \vec{\sigma}(...\mathbf{W}^{0}\vec{x}^{i}))\right)
$$

<p>Now, the input distribution of data is like \(P_{0}(X)\) but, from the perspective of the second layer, the output of layer one is just some new distribution of inputs.  So, the game is to figure out what the effective distribution of inputs is for the second, third and fourth layer and so on.  What we are actually after is some kind of differential equation that tells us how the effective distribution on \(X\) evolves as we move down the network.  </p>

<p>Let's consider the effect in going from layer \(n\) to layer \(n+1\).  First, suppose that the effective distribution at the \(n'th\) layer is \(P_{n}(\mathbf{X}_{n})\).  Let's also include a regulator term \(R(\mathbf{W})=\sum_{n,i,j} \alpha_{n}(W^{n}_{ij})^{2}\) at this stage for reasons that will soon be clear.  We can modify our previous integral as follows: </p>

$$
P_{cg}\left(\vec{W}\right) &\sim& \int \mathcal{D} \mathbf{W}_{>n+1}
$$



<p> ;lkajsdl;fkjas;dlfkjas;ldkfjals;dkjfal;sdkjfal;skdfj</p>

$$
P_{cg}\left(\vec{W}\right) &\sim& \int \mathcal{D} \mathbf{W}_{>n+1} \mathcal{D}\mathbf{W}_{n+1}\int \mathcal{D}\mathbf{X}_{n}\int\mathcal{D}\mathbf{X}_{n+1}\delta\left(\mathbf{X}_{n+1}-\sigma(\mathbf{W}_{n+1}\circ)\mathbf{X}_{n}\right)\delta\left(\langle\mathbf{W}_{>n+1}\rangle_{L}-\vec{W}_{>n+1}\right) \\
&\times& P_{n}(\mathbf{X}_{n})e^{-\beta H_{h-n-1}(\mathbf{X}_{n+1})-R(W)}
$$

<p>We can now recognize the the effective change in \(P\) as: </p>

$$
P_{n+1}(\mathbf{X}_{n+1}) = \int\mathcal{D} \mathbf{W}_{n+1}\mathcal{D}\mathbf{X}_{n}\delta\left(\mathbf{X}_{n+1}-\sigma(\mathbf{W}_{n+1}\circ)X_{n}\right)e^{-\alpha_{n+1}\sum_{i,j}(\mathbf{W}_{n+1}_{ij})^{2}}P_{n}(\mathbf{X}_{n})
$$

<p> Now we actually need to do some math! </p>

$$
P^{n+1}_{cg}\left(\mathbf{W}^{i}_{i>n} \right) \equiv \int d\mathbf{W}^{n} P^{n}_{cg}\left(\mathbf{W}_{i\ge n}\right) \\
= \int d\mathbf{W}d
$$


<p id="note1"><a href="#return_note1">[1]</a> I trust the interested reader can figure out where all the indices go.</p>
