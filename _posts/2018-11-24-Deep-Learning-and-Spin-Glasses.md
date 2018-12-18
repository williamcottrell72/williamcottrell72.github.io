---
layout: post
title: Machine Learning and Spin Glasses
description: What is a glass and what does this have to do with learning?
---

Today I would like to explore the relationship between machine learning and spin glasses.  At first glance, these might seem like strange things to connect.  However, viewed as statistical systems, they actually have a lot in common.  This is due to the key fact that both machine learning models and glasses have an incredibly large number of "nearby" states (exponentially many in some sense).  This is actually very important since it means that these systems can choose from one of many "nearby" states in response to external stimuli.  For learning models, this is good since the system can adapt to the precise training data it was provided so as to reproduce it later.  Another way of saying this is that the large number of states give machine learning models a lot of memory.  We normally don't care about the "memory" properties of ordinary glass, but it is actually quite remarkable.  The glass windows in your home are in some sense held up by this memory, unlike wood or steal which is structurally stable for more mundane reasons.  

The incredibly large memory of these systems can be confounding from an analytic perspective.  We are used to thinking about dynamical systems in terms of energy, i.e., an energy landscape, etc.  Ordinary thermodynamics then tells us that the probability of finding the system in a state 'i' is given by:

<!-- <div lang="latex">
\begingroup -->
$$
P_{i} \sim e^{-\beta E_{i}}
$$
<!-- \endgroup
</div> -->

Thus, higher energy means less probable. However, this is not a very good picture when it comes to glasses.  The issue is that the sheer number of excited states means that it is generically more likely to be in one of these.  Fortunately, many of these excited states are basically idential for all practical purposes.  Similar statements apply to machine learning.  For instance, when doing SGD, you won't necessarily find the true global minimum, in fact, you could find any one of "nearby" similar minimum that all behave nearly the same way.   This is good, otherwise machine learning would not work.

<p>
How do we see this connection more formally?  Well, first we should say what we mean by a glass.  I won't attempt a general definition, here I will just mean a system with quenched disorder.  To be a little more precise, I want to talk about systems where there is some random variable \( \mathbf{J}\) such that the potential for the dynamical degrees of freedom, \( \sigma \) depend on \( \mathbf{J}\).  That's it!  This is said to be disordered since \( \mathbf{J}\) is random and \(\textit{quenched}\) since \( \mathbf{J}\) is frozen and fixed once for all.  Actual glasses in your apartment are not like this at all - there is no fixed \( \mathbf{J}\) to speak of.  However, each molecule locally behaves as if it is in a random potential and so this is nevertheless a good description in some regimes.
</p>

<p id="return_xnote">
For machine learning models, \( \mathbf{J}\) is just the training data, \(\mathbf{X}\) and \(\sigma\) is just the set of parameters to be tuned, e.g., the weights, \(\mathbf{W}\) of a neural network <a href="#xnote"><sup>[1]</sup></a>.  In other words, the training data is 1) random (i.e., randomly drawn from some distribution) and 2) it determines the potential for \(\mathbf{W}\).  Thus, techniques from glasses might be useful.
</p>

<p>So, how do we describe this system mathematically?  As usual for stat-mechy problems, we want to define a free energy which includes both the effects of energy and entropy.  Obviously, when we talk about entropy we are referring to some kind of coarse graining and how we do that will vary from case to case.  For the sake of specificity, I am thinking now of a feed-forward neural network with d-dimensional layers and approximate translation symmetry.  (I'm assuming that the layers are large enough to ignore edge effects.)  Let's label the weights by \(W^{n}_{ij}\), which is the weight between the \( i'th\) node of the \((n-1)'th \) layer and the \( j'{th}\) node of the \( n'th\) layer.  I'll start counting at layer zero, the input layer. </p>

<p> With this notation in place, I'll coarse grain by averaging \(W^{n}_{ij}\) over \(i,j\) for each \(n\).  If this is a convolutional network and the connections are restricted by a kernel, then we can just take the average within that kernel.  (I.e, for each \(j\), take the average over the \(i\) for which \(W^{n}_{ij}\) is not restricted to be zero.)  I'll denote this averaging by with the subscript \(L\) for 'Layer', i.e.,  \( \langle W_{ij}^{n}\rangle_{L} \rightarrow \vec{W}\). After coarse graining, we end up with a state defined simply by \(W^{n}\).  I'll sometimes write this as \(\vec{W}\).  Furthermore, if the network doesn't change very much at each step, we could even go to the continuum limit and describe the state as a function \(W(x)\) on a continuous interval.  </p>

Now we write the free energy:

$$
F(\vec{W},\mathbf{X})=-\frac{1}{\beta} \ln \int \left( \prod dW_{ij}^{n} e^{-\beta H(\mathbf{W},\mathbf{X})} \times   \delta \left(\langle W_{ij}^{n}\rangle_{L} - \vec{W}\right) \right)
$$

<p> Here, \( e^{-\beta H(\mathbf{W},\mathbf{X})}\) denotes the probability of finding the network in the (microscopic or, fine-grained) configuration \(\mathbf{W}\) when the training data is \(\mathbf{X}\).   This could just be viewed as the definition of \(H\), though it is a very reasonable parameterization, especially if we are using a procedure like simulated annealing.  More generally, it is convenient to work with the log of probability (i.e., \(H\)) rather than the probability itself simply because probabilities are built multiplicatively and are thus exponentially suppressed for generic configurations.  </p>

<p>Now, in principle, minimizing the free energy will give us the configuration of \(\vec{W}\) for a given \(\mathbf{X}\).  However, there was nothing special about the particular \(\mathbf{X}\) we chose.  In particular, we will often be drawing \(\mathbf{X}\) from a larger sample and then do mini-batching to train our model.  Thus, what we really want is the average of \(F\) over \(\mathbf{X}\).  This is accomplished via:</p>

$$
\overline{F}(\vec{W})=[[ F(\vec{W},\mathbf{X}) ]] \equiv \int d\mathbf{X} P(\mathbf{X}) F(\vec{W},\mathbf{X})
$$

Taking the integral of a log is very nasty, which is why the replica trick was invented (I think.)  In any case, one can use the following mathematical identity:

$$
\ln x = \lim_{n\rightarrow 0} \frac{x^{n}-1}{n}
$$

<p>So now, the average of \(ln x\) can be converted to an average of \( x^{n}\).  Taking the average over both sides of the above equation, we can even write  </p>

$$
\overline{\ln x} =\lim_{n\rightarrow 0} \frac{1}{n} \ln \overline{x^{n}}
$$

<p>The tricky part is to now treat \(n\) as an integer, compute the result for all \(n\), and then extrapolate \(n\rightarrow 0\), hoping that the formula for integral \(n\) generalizes in the correct way.  Obviously, this procedure is a bit sketchy but it seems to work.</p>  

<p> Applying this procedure means we need to compute the average (over \(\mathbf{X}\)) of the thing appearing in the log.  Let me introduce the notation \(\prod dW_{ij}^{n} \rightarrow \mathcal{D}\mathbf{W}\) for convenience.  What we need to compute is therefore:
</p>

$$
\int d\mathbf{X} P(\mathbf{X}) \left(\int \mathcal{D}\mathbf{W} e^{-\beta H} \delta \left(\langle \mathbf{W}\rangle_{L}-\vec{W}\right)\right)^{n}
\\
=\int d\mathbf{X} P(\mathbf{X})\prod_{i=1}^{n} \mathcal{D}\mathbf{W}^{i}\delta\left(\langle\mathbf{W}\rangle_{L}^{i}-\vec{W}^{i}\right) e^{-\beta \sum_{i}^{n} H(\mathbf{W}^{i},\mathbf{X})}
$$

<p>The extra copies of \(\mathbf{W}\) we've generated this way are known as "replicas" and therefore this is known as the 'replica trick'.</p>

<p>Ok, so where do we go from here?  Usually, for models of glasses we would now try and 1) integrate out the disorder, \(\mathbf{X}\) and 2) write the result in such a way that the large size limit becomes useful.  For us, this is complicated because our \(H\) is complicated.  In certain situations, like when we only have ReLU activation functions, the situations simplifies and we can start to do some analytic estimates.  For more general activation functions we will apparently need new techniques to proceed.  This will be discussed in a subsequent post. </p>

<p> Before saying goodbye, I'd like to point out another sense in which a 'replica' appears in this context.  Often, the data will consist of a batch of samples drawn from a fixed distribution and the loss function \(H\) is just a sum over these samples.  Suppose there are \(\tilde{n}\) samples in a batch.  Then the probability of finding the network in the fine-grained state \(\mathbf{W}\) can be written as </p>

$$
P(\mathbf{W}) = \int \mathcal{D}\mathbf{X}  e^{-\beta H}
\\
= \int \prod_{i}^{\tilde{n}} d\vec{x}^{i} e^{-\beta\sum_{i} h(\mathbf{W},\vec{x}^{i})}
$$

<p>where \(h\) is some loss function like a squared error.  Now, it is the training data that is playing the role of a replica!  However, unlike the previous situation where this was just a trick to compute the average of a log and we wanted to take \(n\rightarrow 0\), here, \(\tilde{n}\) is typically a fixed number throughout the computation. This may be useful later...</p>

<p id="xnote"><a href="#return_xnote">[1]</a> I'm using the notation \( \mathbf{X}\) to denote the features as well as the labels.</p>
<!-- $$
 \text{Score}_{u a} = N_{a} A_{u}+b_{u}
$$ -->
