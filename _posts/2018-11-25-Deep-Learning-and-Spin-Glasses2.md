---
layout: post
title: Machine Learning and Spin Glasses 2
description: What is a glass and what does this have to do with learning?
---

<script type="text/javascript" async
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML">
</script>

In our last post we were discussing the relationship between machine learning and spin glasses.  Now, I'd like to focus more on deep learning.  

The statistical mechanics of deep learning models is very difficult since the loss function is a highly non-linear function of many variables.  For a general deep learning model we cannot make any precise statements since there are just too many architectures to consider.   Thus, we'll make some simplifying assumption to get started.  

<p id="return_note1">First and foremost, I want to assume that the neural network is built by doing the 'same thing' when we go from one layer to the next.  More precisely, I'm going to assume that there are \(d\)-dimensional layers, and that we are performing an equivalent operation in going from the \(n'th\) layer to the \( (n+1)'th\) layer for all \(n\).  I'll schematically denote this operation as \(\mathbf{X}\rightarrow \sigma(\mathbf{W}\circ \mathbf{X})\).  This means that we multiply the input by some weights and then apply some activation function <a href="#note1"><sup>[1]</sup></a>.  More generally, the neural network could be built by repeating the same sequence of layers, including dropout or a mix of different activation functions.</p>

<p> This assumption of self-similarity is critical for making any progress. Since the operation described above is generally also a scale transformation (e.g., in convolutional neural nets) the self-similarity at different scales allows us to use ideas and techniques from renormalization group theory.  Moreover, for very deep networks after many similar transformations, we would expect that the statistics of the weights reaches a fixed point.  We've learned from quantum field theory that classifying theories is tantamount to classifying fixed points plus the deviations on top of these.  Perhaps these ideas can be applied here too? </p>

<p> Let's see if any of these thoughts are actually useful for evaluating some of the nasty integrals we encountered in our last post.  We encountered things like:</p>  


$$
P_{cg}(\vec{W})\sim \int \mathcal{D} \mathbf{W}\int \mathcal{D}\mathbf{X} P_{0}(\mathbf{X}) e^{-\beta H(\mathbf{W},\mathbf{X})}\delta\left(\langle \mathbf{W}\rangle_{L} - \vec{W}\right)
$$

<p>Here, \(P_{cg}(\vec{W})\) is the probability that the neural network assumes the coarse-grained value \(\vec{W}\) and \(P_{0}\) is the prior distribution of data on the input layer, i.e,. layer 0.  Note that this is an annealed quantity (we are computing the average of \(e^{-\beta H}\)) rather than a quenched quantity (computing the average of a log).  However, this is the sort of integral we encounter after applying the replica trick so we may as well practice here.  In any case this integral is intrinsically useful and will accurately describe certain phases of the system.  </p>

<p> In order to proceed, we need to use the following fundamental property of the Hamiltonian:</p>

$$
H_{h}(\mathbf{W},\mathbf{X}) = H_{h-k}(\mathbf{W}_{>k},\mathbf{X}_{k})
$$
<p> where \(h\) is the number of layers and the notation \(\mathbf{W}_{>k}\) refers to the weights at layers above \(k\) and \(\mathbf{X}_{k}\) is the effective input at layer \(k\) and is obtained by applying the \(\sigma(\mathbf{W}\circ)\) mapping \(k\) times.  The following Hamiltonian, for instance, has this property</p>

$$
H_{h}(\mathbf{W},\mathbf{X})=\frac{1}{|\mathbf{X}|}\sum_{1\le i\le |\mathbf{X}|} \left(x^{i}_{0}-\left(\sigma(\mathbf{W} \circ)\right)^{h}\vec{x}^{i}\right)^{2}
$$

<p> where I've broken up the training data as \(X=(x_{0},\vec{x})\) where \(x_{0}\) plays the role of the label. </p>

Let's look more closely at the feed-forward network.  

$$
\left(\sigma(\mathbf{W} \circ)\right)^{H}\vec{x}^{i} =\sigma\left(\mathbf{W}^{H}\vec{\sigma}(\mathbf{W}^{H-1} \vec{\sigma}(...\mathbf{W}^{0}\vec{x}^{i}))\right)
$$

<p>Now, the input distribution of data is like \(P_{0}(X)\) but, from the perspective of the second layer, the output of layer one is just some new distribution of inputs.  So, the game is to figure out what the effective distribution of inputs is for the second, third and fourth layer and so on.  What we are actually after is some kind of differential equation that tells us how the effective distribution on \(X\) evolves as we move down the network.  </p>

<p>Let's consider the effect in going from layer \(n\) to layer \(n+1\).  First, suppose that the effective distribution at the \(n'th\) layer is \(P_{n}(\mathbf{X}_{n})\).  Let's also include a regulator term \(R(\mathbf{W})=\sum_{n,i,j} \alpha_{n}(W^{n}_{ij})^{2}\) at this stage for reasons that will soon be clear.  We can modify our previous integral as follows: </p>


$$
P_{cg}\left(\vec{W}\right) \sim \int \mathcal{D} \mathbf{W}_{>n+1} \mathcal{D}\mathbf{W}_{n+1}\int \mathcal{D}\mathbf{X}_{n}\int\mathcal{D}\mathbf{X}_{n+1}\delta\left(\mathbf{X}_{n+1}-\sigma(\mathbf{W}_{n+1}\circ\mathbf{X}_{n})\right)\delta\left(\langle\mathbf{W}_{>n+1}\rangle_{L}-\vec{W}_{>n+1}\right) \\
\times P_{n}(\mathbf{X}_{n})e^{-\beta H_{h-n-1}(\mathbf{X}_{n+1})-R(W)}
$$

<p>We can now recognize the the effective change in \(P\) as: </p>

$$
\boxed{
P_{n+1}\left(\mathbf{X}_{n+1}\right)=\int\mathcal{D} \mathbf{W}_{n+1}\mathcal{D}\mathbf{X}_{n}\delta\left(\mathbf{X}_{n+1}-\sigma(\mathbf{W}_{n+1}\circ\mathbf{X}_{n})\right)\times e^{-\alpha_{n+1}\sum_{i,j}(W_{ij}^{n+1})^{2}} P_{n}\left(\mathbf{X}_{n}\right)}
$$


<p> This is our fundamental evolution equation which we now need to evaluate in various limits. I'll refer to this as the "RG equation" or "RG flow", borrowing from the physics "renormalization group" terminology.  So far, everything has been exact, but we haven't really done anything but some re-writing.  Now we actually need to do some math! Before we continue, let's be a little bit more precise about what is meant by \(\mathbf{W}\circ \mathbf{X}\).  Roughly, this is \(W_{ij} X_{j}\), but that is not quite right.  In many cases, we want to only connect nodes that satisfy some additional condition.  If we imagine that the neural network is a stack of d-dimensional sheets and refer to the directions tangential to the sheets as "horizontal" (and from one sheet to the next as "vertical") then, for convolutional networks, we only want to connect nodes that are within some distance of one-another in the horizontal directions.  This can be accomplished by introducing some fixed matrix \(\mathbf{K}\) (or \(\mathbf{K}^{n}\) if we care to specify the layer) whose elements are either 0 or 1.  Then, what we really mean is:</p>

$$
\mathbf{W}\circ \mathbf{X} = \sum_{j}K_{ij}W_{ij}X_{j}
$$

<p id="return_note2">We will introduce two parameters to characterize \(\mathbf{K}\).  The first is the "kernel size", which characterizes how many nodes contribute for a single output node.  Formally, we will define this parameter as \(k^{d} = \sum_{j} K_{ij}\), where we are envisioning a square kernel and \(d\) is the dimension of the network.  The other parameter is the "stride" <a href="#note2"><sup>[2]</sup></a> which characterizes the overlap between neighboring kernels.  Again, thinking of a square kernel, we'll define this as \(k^{d-1}(k-s)=\sum_{j} K_{ij}K_{i'j}\), where \(i\) and \(i'\) are neighboring (i.e., the coordinates are equal in all dimensions but one, where they differ by one) nodes in the higher layer.</p>

<p>There are some potentially interesting limits to consider.  Defining \(\gamma\) via \(s=\gamma k\) </p>

<ol>
  <li> \(s,k\rightarrow \infty\),       \(\qquad\qquad\gamma \in (0,1)\)</li>
  <li>\(k\rightarrow \infty\), \(\,\,\,\,\,\qquad\qquad \gamma =0\)</li>
  <li> \(s,k\rightarrow \infty\) \(\qquad\qquad\gamma = 1\) </li>
</ol>

<p> These limits are described in terms of fixing \(\gamma\); we could also thinking about fixing \(s,k\) instead but I'll push that off for now.  Note that the \(\gamma=0\) case is particularly interesting since it allows for the situation where moving from one layer to the next approaches a continuous dilatation operator.  In this limit, we might expect to recover limits akin to conformal field theories.  Since we know that this is very complicated, let's first consider the opposite limit where \(s=k\), i.e., \(\gamma=1\).  In this case the kernels are completely non-overlapping and some simplifications should occur.</p>

<div><font size=14pt>Non-Overlapping Kernels: s=k</font></div>

<p> In this case, the whole integral completely decomposes into separate cells, each of size \(k\).  For each value of \(\mathbf{X}_{n}\), we may pick a basis for \(\mathbf{W}_{n+1}\) consisting of components parallel and perpendicular to the \(\mathbf{X}_{n}\) in each kernel. Let's represent \(\mathbf{W}_{n+1}\) as</p>

$$
\mathbf{W}_{n+1} = w_{\parallel}^{i}\mathbf{W}^{i}_{\parallel}+w_{\perp}^{k}\mathbf{W}^{k}_{\perp}
$$
<p> where the basis vectors are orthonormal and \(\mathbf{W}^{i}_{\parallel}\cdot \mathbf{X}= w_{\parallel}^{i}\times\left(\sum_{j}K_{ij}X_{j}^{2}\right)^{1/2} \equiv w_{\parallel}^{i}\times x^{i} \).  Now let's get a little more concrete.  Suppose the layer we are integrating out is a (hyper-)cube of side \(N_{n}\).  Then there are \(N_{n}^{d}\) degrees of freedom in the \(\mathbf{W}_{n+1}\) variables, of which \( (N_{n}/k)^{d}\) are contained in the parallel modes.  The orthogonal modes simply factor out and the remaining integral may be factorized.  To make the notation a little clearer we will define \(X_{n,i}^{j}\) to be the set of \(X_{n}^{j}\) for which \(K_{ij}=1\). This allows us to write </p>

$$
P_{n+1}\left(\mathbf{X}_{n+1}\right)=\left(\frac{\pi}{\alpha_{n+1}}\right)^{\left(\frac{N_{n}}{2}\right)^{d}-\left(\frac{N_{n}}{2k}\right)^{d}} \prod_{i=1}^{\left(\frac{N_{n}}{k}\right)^{d}}\int \mathcal{D}\mathbf{X}_{n,i} dw_{i} \prod_{j}p(X_{n,i}^{j}) e^{-\alpha_{n+1} w_{i}^{2}} \delta\left(X_{n+1}^{i}-\sigma(|\mathbf{X}|\times w)\right)
$$

<p> Here, \(p(\mathbf{X})\) refers to the ansatz for the distribution within a given cell and this function should (ideally) be the same between layers in the limit of infinite depth.  I'll assume that this \(p\) does indeed approach a fixed function at infinite depth in the case \(s=k\), so that the full probability density is \(P_{n} = \prod_{j} p(X_{n}^{j}) = \prod_{i}\left(\prod_{j} p(X_{n,j}^{i})\right)\).  Note that the factorization property is basically guaranteed since even if the \(n-th\) layer is not factorizable the \((n+1)th\) layer will be, and then every layer after that.  Applying the large \(k\) limit and doing the \(w\) integral we get</p>

$$
P_{n+1}\left(\mathbf{X}_{n+1}\right)=\left(\frac{\pi}{\alpha_{n+1}}\right)^{\left(\frac{N_{n}}{2}\right)^{d}}\prod_{i}^{\left(\frac{N_{n}}{k}\right)^{d}}\int \mathcal{D} \mathbf{X}_{n,i}\prod_{j} p(X_{n,i}^{j})\frac{e^{-\alpha_{n+1} w_{i}^{2}}}{|\mathbf{X}|\sigma'(\sigma^{-1}(X^{i}_{n+1}))}
$$

<p> where \(w_{i}\) is defined by \( \sigma(|\mathbf{X}|w)= X^{i}_{n+1}\). Using the factorization condition also on the LHS of the equation above, we reach:</p>

$$
p(X_{n+1}^{i}) = \left(\frac{\pi}{\alpha_{n+1}}\right)^{\left(\frac{k}{2}\right)^{d}} \prod_{j}\int dX^{j} p(X^{j})\times \frac{e^{-\alpha_{n+1} w_{i}^{2}}}{|\mathbf{X}|\sigma'(\sigma^{-1}(X^{i}_{n+1}))}
$$

<p> To make more progress it would be nice to have a small parameter to work with.  We have declared \(1/k=1/s\) to be small, but it is not clear how it helps at this stage.  The next thing to consider is \(\alpha_{n+1}\), which we coule take to be either very large (low T) or very small (high T).  Roughly speaking, \(\alpha \sim X^{2}\) </p>

<p id="note1"><a href="#return_note1">[1]</a> I trust the interested reader can figure out where all the indices go.</p>

<p id="note2"><a href="#return_note2">[2]</a> Borrowing terminology from sklearn. </p>
