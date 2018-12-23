---
layout: post
title: Machine Learning and Spin Glasses 3
description: Special Limits of Neural Networks
---


<script type="text/javascript" async
src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.5/latest.js?config=TeX-MML-AM_CHTML">
</script>

<p> Now let's look at another limit of neural networks - the limit where \(\gamma=0\).  In this case, neighboring cells are nearly overlapping and the information is compressed by an infinitesimal amount in the large \(k\) limit.  We thus have a continuous "scaling" transformation.  If we replace the indices within each layer via a continuous vector \(\vec{x}\), then the operation of moving up one layer has the effect of sending \(\vec{x} \rightarrow \left(1-\delta\right)\vec{x}\), with the precise value of \(\delta\) depending on details about how we take the continuum limit and the value of the stride, \(s\).</p>

Meanwhile, under this transformation, the probability distribution, \(P(\mathbf{X}_{n})\) may be thought of as a functional on the set of functions, \(X(\vec{x})\), which now satisfies
