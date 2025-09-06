---
title: "Scaling Laws for Uncertainty in Deep Learning"
collection: publications
category: under_review
permalink: /publication/2025-11-06-scaling-laws
excerpt: 'This paper is about a famous math equation, $$E=mc^2$$'
date: 2025-11-06
venue: 'arxiv'
paperurl: 'https://arxiv.org/abs/2506.09648'
citation: 'Mattia Rosso, Simone Rossi, Giulio Franzese, Markus Heinonen, & Maurizio Filippone. (2025). Scaling Laws for Uncertainty in Deep Learning.'
---

Deep learning has recently revealed the existence of scaling laws, demonstrating that model performance follows predictable trends based on dataset and model sizes. Inspired by these findings and fascinating phenomena emerging in the over-parameterized regime, we examine a parallel direction: do similar scaling laws govern predictive uncertainties in deep learning? In identifiable parametric models, such scaling laws can be derived in a straightforward manner by treating model parameters in a Bayesian way. In this case, for example, we obtain O(1/N) contraction rates for epistemic uncertainty with respect to the number of data N. However, in over-parameterized models, these guarantees do not hold, leading to largely unexplored behaviors. In this work, we empirically show the existence of scaling laws associated with various measures of predictive uncertainty with respect to dataset and model sizes. Through experiments on vision and language tasks, we observe such scaling laws for in- and out-of-distribution predictive uncertainty estimated through popular approximate Bayesian inference and ensemble methods. Besides the elegance of scaling laws and the practical utility of extrapolating uncertainties to larger data or models, this work provides strong evidence to dispel recurring skepticism against Bayesian approaches: "In many applications of deep learning we have so much data available: what do we need Bayes for?". Our findings show that "so much data" is typically not enough to make epistemic uncertainty negligible.