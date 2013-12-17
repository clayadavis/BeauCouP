BeauCouP
========
Analytical Bayesian change point detection with the BCMIX approximation.
--------

BeauCouP aims to be a reference implementation of Lai and Xing's analytical
framework for Bayesian detection of multiple change points \[[1](#1)\]. Initial
effort is focused on the Poisson case, where the input is regular time series.
This effort seeks first to be readable and understandable, and second to be
high-performance, with priority in that order. This algorithm should be
amenable to many optimizations including parallelism, GPU acceleration, and
compiling, but never at the expense of clarity.
