BeauCouP
========
Analytical Bayesian change point detection with the BCMIX approximation.
--------

BeauCouP aims to be a reference implementation of Lai and Xing's analytical
framework for Bayesian detection of multiple change points \[[1](#xingetal),
[2](#laixing)\]. Initial
effort is focused on the Poisson case, where the input is regular time series.
This effort seeks first to be readable and understandable, and second to be
high-performance, with priority in that order. This algorithm should be
amenable to many optimizations including parallelism, GPU acceleration, and
compiling, but never at the expense of clarity.

<p id="xingetal">[1] Xing, H., Y. Mo, W. Liao, and M. Q. Zhang. 2012. Genome-Wide
Localization of Protein-DNA Binding and Histone Modification by a Bayesian
Change-Point Method with ChIP-seq Data. PLoS Comput Biol <b>8</b>:e1002613.</p>

<p id="leixing">[2] Lai, T. L., and Xing, H. A Simple Bayesian Approach to Multiple
Change-Points. 2011. Statistica Sinica <b>21</b>:539-569.</p>
