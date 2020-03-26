# Regimes_Public
The code used to compute the atmoshperic circulation regimes for the Euro-Atlantic sector. The clustering method used is k-means clustering, with and without a persistence constraint. When the persistence constraint is included Gurobi software is needed for the linear optimization step (without the code does not run, a free academic license is available). Also included are the computation of principal components and applying a lowpass-filter to the data.

The regimes_tests.file is the main file for running the different algorithms. It requires as input a number of clusters (k) and value of the persistence constraint (C). The other func_[...] files contain the functions for e.g. computation of principal components or persistent k-means clustering. In addition there needs to be a folder present named tests in which the output is stored.

I have tried to make the code insightfull, but if some aspects are not clear please let me know. In addition I would like to note that the way it's written the output requires quite some storage. Furthermore, doing 500 tests takes some time and I would not recommend it for persistent k-means, since the additional step has a long computation time.

The data used is ERA-Interim and can be donwloaded from the ECMWF website.
