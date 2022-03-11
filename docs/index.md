# Ramba Documentation

Ramba is a Python project that provides a fast, distributed, NumPy-like array API using compiled Numba functions 
and a Ray or MPI-based distributed backend.  It also provides a way to easily integrate Numba-compiled remote
functions and remote Actor methods in Ray.  

The main use case for Ramba is as a fast, drop-in replacement for NumPy.  Although NumPy typically uses C
libraries to implement array functions, it is still largely single threaded, and typically does not make
use of multiple cores for most functions, and definitely cannot make use of multiple nodes in a cluster. 

Ramba lets NumPy programs make use of multiple cores and multiple nodes with little to no code changes.

For a quick start guide, installation and usage notes, please see Ramba's
[main page](https://github.com/Python-for-HPC/ramba).
Ramba is working towards full support for the [NumPy](https://numpy.org/doc/) API and the current status
of that effort is documented in the NumPy Compatibility table also on Ramba's
[main page](https://github.com/Python-for-HPC/ramba).

## NumPy API in a Distributed Setting

Since NumPy was designed for non-distributed environments, some of functions in the NumPy API
may have poor performance when applied in a distributed environment.
This poor performance is typically caused by communication induced by these operations.
For example, the NumPy function *reshape* could cause large-scale movement of data between nodes
of the cluster to move from one partitioning scheme for the original array shape to a different
partitioning scheme for the new array shape.  Likewise, *advanced indexing* where an array of
indices is used to index another array is equivalent to a cross-node gather operation.  Note
that in some cases, certain variants of a function in the NumPy-API may have cheap distributed
implementations whereas others might not.  For example, *reshape* that adds or removes dimensions
of size 1 or swaps the positions of dimensions can be done cheaply whereas most other variants of
reshape are potentially very time consuming.

## Ramba's Distribution Friendly APIs

To facilitate writing distribution-friendly, performant code, Ramba provides a variety of APIs in
addition to what NumPy provides.  These fall into two categories: algorithmic skeletons and
groupby operations.

### Algorithmic Skeletons

Algorithmic skeletons take one or more functions that can operate on remote data and encode
certain communication patterns for classes of algorithms.

#### *smap*, *smap_index*

def smap(func, * args)
def smap_index(func, * args)

These skeletons each take a function to be applied to each element of a distributed array to
produce a new distributed array.
The first argument after this function argument must be a Ramba distributed array but
additional arguments are allowed and these may be of any type including other Ramba
distributed arrays.  However, in this latter case, all the Ramba distributed arrays must be
of the same shape and have the same distribution.
Conceptually, Ramba calls the function once for each element in the distributed arrays.
This function will be passed those individual elements and not the array as a whole whereas
all other argument types are passed to the function unmodified.
In some cases, the point in the index space that is being computed is necessary for the
computation itself.  For this purpose, *smap_index* first passes the point in the index space
to the function followed by all the other arguments as in *smap*.

#### *sreduce*, *sreduce_index*

def sreduce(func, reducer, identity, * args)
def sreduce_index(func, reducer, identity, * args)

These skeletons each take a function to be applied to each element of a distributed array
(like *smap*) but the result of this function is then reduced.
The second argument *reducer* is the function that takes two elements and calculates the
reduction across those elements.
The third argument *identity* is the value than when the reducer is applied with any other
value results in that same value.

As in *smap*, the first argument after this function argument must be a Ramba distributed array but
additional arguments are allowed and these may be of any type including other Ramba
distributed arrays.  However, in this latter case, all the Ramba distributed arrays must be
of the same shape and have the same distribution.
Conceptually, Ramba calls the function once for each element in the distributed arrays.
This function will be passed those individual elements and not the array as a whole whereas
all other argument types are passed to the function unmodified.
In some cases, the point in the index space that is being computed is necessary for the
computation itself.  For this purpose, *sreduce_index* first passes the point in the index space
to the function followed by all the other arguments as in *sreduce*.

In some cases, it may be beneficial to perform a different reduction depending on whether
the reduction is for computation within a single worker or across workers.
In such cases, a *SreduceReducer* object may be passed as *sreduce* *reducer* argument
and this object contains one reducer function for use within a worker and a different one
for use across workers.

#### *sstencil*

#### *scumulative*

### Groupby
