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

## Differences betwen Ramba and NumPy

Since NumPy was designed for non-distributed environments, some of functions in the NumPy API
may have poor performance when applied in a distributed environment.
This poor performance is typically caused by communication induced by these operations.
For example, the NumPy function *reshape* could cause large-scale movement of data between nodes
of the cluster to move from one partitioning scheme for the original array shape to a different
partitioning scheme for the new array shape.  Likewise, *advanced indexing*, where an array of
indices is used to index another array, is equivalent to a cross-node gather operation.  Note
that in some cases, certain variants of a function in the NumPy-API may have cheap distributed
implementations whereas others might not.  For example, *reshape* that adds or removes dimensions
of size 1 or swaps the positions of dimensions can be done cheaply whereas most other variants of
reshape are potentially very time consuming.

By default, Ramba will distributed arrays above a certain size threshold (currently 100) across
all the nodes of underlying cluster.
Also by default, Ramba uses a heuristic to determine how to partition such arrays across
the cluster.  This heuristic aims to minimize the surface area of the partition boundaries
across nodes.  This has the effect of splitting arrays across one dimension if that dimension
is much larger than all the others.  It also has the effect of roughly evenly splitting an array
across all the nodes.  If all the dimensions of an array are roughly of equivalent size then Ramba
will most likely produce partitions that are themselves roughly equal in their dimensions.
While this heuristic works well in many cases, in some cases finer control of partitioning may
be required.  For this purpose, all the functions in Ramba that generate a new array take
an additional *distribution* parameter (see the distribution section for details)
that allows the programmer to manually specify a partitioning.  Elementwise operations on such
arrays maintain this selected partitioning on the output arrays.

Another difference between Ramba and NumPy is that most operations in Ramba are executed lazily.
This allows Ramba to fuse many operations together into large functions that are then Numba-compiled
and run at native speeds with a single-pass over the data for cache friendliness.
Generally, Ramba will continue to collect these operations until some data is transfered outside of Ramba's
control such as through the *asarray* function that converts a Ramba array to a NumPy array.
If a programmer wishes to execute all Ramba collected operations, the Ramba *sync* function may be called.
In some cases, Ramba may perform pattern matching on these collected operations to translate them
to a more efficient distributed form.  This is useful in packages such as Xarray that can use
NumPy, Ramba, or Dask arrays to achieve higher-level operations such as groupby's and use lower-level
NumPy APIs to do so.  The individual operations may not be distribution friendly but there is a
higher-level Ramba implementation of the higher-level construct that is distribution friendly and this
pattern matching allows Ramba to determine when these higher-level constructs are being implemented.

## Ramba's Distribution Friendly APIs

To facilitate writing distribution-friendly, performant code, Ramba provides a variety of APIs in
addition to what NumPy provides.  These fall into two categories: algorithmic skeletons and
groupby operations.

### Algorithmic Skeletons

Algorithmic skeletons take one or more functions that can operate on remote data and encode
certain communication patterns for classes of algorithms.  These functions typically operate
on one element at a time and Ramba applies them to data in an efficient manner to achieve
the collective operation.

#### **ramba.smap**, **ramba.smap\_index**

**smap(func, arr, * args)**
**smap\_index(func, arr, * args)**

: Apply a function over a Ramba distributed arrays and optionally other arguments to produce another Ramba distributed array.

: **Parameters**
: : **func - a Python or Numba function**
: : : This function will be called once for each element in the input array and returns the value that is to be placed into the corresponding position in the output array.  The first argument to this function is the value of the input array at a given index.  Subsequent arguments to this function are the same as those passed to the smap function except that in the case of other Ramba distributed arrays the value at the same index of that array is passed instead.  For the smap\_index function, an additional argument is inserted at the beginning of the argument list that contains the given index on which the function is currently operating.
: : **arr - a Ramba distributed array**
: : : The input array to the map operation.  The output array will be of the same shape.
: : ** * args - any type
: : : Additional arguments to the map operation may be of any type including Ramba distributed arrays.  However, in this latter case, all the Ramba distributed arrays must be of the same shape and have the same distribution.
: **Returns**
: : A Ramba distributed array the same shape as the input array whose elements are the result of *func* applied to the corresponding elements of the input array.

#### *sreduce*, *sreduce_index*

**ramba.sreduce(func, reducer, identity, * args)**
**ramba.sreduce\_index(func, reducer, identity, * args)**

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

def sstencil(func, * args)

The *sstencil* skeleton executes a Ramba stencil (see below) on a Ramba distributed array.
The first argument to the *sstencil* skeleton is a Ramba stencil function as decorated
by the Ramba stencil decorator (described in its own section below).
The next argument to *sstencil* must be a Ramba distributed array.
Additional arguments, including other Ramba distributed arrays, can then also be passed.

#### *scumulative*

def scumulative(local\_func, final\_func, array)

This skeleton captures the algorithmic pattern where to compute the N'th element of the output
you need the N'th element from input along with the N-1'th element of the output.

For each output element, the first argument *local_func* is given the N'th element of the input array and the N-1'th element
of the output array and is expected to return the N'th output element.
Ramba does this in parallel on each worker calculating the portion of the array that is resident on that
worker.
Then the skeleton takes the final value of the first worker and sends that to the next
worker which then applies the *final_func* argument with the value from the previous worker
and the output of the *local_func*.  The skeleton repeats this process for each worker.
The third argument is a single Ramba distributed array over which to apply the cumulative operation
and produces another array of the same size.

#### *spmd*

def spmd(func, * args)

This skeleton takes a function that is run on each Ramba worker.  This skeleton also
takes one or more additional arguments that may be of any type.
Ramba distributed arrays passed to this function may have a special *get_local* call
made on them that returns a NumPy array holding the contents of that array that are
local to the executing worker.

### Groupby

def ndarray.groupby(self, dim, value\_to\_group, num\_groups=None)

The groupby method on Ramba array class takes the dimension (i.e., *dim*) on which to group and a
NumPy array *value_to_group* that maps indices on that dimension to a group identifier.
The optional *num_groups* parameter specifies the maximum group number.
If *num_groups* is not present then the maximum value in the *value_to_group* array is used as the number of groups.
This method returns a *RambaGroupby* object.

This *RambaGroupby* object supports array binary operators such as *+, -, \*, //, /, %, \*\*, >, <, >=, <=, ==, and !=*.
It also supports groupby operations such as *mean, nanmean, sum, count, prod, min, max, var, std.*

## Stencil

Ramba supports a stencil decorator similar to [Numba's](https://numba.pydata.org/numba-doc/latest/user/stencil.html) that lets a programmer easily specify a stencil kernel using relative indexing.
Executing this stencil kernel on an array means that the programmer does not have to write the stencil loop nor handle boundary conditions.
When a Ramba stencil is executed, Ramba will attempt to compile the kernel with Numba and then execute it.
An important note is that when using a Ramba stencil directly only NumPy arrays may be used.
If you wish to use a Ramba stencil with a Ramba distributed array then use the Ramba stencil skeleton described above.

## Optional Distribution Arguments

Array generating routines in Ramba generally provide an optional *distribution* argument not present in the original NumPy API.

