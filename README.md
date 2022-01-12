# Ramba
Ramba is a Python project that provides a fast, distributed, Numpy-like array API using compiled Numba functions 
and a Ray or MPI-based distributed backend.  It also provides a way to easily integrate Numba-compiled remote
functions and remote Actor methods in Ray.  

The main use case for Ramba is as a fast, drop-in replacement for Numpy.  Although Numpy typically uses C
libraries to implement array functions, it is still largely single threaded, and typically does not make
use of multiple cores for most functions, and definitely cannot make use of multiple nodes in a cluster. 

Ramba lets Numpy programs make use of multiple cores and multiple nodes with little to no code changes.

## Example
Consider this simple example of a large computation in Numpy:
~~~python
# test-numpy.py
import numpy as np
import time

t0 = time.time()
A = np.arange(1000*1000*1000)/1000.0
B = np.sin(A)
C = np.cos(A)
D = B*B + C**2

t1 = time.time()
print (t1-t0)
~~~

Let us try running this code on a dual-socket server with 36 cores/72 threads and 128GB of DRAM:
~~~
% python test-numpy.py
47.55583119392395
~~~
This takes over 47 seconds, but if we monitor resource usage, we will see that only a single core is used.  All others remains idle.  

We can very easily modify the code to use Ramba instead of Numpy:
~~~python
# test-ramba.py
import ramba as np  # Use ramba in place of numpy
import time

t0 = time.time()
A = np.arange(1000*1000*1000)/1000.0
B = np.sin(A)
C = np.cos(A)
D = B*B + C**2

np.sync()           # Ensure any remote work is complete to get accurate times
t1 = time.time()
print (t1-t0)
~~~
Note that the only changes are the import line, and the addition of the `np.sync()`.  The latter is only needed to wait for 
all remote work to complete, so we can get an accurate measure of execution time.

Now let us try running the ramba version:
~~~
% python test-ramba.py
3.860828161239624
~~~
The Ramba version saturates all of the cores, and results in about 12x speedup over the original numpy version. (Why only 12x?  Three factors 
contribute to this: 1) this total includes some of the intiialization time; 2) Time for JIT compile (~1 second here); 3) This code is 
memory-bandwidth bound, so after a point, additional cores will just end up waiting on memory).  Importantly, this performance gain 
is achieved with no significant change to the code.


# Quick Start

Quick install with Conda / Miniconda and git:
~~~
conda create --name ramba -y
conda activate ramba
conda install "python<3.9" numba cloudpickle pyzmq -y
pip install -U ray
git clone https://github.com/Python-for-HPC/ramba
cd ramba
python setup.py install 
~~~

Now run the provided sample file:
~~~
python sample/test-ramba.py
~~~

The first time you use Ramba, it will compile some internal functions, so the first run may be slow.  Let's run it again to see typical run times:
~~~
python sample/test-ramba.py
~~~

Finally, let's compare to the numpy version:
~~~
python sampe/test-numpy.py
~~~


# Installation
We suggest using conda to setup an environment for running ramba.  
## Prerequisites
Ramba was developed and tested on Linux, usng both Ray and MPI backends.  Ramba may work on Windows using MPI, though this has not been extensively tested.  ZeroMQ is needed for the communication layer.  
Ramba uses pickle version 5 for serializtion of objects.  This should already be available if running Python 3.8 or higher.  If not, please install the pickle5 package.  In addition, cloudpickle is also needed to serialize functions (as this is not possible through the normal pickle package).  
Finally, ramba uses numba for JIT compilation.  
Note: Ray requires Python <3.9 (Update: Python 3.9 is now supported experimentally by Ray)

Thus the requirements are:
- mpi4py and/or ray
- pyzmq
- cloudpickle
- pickle5 (if using python < 3.8)
- numba

Optional packages:
- h5py
- PIL

## Installation process
- Download / clone this repository
- run: python setup.py install

# Usage
Coming soon!

# Numpy Compatibility
Current status of Ramba compatibility with Numpy APIs.  Key:  &#x1f7e2; works   &#x1f7e1; partial    &#x1f534; not implemented

|Feature/API | Function/type |Status&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;| Notes
|:-----------|:--------------  |:--------------------------|:-----
|Array Types | Numerical       | &#x1f7e2; works           | complex not tested
|            | Boolean         | &#x1f7e2; works           |
|            | String          | &#x1f534; not implemented |
|            | Objects         | &#x1f7e1; partial         |
|Creation    | from size/value | &#x1f7e1; partial         | empty, ones_like, etc.;  missing: full, identity
|            | from data       | &#x1f7e1; partial         | fromfunction, fromarray
|            | ranges          | &#x1f7e1; partial         | arange, linspace, mgrid
|Array Manipulation| reshape   | &#x1f7e1; partial         | reshape is very expensive in distributed context, so only very limited support;  use reshape_copy
|            | axis manipulation | &#x1f7e1; partial       | T, transpose; missing: swapaxes, rollaxis, moveaxis 
|            | dimensionality  | &#x1f7e1; partial         | only broadcast_to
|            | joining arrays  | &#x1f7e1; partial         | only concatenate
|            | splitting arrays| &#x1f534; not implemented |
|            | tiling          | &#x1f534; not implemented |
|            | insert/remove elements | &#x1f534; not implemented |
|            | rearrange elements | &#x1f534; not implemented |
|Index/slice | range slice     | &#x1f7e1; partial         | produces view like in numpy; skips not supported
|            | masked arrays   | &#x1f534; not implemented |
|            | fancy indexing  | &#x1f534; not implemented |
|            | index routines  | &#x1f534; not implemented | ("where" partly works)
|Math        | arithmetic operations | &#x1f7e2; works     | +, -, +=, //, etc. 
|            | comparisons     | &#x1f7e2; works           | 
|            | logical operations | &#x1f534; not implemented |
|            | trig functions  | &#x1f7e1; partial         |
|            | power           | &#x1f7e1; partial         | pow, exp, log, sqrt, square
|            | floating manip. | &#x1f534; not implemented | (isnan works, though)
|            | bit twiddling   | &#x1f534; not implemented |
|            | reductions      | &#x1f7e1; partial         | sum, prod, min, max; axis parameter works
|            | matmul          | &#x1f7e1; partial         | matmul / dot for 1D and 2D only
| ufunc      |                 | &#x1f7e2; works           |
| FFT        |                 | &#x1f534; not implemented |
| linalg     |                 | &#x1f534; not implemented |
| random     |                 | &#x1f7e1; partial         |
| matlib     |                 | &#x1f534; not implemented |
| statistics |                 | &#x1f534; not implemented | (except: bincount is implemented)

It can be assumed that Numpy features not mentioned in this table are not implemented.


## Security Note
Please note that this work is a research prototype and that it internally uses Ray and/or ZeroMQ for
communication.  These communication channels are generally not secured or authenticated.  This means
that data sent across those communication channels may be visible to eavesdroppers.  Also, it is means
that malicious users may be able to send messages to this system that are interpreted as legitimate.
This may result in corrupted data and since pickled functions are also sent over the communication
channel, malicious attackers may be able to run arbitrary code.

Since this prototype uses Ray, occasionally orphaned Ray processes may be left around.  These can
be stopped with the command "ray stop".
