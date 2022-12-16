import ramba
import numpy as np

# Activation functions
def relu(A):
    return ramba.maximum(A,0)

def sigmoid(A):
    return ramba.smap("lambda x: 1/(1+numpy.e^(-x))", A, imports=['numpy'])


# Dropout layer
# This is only used in training and acts as a no-op in inferencing
# included for compatibility
class dropout:
    def __init__(self, rate):
        self.rate = rate
    def forward(self, A):
        return A

# Pooling layer
# Forward pass is implemented 
# Poolfunc is 'max', 'min', 'sum', or 'mean'
# Input size is nframes * in_channels * x_0 * x_1 * ... * x_ndim-1
# Output size is nframes * in_channels * xout_0 * xout_1 * ... * xout_ndim-1
#   where xout_i = (x_i - k ) // stride + 1
class pool:
    def __init__(self, k, poolfunc='max', ndim=2, stride=1):
        self.k = k
        self.ndim = ndim
        self.stride = stride
        assert poolfunc in ['max', 'min', 'sum', 'mean']
        self.poolfunc = poolfunc

    @classmethod
    def _pool__fwd(cls, A, fn, k, n, s, j=0, o=None):
        if j<n:
            l = A.shape[-(n-j)]
            extra = (slice(None),)*(n-j-1)
            start=0
            if o is None:
                o = cls.__fwd(A[(...,slice(0,l-k+1,s))+extra], fn, k, n, s, j+1)
                start=1
            for i in range(start,k):
                o = cls.__fwd(A[(...,slice(i,l-k+1+i,s))+extra], fn, k, n, s, j+1, o)
        else:
            if o is None:
                o = A
            elif fn=='min':
                o = ramba.minimum(o,A)
            elif fn=='max':
                o = ramba.maximum(o,A)
            elif fn=='sum':
                o = o + A
        return o

    def forward(self, A, stride=None):
        k = self.k
        ndim = self.ndim
        poolfunc = self.poolfunc
        if poolfunc=='mean':
            poolfunc = 'sum'
            needscale=True
        else:
            needscale=False
        if stride is None:
            stride = self.stride
        if A.ndim<ndim:
            raise ValueError("Insufficient number of dimensions in input")
        out = self.__fwd(A,poolfunc,k,ndim,stride)
        if needscale:
            out = out * (1.0/(k*k))
        return out


# Convolution
# Weights size is nfilters * in_channels * k^ndim
# Forward pass is implemented 
# Input size is nframes * in_channels * x_0 * x_1 * ... * x_ndim-1
# Output size is nframes * nfilters * xout_0 * xout_1 * ... * xout_ndim-1
#   where xout_i = (x_i - k ) // stride + 1
class conv:
    def __init__(self, k, in_channels, ndim=2, nfilters=1, stride=1, padding=0, has_bias=False, weights=None, bias=None, dtype=np.float32):
        if weights is None:
            self.W = ramba.zeros((nfilters, in_channels)+(k,)*ndim, dtype=dtype)
        else:
            self.W = ramba.empty((nfilters, in_channels)+(k,)*ndim, dtype=dtype)
            self.W[:] = ramba.fromarray(weights)
        if bias is None:
            self.bias = ramba.zeros(nfilters, dtype=dtype) if has_bias else None
        else:
            self.bias = ramba.empty(nfilters, dtype=dtype)
            self.bias[:] = ramba.fromarray(bias)
        self.stride = stride
        self.padding = padding

    @classmethod
    def _conv__fwd(cls, W, A, n, s, j=0, o=None):
        if j<n:
            k = W.shape[-(n-j)]
            l = A.shape[-(n-j)]
            extra = (slice(None),)*(n-j-1)
            start=0
            if o is None:
                o = cls.__fwd(W[(...,slice(0,1))+extra], A[(...,slice(0,l-k+1,s))+extra], n, s, j+1)
                start=1
            for i in range(start,k):
                o = cls.__fwd(W[(...,slice(i,i+1))+extra], A[(...,slice(i,l-k+1+i,s))+extra], n, s, j+1, o)
        else:
            k = W.shape[-(n+1)]
            extra = (slice(None),)*n
            start=0
            if o is None:
                o = W[(...,0)+extra] * A[(...,0)+extra]
                start=1
            for i in range(start,k):
                o = o+W[(...,i)+extra] * A[(...,i)+extra]
        return o

    def forward(self, A, stride=None, padding=None):
        if isinstance(A, np.ndarray):
            A = ramba.fromarray(A)
        nfilters = self.W.shape[0]
        in_channels = self.W.shape[1]
        k = self.W.shape[2]
        ndim = self.W.ndim-2
        if stride is None:
            stride = self.stride
        if padding is None:
            padding = self.padding
        if A.ndim<ndim:
            raise ValueError("Insufficient number of dimensions in input")
        if in_channels >1:
            if A.ndim<ndim+1:
                raise ValueError("Insufficient number of dimensions in input")
            if A.shape[-(ndim+1)]!=in_channels:
                raise ValueError("Number of channels do not match")
        else: # insert channels dimension if in_channels is 1 and input does not have it
            if A.ndim==ndim or A.shape[-(ndim+1)]!=1:
                A = ramba.expand_dims(A, -(ndim+1))
        W = self.W
        if nfilters>1: # insert nfilters dimension if needed (nfilters>1)
            A = ramba.broadcast_to(A, (nfilters,)+A.shape)
            A = ramba.moveaxis(A, 0, -(ndim+2))
        else: # remove nfilters dimension from weights if it is 1
            W = W[0]
        if padding>0:
            A = ramba.pad(A, ((0,0),)*(A.ndim-ndim)+((padding,padding),)*ndim)
        out = self.__fwd(W,A,ndim,stride)
        if self.bias is not None:
            bias = ramba.expand_dims(self.bias, tuple(range(1,ndim+1)))
            out = out + bias
        return out


class dense:
    def __init__(self, inshape, k, dtype=np.float32):
        self.inshape = inshape
        self.k = k
        if k>1:
            self.W = ramba.zeros((k,)+inshape, dtype=dtype)
        else:
            self.W = ramba.zeros(inshape, dtype=dtype)

    def forward(self, A):
        W = self.W
        if A.shape[-len(self.inshape):]!=self.inshape:
            raise ValueError("Shape mismatch")
        if len(A.shape)>len(self.inshape):
            W = np.broadcast_to(W, A.shape[:-len(self.inshape)]+W.shape)
        if k>1:
            A = np.expand_dims(A, -len(self.inshape))
        return (W*A).sum(axis=tuple(range(-len(self.inshape),0)))


