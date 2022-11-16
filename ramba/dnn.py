import ramba
import numpy as np


class relu:
    def forward(self, A):
        return ramba.smap("lambda x: max(0,x)", A)

# Convolution
# Weights size is nfilters * in_channels * k^ndim
# Forward pass is implemented 
# Input size is nframes * in_channels * x_0 * x_1 * ... * x_ndim-1
# Output size is nframes * nfilters * xout_0 * xout_1 * ... * xout_ndim-1
#   where xout_i = ceil( (x_i - k + 1) / stride )
class conv:
    def __init__(self, k, in_channels, ndim=2, nfilters=1, stride=1, has_bias=False, weights=None, bias=None):
        if weights is None:
            self.W = ramba.zeros((nfilters, in_channels)+(k,)*ndim)
        else:
            self.W = ramba.empty((nfilters, in_channels)+(k,)*ndim)
            self.W[:] = weights
        if bias is None:
            self.bias = ramba.zeros(nfilters) if has_bias else None
        else:
            self.bias = ramba.empty(nfilters)
            self.bias[:] = bias
        self.stride = stride

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

    def forward(self, A, stride=None):
        nfilters = self.W.shape[0]
        in_channels = self.W.shape[1]
        k = self.W.shape[2]
        ndim = self.W.ndim-2
        if stride is None:
            stride = self.stride
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
        out = self.__fwd(W,A,ndim,stride)
        if self.bias is not None:
            bias = ramba.expand_dims(self.bias, tuple(range(1,ndim)))
            out += bias
        return out


class dense:
    def __init__(self, inshape, k):
        self.inshape = inshape
        self.k = k
        if k>1:
            self.W = ramba.zeros((k,)+inshape)
        else:
            self.W = ramba.zeros(inshape)

    def forward(self, A):
        W = self.W
        if A.shape[-len(self.inshape):]!=self.inshape:
            raise ValueError("Shape mismatch")
        if len(A.shape)>len(self.inshape):
            W = np.broadcast_to(W, A.shape[:-len(self.inshape)]+W.shape)
        if k>1:
            A = np.expand_dims(A, -len(self.inshape))
        return (W*A).sum(axis=tuple(range(-len(self.inshape),0)))


