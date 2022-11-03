import ramba
import numpy as np


class relu:
    def forward(self, A):
        return ramba.smap("lambda x: max(0,x)", A)


class conv:
    def __init__(self, k, in_channels, ndim=2, nfilters=1, stride=1, has_bias=False):
        self.k = k
        self.ndim=ndim
        self.in_channels = in_channels
        self.W = np.zeros((nfilters, in_channels)+(k,)*ndim)
        self.has_bias=has_bias
        if has_bias:
            self.bias = np.zeros(nfilters)

    def forward(self, A):
        inshape = A.shape
        if len(inshape)<self.ndim:
            raise ValueError("Insufficient number of dimension in input")
        outshape_low = tuple([x-self.k+1 for x in inshape[-self.ndim:]])
        W = self.W
        if self.in_channels >1:
            if len(inshape)<self.ndim+1:
                raise ValueError("Insufficient number of dimension in input")
            if inshape[-(self.ndim+1)]!=self.in_channels:
                raise ValueError("Number of channels do not match")
        else:
            if len(inshape)==self.ndim or inshape[-(self.ndim+1)]!=1:
                A = np.broadcast_to(A, (1,)+A.shape)
                A = np.moveaxis(A, 0, -(self.ndim+1))
                inshape = A.shape
        outshape_high = inshape[:-(self.ndim+1)]
        if len(outshape_mid)>0:
            W = np.broadcast_to(W, outshape_mid+W.shape)
            W = np.moveaxis(W, len(outshape_mid), 0)
        if W.shape[0]>1:
            outshape_mid=(W.shape[0],)
        else:
            outshape_mid=()



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
        return (W*A).sum(axis=list(range(-len(self.inshape),0)))


