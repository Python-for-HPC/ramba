"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import numba
from cffi import FFI
import functools
import numpy as np
import math
import operator
from ramba.common import *

# simple array to specify a shard or view
# Here size is the k-dimensional size of the view/shard
#   index_start specifies the beginning of the global index range for this shard/view portion
#   base_offset is the start position in the base container
#   axis_map maps view axes to base container axes (or -1 for broadcasted axes)
#   Note: size, index_start, and axis_map are of length k; base_offset may have some other length
# the "shardview" is m*k array;  row 0 is size, row 1 is index_start, row 2 is axis map
#   item[3,0] is l, the length of the base_offset;  items[3,1]... are the elements of base_offset
# A distribution is an array of shardviews -- a n*m*k array, where n is number of nodes / remote workers
#   item[i] is the m*k sharview corresponding to the portion on node/worker i

@numba.njit(cache=True)
def shardview(size, index_start=None, base_offset=None, axis_map=None):
    #if np.any(size<1): size=size*0
    #i_s = size*0 if index_start is None else index_start
    #b_o = size*0 if base_offset is None else base_offset
    #a_m = np.arange(size.shape[0]) if axis_map is None else axis_map
    #assert((len(size)==len(i_s)) and (len(size)==len(a_m)))
    #m = 3 + (len(size)+len(b_o))//len(size)
    m = 3 + (len(size)+(len(size) if base_offset is None else len(base_offset)))//len(size)
    sv = np.zeros((m,len(size)), dtype=np.int32)
    if np.all(size>0): sv[0] = size.astype(np.int32)
    #sv[1] = i_s
    #sv[2] = a_m
    if index_start is not None: sv[1] = index_start.astype(np.int32)
    sv[2] = np.arange(len(size),dtype=np.int32) if axis_map is None else axis_map.astype(np.int32)
    sv[3,0] = len(size) if base_offset is None else len(base_offset)
    if base_offset is not None: _base_offset(sv)[:] = base_offset.astype(np.int32)
    return sv

#    def __repr__(self):
#        return '[start='+repr(self._index_start)+" size="+repr(self._size)+" base_offset="+repr(self._base_offset)+" axis_map="+repr(self._axis_map)+']'
#
#    #def __getstate__(self):
#    #    return {'index_start':self._index_start, 'size':self._size, 'base_offset':self._base_offset, 'axis_map':self._axis_map}
#
#    def __eq__(self, other):
#        raise Exception("Should use is_eq or dist_is_eq")

@numba.njit(cache=True)
def _size(s):
    return s[0]

@numba.njit(cache=True)
def _index_start(s):
    return s[1]

@numba.njit(cache=True)
def _stop(s):
    return s[0]+s[1]

@numba.njit(cache=True)
def _axis_map(s):
    return s[2]

@numba.njit(cache=True)
def _base_offset(s):
    return s[3:].reshape(-1)[1:s[3,0]+1]

@numba.njit(cache=True)
def len_size(s):
    return s.shape[1]

@numba.njit(cache=True)
def len_base_offset(s):
    return s[3,0]

@numba.njit(cache=True)
def _start(s):
    return _index_start(s)

@numba.njit(cache=True)
def get_start(sv):
    return _index_start(sv)

@numba.njit(cache=True)
def get_size(sv):
    return _size(sv)

@numba.njit(cache=True)
def get_base_offset(sv):
    return _base_offset(sv)

@numba.njit(cache=True)
def get_axis_map(sv):
    return sv._axis_map

@numba.njit(cache=True)
def get_stop(sv):
    return _stop(sv)

@numba.njit(cache=True)
def is_eq(sv, other):
    return (_size(sv)==_size(other)).all() and (_index_start(sv)==_index_start(other)).all() and (_base_offset(sv)==_base_offset(other)).all() and (_axis_map(sv)==_axis_map(other)).all()

@numba.njit(cache=True)
def is_empty(sv):
    return (_size(sv)==0).any()

@numba.njit(cache=True)
def is_compat(sv, other):
    return (_size(sv)==_size(other)).all() and (_index_start(sv)==_index_start(other)).all()

@numba.njit(cache=True)
def overlaps(sv, other):
    return ( np.logical_or(np.logical_and(_start(sv)<=_start(other), _start(other)<_stop(sv)), np.logical_and(_start(other)<=_start(sv), _start(sv)<_stop(other))) ).all()

@numba.njit(cache=True)
def contains(sv, other):
    return (not is_empty(other)) and ( np.logical_and(_start(sv)<=_start(other),_stop(other)<=_stop(sv)) ).all()

@numba.njit(cache=True)
def clean_range(sv):     # remove offset, axis_map
    return shardview( _size(sv), _index_start(sv) )

def base_to_index(sv, base):
    assert(len(base)==len_base_offset(sv))
    offset = [base[i]-b for i,b in enumerate(_base_offset(sv))]
    am = _axis_map(sv)
    return tuple([s +(0 if am[i]<0 else offset[am[i]]) for i,s in enumerate(_size(sv))])

def index_to_base(sv, index):
    assert(len(index)==len_size(sv))
    offset = [index[i]-s for i,s in enumerate(_start(sv))]
    invmap = -np.ones(len_base_offset(sv),dtype=np.int32)
    for i,v in enumerate(_axis_map(sv)):
        if v>=0: invmap[v]=i
    return tuple([bo + (0 if invmap[i]<0 else offset[invmap[i]]) for i,bo in enumerate(_base_offset(sv))])

@numba.njit(cache=True)
def index_to_base_as_array(sv, index):
    assert(len(index)==len_size(sv))
    offset = [index[i]-s for i,s in enumerate(_start(sv))]
    invmap = -np.ones(len_base_offset(sv),dtype=np.int32)
    for i,v in enumerate(_axis_map(sv)):
        if v>=0: invmap[v]=i
    bo = _base_offset(sv)
    return np.array([bo[i] + (0 if invmap[i]<0 else offset[invmap[i]]) for i in range(len(bo))])

def slice_to_local(sv, sl):
    assert(len(sl)==len_size(sv))
    s = index_to_base(sv, [x.start for x in sl])
    e = index_to_base(sv, [x.stop for x in sl])
    e = [x if x!=0 else None for x in e]   # special case to let border computation work with neg offset
    return tuple( [slice(s[i],e[i]) for i in range(len(s))] )

def div_to_local(sv, div):
    s = index_to_base(sv, div[0])
    e = index_to_base(sv, div[1]+1)
    e = [x if x!=0 else None for x in e]   # special case to let border computation work with neg offset
    #print (div,s,e)
    return tuple( [slice(s[i],e[i]) for i in range(len(s))] )

def to_slice(sv):
    s = _index_start(sv)
    e = s + _size(sv)
    return tuple( [slice(s[i],e[i]) for i in range(len(s))] )

def to_base_slice(sv):
    s = _base_offset(sv)
    e = np.ones(len_base_offset(sv), dtype=np.int32)
    for i,v in enumerate(_axis_map(sv)):
        if v>=0: e[v]=_size(sv)[i]
    e += s
    return tuple( [slice(s[i],e[i]) for i in range(len(s))] )


import numba.cpython.unsafe.tuple as UT
@numba.njit(cache=True)
def get_base_slice(sv, arr):
    t = UT.build_full_slice_tuple(arr.ndim)
    s = _base_offset(sv)
    e = np.ones(len_base_offset(sv), dtype=np.int32)
    for i,v in enumerate(_axis_map(sv)):
        if v>=0: e[v]=_size(sv)[i]
    e += s
    for i in range(arr.ndim):
        t = UT.tuple_setitem(t, i, slice(s[i],e[i]))
    return arr[t]



@numba.njit(cache=True)
def has_index(sv, index):
    s = _index_start(sv)
    e = _stop(sv)
    for i in range(len(index)):
        if s[i]>index[i] or index[i]>=e[i]: 
            return False
    return True
    #return (_index_start(sv)<=index).all() and (index<_stop(sv)).all()

@numba.njit(cache=True)
def mapslice(sv, sl):
    #print("HERE:",sl, sv, len(sl), len_size(sv))
    assert(len(sl)==len_size(sv))
    sv_s = _index_start(sv)
    sv_e = _stop(sv)
    #s = np.array([min(max(sl[i].start,sv_s[i]),sv_e[i]) for i in range(len(sl))])
    #e = np.array([min(max(sl[i].stop,sv_s[i]),sv_e[i]) for i in range(len(sl))])
    #s = np.array([min(max(x.start,sv_s[i]),sv_e[i]) for i,x in enumerate(numba.literal_unroll(sl))])
    #e = np.array([min(max(x.stop,sv_s[i]),sv_e[i]) for i,x in enumerate(numba.literal_unroll(sl))])
    s = np.array([min(max(sl[i].start,sv_s[i]),sv_e[i]) for i in range(len_size(sv))])
    e = np.array([min(max(sl[i].stop,sv_s[i]),sv_e[i]) for i in range(len_size(sv))])
    #si = s - np.array([sd.start for sd in sl])
    si = s - np.array([sl[i].start for i in range(len_size(sv))])
    #return shardview(e-s, si, np.array(index_to_base(sv, s)), _axis_map(sv))
    return shardview(e-s, si, index_to_base_as_array(sv, s), _axis_map(sv))

@numba.njit(cache=True)
def mapsv(sv, sl):
    #print("HERE:",sl, sv, len(sl), len_size(sv))
    assert(len_size(sl)==len_size(sv))
    sv_s = _index_start(sv)
    sv_e = _stop(sv)
    s = np.minimum(np.maximum(_index_start(sl),sv_s),sv_e) 
    e = np.minimum(np.maximum(_stop(sl),sv_s),sv_e)
    si = s - _index_start(sl)
    #return shardview(e-s, si, np.array(index_to_base(sv, s)), _axis_map(sv))
    return shardview(e-s, si, index_to_base_as_array(sv, s), _axis_map(sv))

@numba.njit(cache=True)
def intersect(sv, sl):
    assert(len_size(sv)==len_size(sl))
    sv_s = _index_start(sv)
    sv_e = _stop(sv)
    sl_s = _index_start(sl)
    sl_e = _stop(sl)
    s = np.array([min(max(sl_s[i],sv_s[i]),sv_e[i]) for i in range(len_size(sl))])
    e = np.array([min(max(sl_e[i],sv_s[i]),sv_e[i]) for i in range(len_size(sl))])
    return shardview(e-s, s, axis_map=_axis_map(sv))


# get a view of array (e.g. piece of a bcontainer) based on this shardview
# output is an np array with shape same as shardview size
# Will broadcast along additional dimensions as needed
def array_to_view(sv, arr):
    # sanity check
    #print ("axis map, size arr, offset",sv._axis_map,arr.shape,sv._base_offset)
    #assert(len(arr.shape)==len(sv._base_offset))
    # special case 0 size
    if any([v==0 for v in arr.shape]):
        return np.zeros(tuple([0]*len_size(sv)))
    # special case do nothing
    if len_size(sv)==len(arr.shape) and all([ arr.shape[i]==_size(sv)[i] and _axis_map(sv)[i]==i for i in range(len_size(sv))]):
        return arr
    # compute slice of array needed for output (and sanity check expected shape of array)
    #   expected size is 1 in any broadcasted dimensions, and no more than shard size in others
    sl = [0]*len_base_offset(sv)
    #sl = [ slice(None) if v==0 else 0 for v in arr.shape ]
    shp = [1]*len_base_offset(sv)
    #shp = [ min(1,v) for v in arr.shape ]
    for i,v in enumerate(_axis_map(sv)):
        if v>=0:
            sl[v] = slice(None)
            shp[v] = min(_size(sv)[i],arr.shape[v])
    sl = tuple(sl)
    shp = tuple(shp)
    #print ("axis map, size arr, expected:",sv._axis_map,arr.shape,shp)
    assert (shp==arr.shape)
    arr2 = arr[sl]
    if not isinstance(arr2, (np.ndarray)):
        arr2 = np.array([arr2])  # in case we get single element
    # add additonal axes as needed (numpy broadcast)
    newdims = [_size(sv)[i] for i,v in enumerate(_axis_map(sv)) if v<0]
    if len(newdims)>0:
        newdims = tuple(newdims)+arr2.shape
        arr2 = np.broadcast_to(arr2, newdims)
    # get new mapping from arr2 axes to output axesa
    sortmap = sorted( list(_axis_map(sv)) )
    if all(_axis_map(sv) == sortmap):  # nothing more to do
        return arr2
    outmap = []
    for i,v in enumerate(_axis_map(sv)):
        j = sortmap.index(v)
        sortmap[j]=-2
        outmap.append(j)
    # move axes and return
    #print ("outmap", outmap)
    outarr = np.moveaxis(arr2, outmap, [i for i in range(len_size(sv))])
    return outarr

@numba.njit(cache=True)
def to_division(sv):
    ret = np.full((2,len(_index_start(sv))),-1)
    ret[0] = _index_start(sv)
    ret[1] += _stop(sv)
    return ret
    #return np.array([_index_start(sv), _stop(sv) - 1])

@numba.njit(cache=True)
def shape_to_div(shape):
    res = np.zeros((2,len(shape)))
    for i in range(len(shape)):
        res[1,i] = shape[i] - 1
    return res

# still need?
@numba.njit(cache=True)
def slice_to_fortran(sl):
    ret = list(sl)
    ret.reverse()
    return tuple(ret)

@numba.njit(cache=True)
def clean_dist(dist):
    first_clean = clean_range(dist[0])
    dshape = dist.shape
    d2 = np.empty((dshape[0], first_clean.shape[0], first_clean.shape[1]), dtype=np.int32)
    d2[0] = first_clean
    for i,s in enumerate(dist[1:]):
        d2[i+1] = clean_range(s)
    return d2


@numba.njit # doesn't work with cache=True
def get_splits( r, v, s, e ):
    if len(r)==0:
        s1 = np.array(s[1:])
        e1 = np.array(e[1:])
        v.append( shardview( e1-s1, s1 ) )
        return
    r0 = r[0]
    for i in range(len(r0)-1):
        get_splits( r[1:], v, s+[r0[i]], e+[r0[i+1]] )


#@numba.njit
#def get_range_splits(s1, s2):
#    assert(len_size(s1)==len_size(s2))
#
#    axis_ranges = [ sorted(set([ _start(s1)[i], _stop(s1)[i], _start(s2)[i], _stop(s2)[i] ])) for i in range(len_size(s1)) ]
#    all_splits = [ s1 ]
#    get_splits( axis_ranges, all_splits, [-1000], [-1000] )
#    all_splits = all_splits[1:]
#    s1_splits = [ s for s in all_splits if contains(s1, s) ]
#    s2_splits = [ s for s in all_splits if contains(s2, s) ]
#    return s1_splits, s2_splits

@numba.njit # doesn't work with cache=True
def get_range_splits_list(svl):
    axis_ranges = [ sorted(set( [x for s in svl for x in [_start(s)[i], _stop(s)[i]]] )) for i in range(len_size(svl[0])) ]
    all_splits = [ svl[0] ]
    get_splits( axis_ranges, all_splits, [-1000], [-1000] )
    all_splits = all_splits[1:]
    return all_splits

@numba.njit(cache=True)
def get_range_splits(s1, s2):
    all_splits = get_range_splits_list([s1, s2])
    s1_splits = [ s for s in all_splits if contains(s1, s) ]
    s2_splits = [ s for s in all_splits if contains(s2, s) ]
    return s1_splits, s2_splits

@numba.njit(cache=True)
def compatible_distributions(d1, d2):
    if not len(d1)==len(d2): return False
    #return all([ is_compat(d1[i],d2[i]) for i in range(len(d1))])
    for i in range(len(d1)):
        if not is_compat(d1[i],d2[i]): return False
    return True

@numba.njit(cache=True)
def dist_is_eq(d1, d2):
    #return len(d1)==len(d2) and all([ is_eq(d1[i],d2[i]) for i in range(len(d1))])
    if not len(d1)==len(d2): return False
    for i in range(len(d1)):
        if not is_eq(d1[i], d2[i]): return False
    return True


@numba.njit(cache=True)
def slice_distribution(sl, dist):
    ret = np.empty_like(dist,dtype=np.int32)
    for i in range(dist.shape[0]):
        ret[i] = mapslice(dist[i],sl)
    return ret
    #return np.array([ mapslice(dist[i],sl) for i in range(dist.shape[0]) ])

@numba.njit(cache=True)
def find_index(dist, index):
    for i in range(len(dist)):
        if has_index(dist[i],index): return i
    return None


@numba.njit(cache=True)
def get_overlaps(k, dist1, dist2):
    return [i  for i in range(dist1.shape[0]) if overlaps(dist1[i],dist2[k]) or overlaps(dist1[k],dist2[i])]
    #return [i  for i in range(dist1.shape[0])if (not is_empty(intersect(dist1[i],dist2[k]))) or (not is_empty(intersect(dist1[k],dist2[i])))]

@numba.njit(cache=True)
def divisions_to_distribution(divs, base_offset=None, axis_map=None):
    #dprint(4,"Divisions to convert:", divs, divs.shape, type(divs))
    divshape = divs.shape
    if base_offset is None:
        svl = [ shardview(size=divs[i][1]-divs[i][0]+1, index_start=divs[i][0], base_offset=None, axis_map=axis_map) for i in range(divshape[0]) ]
    else:
        svl = [ shardview(size=divs[i][1]-divs[i][0]+1, index_start=divs[i][0], base_offset=base_offset[i], axis_map=axis_map) for i in range(divshape[0]) ]
    ret = np.empty((divshape[0],svl[0].shape[0],svl[0].shape[1]),dtype=np.int32)
    for i,sv in enumerate(svl):
        ret[i]=sv
    return ret

def division_to_shape(divs):
    assert(isinstance(divs, np.ndarray))
    divshape = divs.shape
    assert(len(divshape) == 2)
    return tuple(divs[1,:] - divs[0,:] + 1)

@numba.njit(cache=True)
def distribution_to_divisions(dist):
    #return np.array([ [_index_start(d), _stop(d)-1] for d in dist ])
    ret = np.empty((dist.shape[0],2,dist.shape[2]),dtype=np.int32)
    for i,d in enumerate(dist):
        ret[i][0] = _index_start(d)
        ret[i][1] = _stop(d)-1
    return ret

def default_distribution(size):
    num_dim = len(size)
    starts = np.zeros(num_dim, dtype=np.int64)
    ends = np.array(list(size), dtype=np.int64)
    # the ends are inclusive, not one past the last index
    ends -= 1
    divisions = np.empty((num_workers,2,num_dim), dtype=np.int64)
    if do_not_distribute(size):
        make_uni_dist(divisions, 0, starts, ends)
    else:
        if regular_schedule:
            compute_regular_schedule(size, divisions)
            dprint(3, "compute_regular output:", size, divisions)
        else:
            numba_workqueue.do_scheduling_signed(num_dim, ffi.cast("int*", starts.ctypes.data), ffi.cast("int*", ends.ctypes.data), num_workers, ffi.cast("int*", divisions.ctypes.data), 0)
    return divisions_to_distribution(divisions)

def block_intersection(a, b):
    ashape = a.shape
    bshape = b.shape
    assert(len(ashape) == len(bshape))
    num_dims = ashape[1]
    dprint(3, "block_intersection:", a, a.shape, b, b.shape, num_dims)
    shared = np.empty((2, num_dims), dtype=np.int64)
    empty = False

    for k in range(num_dims):
        fstart = b[0][k]
        fend = b[1][k]
        if fstart > fend:
            empty = True
            break

        sstart = a[0][k]
        send = a[1][k]
        if sstart > send:
            empty = True
            break

        dprint(3, "vals:", k, fstart, fend, sstart, send)

        shared[0][k] = max(fstart, sstart)
        shared[1][k] = min(fend, send)

        # The range in this dimension contains no items so the whole shared region is 0 sized.
        if shared[0][k] > shared[1][k]:
            empty = True
            break

    if empty:
        dprint(3, "block_intersection empty")
        return None
    else:
        dprint(3, "block_intersection results non-empty shared:", shared)
        return shared

def broadcast(distribution, broadcasted_dims, size):
    new_dims = len(size) - len(_size(distribution[0]))
    #new_dims = len(size) - len(distribution[0]._size)
    dprint(4, "shardview::broadcast", distribution, broadcasted_dims, size, new_dims)
    new_axis_map = np.array([-1 if broadcasted_dims[j] else _axis_map(distribution[0])[j - new_dims] for j in range(len(size))])
    ret = []
    for i in range(len(distribution)):
        new_size = np.array([size[j] if broadcasted_dims[j] else _size(distribution[i])[j - new_dims] for j in range(len(size))])
        new_start = np.array([0 if broadcasted_dims[j] else _index_start(distribution[i])[j - new_dims] for j in range(len(size))])
        #new_offset = np.array([0 if broadcasted_dims[j] else distribution[i]._base_offset[j - new_dims] for j in range(len(size))])
        #ret.append(shardview(new_size, new_start, new_offset))
        ret.append(shardview(new_size, new_start, _base_offset(distribution[i]), new_axis_map))
    return np.array(ret)

# re-orders and/or removes axes.  newmap is a list that specifies which axis maps to the index.  Removes axes if lenght less than curent shardview dimensionaily.  Elements of newmap must be unique, and in range
def remap_axis(size, distribution, newmap):
    old_ndims = len_size(distribution[0])
    old_map = _axis_map(distribution[0])
    assert(len(newmap)<=old_ndims and all([0<=v and v<old_ndims for v in newmap]) and len(newmap)==len(set(newmap)))
    new_axis_map = np.array([ old_map[i] for i in newmap ])
    new_global_size = tuple([ size[i] for i in newmap ])
    new_dist = []
    for i in range(len(distribution)):
        new_size = np.array([_size(distribution[i])[j] for j in newmap])
        new_start = np.array([_index_start(distribution[i])[j] for j in newmap])
        new_dist.append(shardview(new_size, new_start, _base_offset(distribution[i]), new_axis_map))
    return new_global_size, np.array(new_dist)


def compute_from_border(size, distribution, border):
    # convert distribution to old format
    distribution = distribution_to_divisions(distribution)
    # with_border will hold the range that each worker needs including the border
    with_border = np.zeros_like(distribution)
    num_dims = len(size)
    dist_shape = distribution.shape
    dprint(3, "distribution:", distribution)
    num_workers = dist_shape[0]
    # fill in with_border making sure that you don't say you need indices outside
    # the total size.
    for i in range(num_workers):
        for j in range(num_dims):
            with_border[i][0][j] = max(0, distribution[i][0][j] - border)
        for j in range(num_dims):
            with_border[i][1][j] = min(size[j], distribution[i][1][j] + border)

    # create the output which is all possible pairs of what each worker would need
    # from each other worker.  This is nothing but the overlap between the worker that
    # owns the block and the worker that needs the border for each dimension.
    shared = np.empty((num_workers, num_workers, 2, num_dims), dtype=np.int64)
    # will hold only the non-empty shared regions between workers
    from_ret = [{} for i in range(num_workers)]
    to_ret = [{} for i in range(num_workers)]

    for i in range(num_workers):
        for j in range(num_workers):
            if i == j:
                continue

            empty = False

            for k in range(num_dims):
                dprint(3, "inner:", i, j, distribution[j], with_border[j])
                fstart = with_border[j][0][k]
                fend = with_border[j][1][k]
                sstart = distribution[i][0][k]
                send = distribution[i][1][k]

                if fstart < sstart:
                    shared[i][j][0][k] = sstart
                    shared[i][j][1][k] = fend
                elif sstart < fstart:
                    shared[i][j][0][k] = fstart
                    shared[i][j][1][k] = send
                else:
                    shared[i][j][0][k] = fstart
                    shared[i][j][1][k] = min(fend, send)

                # The range in this dimension contains no items so the whole shared region is 0 sized.
                if shared[i][j][0][k] > shared[i][j][1][k]:
                    empty = True

            if not empty:
                dprint(3, "adding:", i, j, shared[i][j], type(shared[i][j]))
                from_ret[j][i] = shared[i][j]
                to_ret[i][j] = shared[i][j]

    """
    dprint(2, "compute_from_border result:")
    if debug:
        for i in range(num_workers):
            print(i, from_ret[i])
        for i in range(num_workers):
            print(i, to_ret[i])
    """

    return from_ret, to_ret

# ------------------------------------


def make_uni_dist(divisions, node, starts, ends):
    # These two lines put no distribution on any worker.
    divisions[:,0,:] = 1
    divisions[:,1,:] = 0
    # Now put everything on worker 0.
    divisions[node,0,:] = starts
    divisions[node,1,:] = ends

def make_uni_dist_from_shape(num_workers, node, shape):
    divisions = np.empty((num_workers, 2, len(shape)), dtype=np.int64)
    starts = np.zeros(len(shape), dtype=np.int64)
    ends = np.array(list(shape), dtype=np.int64)
    # the ends are inclusive, not one past the last index
    ends -= 1
    make_uni_dist(divisions, node, starts, ends)
    return divisions



def compute_regular_schedule(size, divisions):
    divisions[:] = compute_regular_schedule_internal(size)

@functools.lru_cache(maxsize=None)
def compute_regular_schedule_internal(size):
    num_dim = len(size)
    divisions = np.empty((num_workers,2,num_dim), dtype=np.int64)
    the_factors = dim_factor_dict[num_dim]
    best = None
    best_value = math.inf

    largest = [0] * num_dim
    smallest = [0] * num_dim

    def get_div_sizes(dim_len, num_div):
        low = dim_len // num_div
        if dim_len % num_div == 0:
            return low, low, low, low

        rem = dim_len - (low * (num_div-1))
        if rem >= num_div:
            main = low + 1
            rem = dim_len - (main * (num_div-1))
        else:
            main = low
        if rem == 0:
            rem = main
        return main, rem, max(main, rem), min(main, rem)

    for factored in the_factors:
        not_possible = False
        for i in range(num_dim):
            if factored[i] > size[i]:
               not_possible = True
               break
            _, _, largest[i], smallest[i] = get_div_sizes(size[i], factored[i])
        if not_possible:
            continue
        ratio = np.prod(largest) / np.prod(smallest)
        if ratio < best_value:
            best_value = ratio
            best = factored

    assert(best is not None)
    divshape = divisions.shape
    assert(divshape[2] == len(best))
    main_divs = [0] * num_dim
    for j in range(num_dim):
        main_divs[j], _, _, _ = get_div_sizes(size[j], best[j])

    def crsi_div(divisions, best, main_divs, index, min_worker, max_worker, size):
        if index >= len(best):
            return

        total_workers = max_worker - min_worker + 1
        chunks_here = total_workers // best[index]
        last = -1

        for i in range(min_worker, max_worker + 1, chunks_here):
            for j in range(chunks_here):
                divisions[i+j,0,index] = last + 1
                divisions[i+j,1,index] = last + main_divs[index]
                if divisions[i+j,1,index] > size[index]:
                    divisions[i+j,1,index] = size[index]
            last += main_divs[index]
            crsi_div(divisions, best, main_divs, index + 1, i, i + chunks_here - 1, size)

    crsi_div(divisions, best, main_divs, 0, 0, num_workers-1, np.array(size) - 1)
    return divisions

def exps_to_factor(factors, exps):
    rest = 1
    for i in range(len(factors)):
        rest *= (factors[i] ** exps[i])
    return rest

def gen_ind_factor_internal(fset, factors, exps, remaining_len, thus_far, index, part_exps):
    if index >= len(exps):
        rest = exps_to_factor(factors, part_exps)
        new_thus = thus_far + [rest]
        gen_ind_factors(fset, factors, list(map(operator.sub, exps, part_exps)), remaining_len - 1, new_thus)
    else:
        for i in range(exps[index]+1):
            part_exps[index] = i
            gen_ind_factor_internal(fset, factors, exps, remaining_len, thus_far, index + 1, part_exps)

def gen_ind_factors(fset, factors, exps, remaining_len, thus_far):
    if remaining_len == 1:
        rest = exps_to_factor(factors, exps)
        complete_factors = thus_far + [rest]
        fset.add(tuple(complete_factors))
    else:
        gen_ind_factor_internal(fset, factors, exps, remaining_len, thus_far, 0, [0]*len(exps))

def gen_dim_factor_dict(factors, exps):
    dim_factor = {1:set(), 2:set(), 3:set(), 4:set()}

    gen_ind_factors(dim_factor[1], factors, exps, 1, [])
    gen_ind_factors(dim_factor[2], factors, exps, 2, [])
    gen_ind_factors(dim_factor[3], factors, exps, 3, [])
    gen_ind_factors(dim_factor[4], factors, exps, 4, [])

    return dim_factor

def gen_prime_factors(n):
    factors = []
    exps = []

    def one_prime(n, p):
        if n % p == 0:
            factors.append(p)
            exps.append(1)
            n = n // p

            while n % p == 0:
                n = n // p
                exps[-1] += 1
        return n

    n = one_prime(n, 2)

    for i in range(3, int(math.sqrt(n)) + 1, 2):
        n = one_prime(n, i)

    if n != 1:
        factors.append(n)
        exps.append(1)

    return factors, exps

num_worker_factors, num_worker_exps = gen_prime_factors(num_workers)
dim_factor_dict = gen_dim_factor_dict(num_worker_factors, num_worker_exps)

