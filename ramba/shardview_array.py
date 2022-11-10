"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import numba
from numba import literal_unroll
import numpy as np
from ramba.common import *

ramba_dist_dtype = np.int32

if ramba_big_data:
    dprint(1, "Will use big data.")
    ramba_dist_dtype = np.int64

ramba_dummy_index = ramba_dist_dtype(0)

# simple array to specify a shard or view
# Here size is the k-dimensional size of the view/shard
#   index_start specifies the beginning of the global index range for this shard/view portion
#   base_offset is the start position in the base container
#   axis_map maps view axes to base container axes (or -1 for broadcasted axes)
#   steps is the step size used for each axis
#   Note: size, index_start, and axis_map are of length k; base_offset may have some other length
# the "shardview" is m*k array;  row 0 is size, row 1 is index_start, row 2 is axis map
#   item[3,0] is l, the length of the base_offset;  items[3,1]... are the elements of base_offset
# A distribution is an array of shardviews -- a n*m*k array, where n is number of nodes / remote workers
#   item[i] is the m*k shardview corresponding to the portion on node/worker i


@numba.njit(cache=True)
def shardview(size, index_start=None, base_offset=None, axis_map=None, steps=None, dummy=ramba_dummy_index):
    m = 4 + (
        len(size) + (len(size) if base_offset is None else len(base_offset))
    ) // len(size)
    sv = np.zeros((m, len(size)), dtype=ramba_dist_dtype)
    if np.all(size > 0):
        sv[0] = size.astype(ramba_dist_dtype)
    # sv[1] = i_s
    # sv[2] = a_m
    if index_start is not None:
        sv[1] = index_start.astype(ramba_dist_dtype)
    sv[2] = (
        np.arange(len(size), dtype=ramba_dist_dtype)
        if axis_map is None
        else axis_map.astype(ramba_dist_dtype)
    )
    sv[3] = (
        np.ones(len(size), dtype=ramba_dist_dtype)
        if steps is None
        else steps.astype(ramba_dist_dtype)
    )
    sv[4, 0] = len(size) if base_offset is None else len(base_offset)
    if base_offset is not None:
        _base_offset(sv)[:] = base_offset.astype(ramba_dist_dtype)
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
    return s[0] + s[1]


@numba.njit(cache=True)
def _axis_map(s):
    return s[2]

@numba.njit(cache=True)
def _steps(s):
    return s[3]


@numba.njit(cache=True)
def _base_offset(s):
    return s[4:].reshape(-1)[1 : s[4, 0] + 1]


@numba.njit(cache=True)
def len_size(s):
    return s.shape[1]


@numba.njit(cache=True)
def len_base_offset(s):
    return s[4, 0]


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
    return _axis_map(sv)


@numba.njit(cache=True)
def get_stop(sv):
    return _stop(sv)

@numba.njit(cache=True)
def get_steps(sv):
    return _steps(sv)


@numba.njit(cache=True)
def is_eq(sv, other):
    return (
        (_size(sv) == _size(other)).all()
        and (_index_start(sv) == _index_start(other)).all()
        and (_base_offset(sv) == _base_offset(other)).all()
        and (_axis_map(sv) == _axis_map(other)).all()
        and (_steps(sv) == _steps(other)).all()
    )


@numba.njit(cache=True)
def is_empty(sv):
    return (_size(sv) == 0).any()


@numba.njit(cache=True)
def is_compat(sv, other):
    return (_size(sv) == _size(other)).all() and (
        _index_start(sv) == _index_start(other)
    ).all()


@numba.njit(cache=True)
def overlaps(sv, other):
    return (
        np.logical_or(
            np.logical_and(_start(sv) <= _start(other), _start(other) < _stop(sv)),
            np.logical_and(_start(other) <= _start(sv), _start(sv) < _stop(other)),
        )
    ).all()


@numba.njit(cache=True)
def contains(sv, other):
    return (not is_empty(other)) and (
        np.logical_and(_start(sv) <= _start(other), _stop(other) <= _stop(sv))
    ).all()


@numba.njit(cache=True)
def clean_range(sv):  # remove offset, axis_map, steps
    return shardview(_size(sv), _index_start(sv))


# This does not apeear to be used;  should update for negative steps if we need to keep
def base_to_index(sv, base):
    assert len(base) == len_base_offset(sv)
    offset = [(base[i] - b) // _steps(sv)[i] for i, b in enumerate(_base_offset(sv))]
    am = _axis_map(sv)
    return tuple(
        [s + (0 if am[i] < 0 else offset[am[i]]) for i, s in enumerate(_size(sv))]
    )


def index_to_base(sv, index, end=False):
    assert len(index) == len_size(sv)
    offset = [index[i] - s for i,s in enumerate(_start(sv))]
    offset = [o if _steps(sv)[i]>0 else max(-1,_size(sv)[i]-1-o) for i,o in enumerate(offset)]
    offset = [o*abs(_steps(sv)[i]) for i,o in enumerate(offset)]
    invmap = -np.ones(len_base_offset(sv), dtype=ramba_dist_dtype)
    for i, v in enumerate(_axis_map(sv)):
        if v >= 0:
            invmap[v] = i
    bcastval = 1 if end else 0
    return tuple(
        [
            bo + (bcastval if invmap[i] < 0 else offset[invmap[i]])
            for i, bo in enumerate(_base_offset(sv))
        ]
    )


@numba.njit(cache=True)
def index_to_base_as_array(sv, index, end=False):
    assert len(index) == len_size(sv)
    offset = [index[i] - s for i,s in enumerate(_start(sv))]
    offset = [o if _steps(sv)[i]>0 else max(-1,_size(sv)[i]-1-o) for i,o in enumerate(offset)]
    offset = [o*abs(_steps(sv)[i]) for i,o in enumerate(offset)]
    #offset = [(index[i] - s)*_steps(sv)[i] for i, s in enumerate(_start(sv))]
    invmap = -np.ones(len_base_offset(sv), dtype=ramba_dist_dtype)
    for i, v in enumerate(_axis_map(sv)):
        if v >= 0:
            invmap[v] = i
    bo = _base_offset(sv)
    bcastval = 1 if end else 0
    return np.array(
        [
            bo[i] + (bcastval if invmap[i] < 0 else offset[invmap[i]])
            for i in range(len(bo))
        ]
    )

def get_base_steps(sv, sl=None):
    assert sl is None or len(sl)==len_size(sv)
    st = np.ones(len_base_offset(sv), dtype=ramba_dist_dtype)
    for i, v in enumerate(_axis_map(sv)):
        if v>=0:
            st[v] = _steps(sv)[i]
            if sl is not None and sl[i].step is not None:
                st[v] *= sl[i].step
    return st


# Need to update for steps?? Yes!
def as_base(sv, pv):
    assert len_size(sv) == len_size(pv)
    s = index_to_base_as_array(sv, _index_start(pv))
    e = index_to_base_as_array(sv, _stop(pv), end=True)
    st = get_base_steps(sv)
    for i,step in enumerate(st):
        if step<0:
            s[i],e[i] = e[i]-step, s[i]+1
        else:
            e[i] = e[i]-step+1
    return shardview(e-s, s, steps=st)

def slice_to_local(sv, sl):
    assert len(sl) == len_size(sv)
    s = index_to_base(sv, [x.start for x in sl])
    e = index_to_base(sv, [x.stop for x in sl], end=True)
    e = [
        x if x != 0 else None for x in e
    ]  # special case to let border computation work with neg offset
    #st = [_steps(sv)[i] * (1 if sl[i].step is None else sl[i].step) for i in range(len(sl))]
    st = get_base_steps(sv, sl)
    return tuple([slice(s[i], None if st[i]<0 and e[i] is not None and e[i]<0 else e[i], st[i]) for i in range(len(s))])


def div_to_local(sv, div):
    s = index_to_base(sv, div[0])
    e = index_to_base(sv, div[1] + 1, end=True)
    e = [
        x if x != 0 else None for x in e
    ]  # special case to let border computation work with neg offset
    # print (div,s,e)
    st = get_base_steps(sv)
    return tuple([slice(s[i], None if st[i]<0 and e[i] is not None and e[i]<0 else e[i], st[i]) for i in range(len(s))])


def to_slice(sv):
    s = _index_start(sv)
    e = s + (_size(sv)-1)*abs(_steps(sv))+1
    st = _steps(sv)
    return tuple([slice(
        s[i] if st[i]>0 else e[i]-1, 
        e[i] if st[i]>=0 else (s[i]-1 if s[i]>0 else None), 
        st[i] ) for i in range(len(s))])

# Note: subtle difference between to_slice and as_slice;  as_slice assumes length = slice end-start
def as_slice(sv):
    s = _index_start(sv)
    e = s + _size(sv)
    st = _steps(sv)
    return tuple([slice(
        s[i] if st[i]>0 else e[i]-1, 
        e[i] if st[i]>=0 else (s[i]-1 if s[i]>0 else None), 
        st[i] ) for i in range(len(s))])


def to_base_slice(sv):
    s = _base_offset(sv)
    e = np.ones(len_base_offset(sv), dtype=ramba_dist_dtype)
    st = np.ones(len_base_offset(sv), dtype=ramba_dist_dtype)
    for i, v in enumerate(_axis_map(sv)):
        if v >= 0:
            e[v] += (_size(sv)[i]-1) * abs(_steps(sv)[i])
            st[v] = _steps(sv)[i]
    e += s
    return tuple([slice(
        s[i] if st[i]>0 else e[i]-1, 
        e[i] if st[i]>=0 else (s[i]-1 if s[i]>0 else None), 
        st[i] ) for i in range(len(s))])


import numba.cpython.unsafe.tuple as UT
import ramba.numba_ext as UTx


@numba.njit(cache=True)
def get_base_slice(sv, arr):
    t = UTx.build_full_slice3_tuple(arr.ndim)
    s = _base_offset(sv)
    e = np.ones(len_base_offset(sv), dtype=ramba_dist_dtype)
    st = np.ones(len_base_offset(sv), dtype=ramba_dist_dtype)
    for i, v in enumerate(_axis_map(sv)):
        if v >= 0:
            e[v] += (_size(sv)[i]-1) * abs(_steps(sv)[i])
            st[v] = _steps(sv)[i]
    e += s
    for i in range(arr.ndim):
        t = UT.tuple_setitem(t, i, slice(
            s[i] if st[i]>0 else e[i]-1, 
            e[i] if st[i]>=0 else (s[i]-1 if s[i]>0 else None), 
            st[i] ) )
    return arr[t]


@numba.njit(cache=True)
def has_index(sv, index):
    s = _index_start(sv)
    e = _stop(sv)
    for i in range(len(index)):
        if s[i] > index[i] or index[i] >= e[i]:
            return False
    return True
    # return (_index_start(sv)<=index).all() and (index<_stop(sv)).all()


@numba.generated_jit(nopython=True,cache=True)
def calc_map_internal(sl_i, sv_s, sv_e, sv_st):
    if isinstance(sl_i,numba.types.SliceType):
        if (sl_i.members==2):
            def impl(sl_i, sv_s, sv_e, sv_st):
                s = min(max(sl_i.start, sv_s), sv_e)
                e = min(max(sl_i.stop, sv_s), sv_e)
                sz = e-s
                si = s-sl_i.start
                st = sv_st
                return s, e-1, sz, si, st
            return impl
        def impl(sl_i, sv_s, sv_e, sv_st):
            if sl_i.step is None:
                s = min(max(sl_i.start, sv_s), sv_e)
                e = min(max(sl_i.stop, sv_s), sv_e)
                sz = e-s
                si = s-sl_i.start
                st = sv_st
            else:
                if sl_i.step>0:
                    s = min(max(sl_i.start, sv_s + (sl_i.start-sv_s)%sl_i.step), sv_e)
                    e = min(max(sl_i.stop - (sl_i.stop-1-sl_i.start)%sl_i.step, sv_s), sv_e)
                    si = max(0,(s-sl_i.start)//sl_i.step)
                else:
                    s = min(max(sl_i.stop + 1 + (sl_i.start-sl_i.stop-1)%abs(sl_i.step), sv_s+(sl_i.start-sv_s)%abs(sl_i.step)), sv_e)
                    e = min(max(sl_i.start+1, sv_s), sv_e)
                    si = max(0,int(np.ceil((e-1-sl_i.start)/sl_i.step)))
                sz = int(np.ceil((e-s)/abs(sl_i.step)))
                st = sv_st*sl_i.step
                e = s + (sz-1)*abs(sl_i.step)+1
            return s, max(s,e-1), sz, si, st
        return impl
    else:
        raise numba.core.errors.TypingError("ERR: slice contains something unexpected!", type(sl_i))

@numba.njit(cache=True)
def mapslice(sv, sl):
    # assert len(sl) == len_size(sv)  # This assert causes compilation failure at i+=1; likely due to Numba bug;  need to revisit
    sv_s = _start(sv)
    sv_e = _stop(sv)
    sv_st = _steps(sv)
    s = np.zeros(len(sl), dtype=ramba_dist_dtype)
    e = np.zeros(len(sl), dtype=ramba_dist_dtype)   # last item index, inclusive
    sz = np.zeros(len(sl), dtype=ramba_dist_dtype)
    si = np.zeros(len(sl), dtype=ramba_dist_dtype)
    st = np.zeros(len(sl), dtype=ramba_dist_dtype)
    i = 0
    for sl_i in literal_unroll(sl):
        s[i], e[i], sz[i], si[i], st[i] = calc_map_internal(sl_i,sv_s[i], sv_e[i], sv_st[i])
        i += 1
    b = index_to_base_as_array(sv,s)
    b2 = index_to_base_as_array(sv,e)
    for i in range(len(b)):
        b[i] = min(b[i], b2[i])
    return shardview(sz, si, b, _axis_map(sv), st)


@numba.njit(cache=True)
def mapsv(sv, sl):
    assert len_size(sl) == len_size(sv)
    sv_s = _start(sv)
    sv_e = _stop(sv)
    sv_st = _steps(sv)
    sl_s = _start(sl)
    sl_e = _stop(sl)
    sl_st = _steps(sl)
    s = np.zeros(len_size(sl), dtype=ramba_dist_dtype)
    e = np.zeros(len_size(sl), dtype=ramba_dist_dtype)   # last item index, inclusive
    si = np.zeros(len_size(sl), dtype=ramba_dist_dtype)
    for i in range(len(sv_s)):
        if sl_st[i]>0:
            s[i] = min(max(sl_s[i], sv_s[i] + (sl_s[i]-sv_s[i])%sl_st[i]), sv_e[i])
            e[i] = min(max(sl_e[i] - (sl_e[i]-1-sl_s[i])%sl_st[i], sv_s[i]), sv_e[i])
            si[i] = max(0,(s[i]-sl_s[i])//sl_st[i])
        else:
            s[i] = min(max(sl_s[i] + (sl_e[i]-sl_s[i]-1)%abs(sl_st[i]), sv_s[i]+(sl_e[i]-sv_s[i]-1)%abs(sl_st[i])), sv_e[i])
            e[i] = min(max(sl_e[i], sv_s[i]), sv_e[i])
            si[i] = max(0,int(np.ceil((e[i]-sl_e[i])/sl_st[i])))
    sz = np.ceil((e-s)/np.abs(sl_st)).astype(ramba_dist_dtype)
    st = sv_st*sl_st
    for i in range(len(e)):
        e[i] = s[i] + max(0,sz[i]-1)*np.abs(sl_st[i])

    b = index_to_base_as_array(sv,s)
    b2 = index_to_base_as_array(sv,e)
    for i in range(len(b)):
        b[i] = min(b[i], b2[i])
    return shardview(sz, si, b, _axis_map(sv), st)


# Don't need to update for steps??
@numba.njit(cache=True)
def intersect(sv, sl):
    assert len_size(sv) == len_size(sl)
    sv_s = _index_start(sv)
    sv_e = _stop(sv)
    sl_s = _index_start(sl)
    sl_e = _stop(sl)
    s = np.array([min(max(sl_s[i], sv_s[i]), sv_e[i]) for i in range(len_size(sl))])
    e = np.array([min(max(sl_e[i], sv_s[i]), sv_e[i]) for i in range(len_size(sl))])
    return shardview(e - s, s, axis_map=_axis_map(sv), base_offset=_base_offset(sv) * 0)
    # return shardview(e-s, s, axis_map=_axis_map(sv))

# Don't need to update for steps??
@numba.njit(cache=True)
def union(sv, sl):
    assert len_size(sv) == len_size(sl)
    sv_s = _index_start(sv)
    sv_e = _stop(sv)
    sl_s = _index_start(sl)
    sl_e = _stop(sl)
    s = np.array([min(sl_s[i], sv_s[i]) for i in range(len_size(sl))])
    e = np.array([max(sl_e[i], sv_e[i]) for i in range(len_size(sl))])
    return shardview(e - s, s, axis_map=_axis_map(sv), base_offset=_base_offset(sv) * 0)

# No need to update for stpes?
# get a view of array (e.g. piece of a bcontainer) based on this shardview
# output is an np array with shape same as shardview size
# Will broadcast along additional dimensions as needed
def array_to_view(sv, arr):
    # sanity check
    # print ("axis map, size arr, offset",sv._axis_map,arr.shape,sv._base_offset)
    # assert(len(arr.shape)==len(sv._base_offset))
    # special case 0 size
    if any([v == 0 for v in arr.shape]) or any([v==0 for v in _size(sv)]):
        return np.zeros(tuple([0] * len_size(sv)))
    # special case do nothing
    if len_size(sv) == len(arr.shape) and all(
        [
            arr.shape[i] == _size(sv)[i] and _axis_map(sv)[i] == i
            for i in range(len_size(sv))
        ]
    ):
        return arr
    # compute slice of array needed for output (and sanity check expected shape of array)
    #   expected size is 1 in any broadcasted dimensions, and no more than shard size in others
    sl = [0] * len_base_offset(sv)
    # sl = [ slice(None) if v==0 else 0 for v in arr.shape ]
    shp = [1] * len_base_offset(sv)
    # shp = [ min(1,v) for v in arr.shape ]
    for i, v in enumerate(_axis_map(sv)):
        if v >= 0:
            sl[v] = slice(None)
            shp[v] = min(_size(sv)[i], arr.shape[v])
    if all([not isinstance(i, slice) for i in sl]):  # fully specified -- will result in a single element
        sl[0] = slice(sl[0],sl[0]+1) # ensure at least one slice
    sl = tuple(sl)
    shp = tuple(shp)
    # print ("axis map, size arr, expected:",sv._axis_map,arr.shape,shp)
    if shp != arr.shape:
        print("ERR", shp, arr.shape)  ## !!!!!!!
    assert shp == arr.shape
    arr2 = arr[sl]
    if not isinstance(arr2, (np.ndarray)):
        print("ERR -- got single element, not array")
        arr2 = np.array([arr2])  # in case we get single element
    # add additonal axes as needed (numpy broadcast)
    newdims = [_size(sv)[i] for i, v in enumerate(_axis_map(sv)) if v < 0]
    if len(newdims) > 0:
        if len_size(sv)>len(newdims):
            newdims = tuple(newdims) + arr2.shape
        else:
            newdims = tuple(newdims)
        arr2 = np.broadcast_to(arr2, newdims)
    # get new mapping from arr2 axes to output axesa
    sortmap = sorted(list(_axis_map(sv)))
    if all(_axis_map(sv) == sortmap):  # nothing more to do
        return arr2
    outmap = []
    for i, v in enumerate(_axis_map(sv)):
        j = sortmap.index(v)
        sortmap[j] = -2
        outmap.append(j)
    # move axes and return
    # print ("outmap", outmap)
    outarr = np.moveaxis(arr2, outmap, [i for i in range(len_size(sv))])
    return outarr


@numba.njit(cache=True)
def to_division(sv):
    ret = np.full((2, len(_index_start(sv))), -1)
    ret[0] = _index_start(sv)
    ret[1] += _stop(sv)
    return ret
    # return np.array([_index_start(sv), _stop(sv) - 1])


@numba.njit(cache=True)
def shape_to_div(shape, dummy=ramba_dummy_index):
    res = np.zeros((2, len(shape)), dtype=ramba_dist_dtype)
    for i in range(len(shape)):
        res[1, i] = shape[i] - 1
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
    d2 = np.empty(
        (dshape[0], first_clean.shape[0], first_clean.shape[1]), dtype=ramba_dist_dtype
    )
    d2[0] = first_clean
    for i, s in enumerate(dist[1:]):
        d2[i + 1] = clean_range(s)
    return d2


# @numba.njit # doesn't work with cache=True
# def get_splits( r, v, s, e ):
#    if len(r)==0:
#        s1 = np.array(s[1:])
#        e1 = np.array(e[1:])
#        v.append( shardview( e1-s1, s1 ) )
#        return
#    r0 = r[0]
#    for i in range(len(r0)-1):
#        get_splits( r[1:], v, s+[r0[i]], e+[r0[i+1]] )


# @numba.njit
# def get_range_splits(s1, s2):
#    assert(len_size(s1)==len_size(s2))
#
#    axis_ranges = [ sorted(set([ _start(s1)[i], _stop(s1)[i], _start(s2)[i], _stop(s2)[i] ])) for i in range(len_size(s1)) ]
#    all_splits = [ s1 ]
#    get_splits( axis_ranges, all_splits, [-1000], [-1000] )
#    all_splits = all_splits[1:]
#    s1_splits = [ s for s in all_splits if contains(s1, s) ]
#    s2_splits = [ s for s in all_splits if contains(s2, s) ]
#    return s1_splits, s2_splits


@numba.njit(cache=True)
def cart_prod(r, shp):
    ndim = len(r)
    t = shp
    for i in range(len(shp)):
        t = UT.tuple_setitem(t, i, len(r[i]))
    t2 = t + (ndim,)
    outarr = np.empty(t2, dtype=np.int64)
    t = shp
    for i in range(len(shp)):
        t = UT.tuple_setitem(t, i, 1)
    for i in range(ndim):
        outarr[..., i] = np.array(r[i]).reshape(UT.tuple_setitem(t, i, len(r[i])))
    return outarr.reshape(-1, ndim)


@numba.njit(cache=True)
def get_splits(r, v, shp):
    r_s = [x[:-1] for x in r]
    r_e = [x[1:] for x in r]
    sl = cart_prod(r_s, shp)
    el = cart_prod(r_e, shp)
    for i in range(len(sl)):
        v.append(shardview(el[i] - sl[i], sl[i]))


@numba.njit(cache=True)  # doesn't work with cache=True
def _get_range_splits_list(svl, shp):
    axis_ranges = [
        sorted(set([x for s in svl for x in [_start(s)[i], _stop(s)[i]]]))
        for i in range(len_size(svl[0]))
    ]
    all_splits = [svl[0]]
    # get_splits( axis_ranges, all_splits, [-1000], [-1000] )
    get_splits(axis_ranges, all_splits, shp)
    all_splits = all_splits[1:]
    return all_splits


def get_range_splits_list(svl):
    return _get_range_splits_list(svl, tuple([0] * svl[0].shape[1]))


@numba.njit(cache=True)
def _get_range_splits(s1, s2, shp):
    all_splits = get_range_splits_list([s1, s2], shp)
    s1_splits = [s for s in all_splits if contains(s1, s)]
    s2_splits = [s for s in all_splits if contains(s2, s)]
    return s1_splits, s2_splits


def get_range_splits(s1, s2):
    return _get_range_splits(s1, s2, tuple([0] * s1.shape[1]))


@numba.njit(cache=True)
def compatible_distributions(d1, d2):
    if not len(d1) == len(d2):
        return False
    # return all([ is_compat(d1[i],d2[i]) for i in range(len(d1))])
    for i in range(len(d1)):
        if not is_compat(d1[i], d2[i]):
            return False
    return True


@numba.njit(cache=True)
def dist_is_eq(d1, d2):
    # return len(d1)==len(d2) and all([ is_eq(d1[i],d2[i]) for i in range(len(d1))])
    if not len(d1) == len(d2):
        return False
    for i in range(len(d1)):
        if not is_eq(d1[i], d2[i]):
            return False
    return True


@numba.njit(cache=True)
def slice_distribution(sl, dist):
    ret = np.empty_like(dist, dtype=ramba_dist_dtype)
    for i in range(dist.shape[0]):
        ret[i] = mapslice(dist[i], sl)
    return ret
    # return np.array([ mapslice(dist[i],sl) for i in range(dist.shape[0]) ])


@numba.njit(cache=True)
def find_index(dist, index):
    for i in range(len(dist)):
        if has_index(dist[i], index):
            return i
    return None


@numba.njit(cache=True)
def get_overlaps(k, dist1, dist2):
    return [
        i
        for i in range(dist1.shape[0])
        if overlaps(dist1[i], dist2[k]) or overlaps(dist1[k], dist2[i])
    ]
    # return [i  for i in range(dist1.shape[0])if (not is_empty(intersect(dist1[i],dist2[k]))) or (not is_empty(intersect(dist1[k],dist2[i])))]


@numba.njit(cache=True)
def divisions_to_distribution(divs, base_offset=None, axis_map=None, dummy=ramba_dummy_index):
    # dprint(4,"Divisions to convert:", divs, divs.shape, type(divs))
    divshape = divs.shape
    if base_offset is None:
        svl = [
            shardview(
                size=divs[i][1] - divs[i][0] + 1,
                index_start=divs[i][0],
                base_offset=None,
                axis_map=axis_map,
            )
            for i in range(divshape[0])
        ]
    else:
        svl = [
            shardview(
                size=divs[i][1] - divs[i][0] + 1,
                index_start=divs[i][0],
                base_offset=base_offset[i],
                axis_map=axis_map,
            )
            for i in range(divshape[0])
        ]
    ret = np.empty(
        (divshape[0], svl[0].shape[0], svl[0].shape[1]), dtype=ramba_dist_dtype
    )
    for i, sv in enumerate(svl):
        ret[i] = sv
    return ret


def division_to_shape(divs):
    assert isinstance(divs, np.ndarray)
    divshape = divs.shape
    assert len(divshape) == 2
    return tuple(divs[1, :] - divs[0, :] + 1)


def div_to_factors(divs):
    dshape = divs.shape
    res = []
    for i in range(dshape[2]):
        diffset = set()
        for j in range(dshape[0]):
            diffset.add(divs[j,1,i])
        res.append(len(diffset))
    return res

# @numba.njit(cache=True)
def global_to_divisions(dist):
    ret = np.empty((2, dist.shape[1]), dtype=ramba_dist_dtype)
    ret[0] = _index_start(dist)
    ret[1] = _stop(dist) - 1
    return ret


@numba.njit(cache=True)
def distribution_to_divisions(dist):
    # return np.array([ [_index_start(d), _stop(d)-1] for d in dist ])
    ret = np.empty((dist.shape[0], 2, dist.shape[2]), dtype=ramba_dist_dtype)
    for i, d in enumerate(dist):
        ret[i][0] = _index_start(d)
        ret[i][1] = _stop(d) - 1
    return ret


def default_distribution(size, dims_do_not_distribute=[], dist_dims=None):
    num_dim = len(size)
    if isinstance(dist_dims, int):
        dist_dims = [dist_dims]
    if isinstance(dist_dims, list):
        assert dims_do_not_distribute is None or len(dims_do_not_distribute) == 0
        dims_do_not_distribute = [i for i in range(num_dim) if i not in dist_dims]
    starts = np.zeros(num_dim, dtype=np.int64)
    ends = np.array(list(size), dtype=np.int64)
    # the ends are inclusive, not one past the last index
    ends -= 1
    divisions = np.empty((num_workers, 2, num_dim), dtype=np.int64)
    if do_not_distribute(size):
        make_uni_dist(divisions, 0, starts, ends)
    else:
        compute_regular_schedule(
            size, divisions, dims_do_not_distribute=dims_do_not_distribute
        )
        dprint(3, "compute_regular output:", size, divisions)
    return divisions_to_distribution(divisions)


def block_intersection(a, b):
    ashape = a.shape
    bshape = b.shape
    assert len(ashape) == len(bshape)
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
    # new_dims = len(size) - len(distribution[0]._size)
    dprint(4, "shardview::broadcast", distribution, broadcasted_dims, size, new_dims)
    new_axis_map = np.array(
        [
            -1 if broadcasted_dims[j] else _axis_map(distribution[0])[j - new_dims]
            for j in range(len(size))
        ]
    )
    ret = []
    for i in range(len(distribution)):
        new_size = np.array(
            [
                (size[j] if j<new_dims or _size(distribution[i])[j-new_dims]>0 else 0) if broadcasted_dims[j] else _size(distribution[i])[j - new_dims]
                for j in range(len(size))
            ]
        )
        new_start = np.array(
            [
                0
                if broadcasted_dims[j]
                else _index_start(distribution[i])[j - new_dims]
                for j in range(len(size))
            ]
        )
        new_steps = np.array(
            [
                1
                if broadcasted_dims[j]
                else _steps(distribution[i])[j - new_dims]
                for j in range(len(size))
            ]
        )
        # new_offset = np.array([0 if broadcasted_dims[j] else distribution[i]._base_offset[j - new_dims] for j in range(len(size))])
        # ret.append(shardview(new_size, new_start, new_offset))
        ret.append(
            shardview(new_size, new_start, _base_offset(distribution[i]), new_axis_map, new_steps)
        )
    return np.array(ret)


def remap_axis_result_shape(size, newmap):
    return tuple([size[i] for i in newmap])

# re-orders and/or removes axes.  newmap is a list that specifies which axis maps to the index.  Removes axes if length less than curent shardview dimensionality.  Elements of newmap must be unique, and in range
def remap_axis(size, distribution, newmap):
    old_ndims = len_size(distribution[0])
    old_map = _axis_map(distribution[0])
    assert (
        len(newmap) <= old_ndims
        and all([0 <= v and v < old_ndims for v in newmap])
        and len(newmap) == len(set(newmap))
    )
    new_axis_map = np.array([old_map[i] for i in newmap])
    new_global_size = tuple([size[i] for i in newmap])
    new_dist = []
    for i in range(len(distribution)):
        new_size = np.array([_size(distribution[i])[j] for j in newmap])
        new_start = np.array([_index_start(distribution[i])[j] for j in newmap])
        new_steps = np.array([_steps(distribution[i])[j] for j in newmap])
        new_dist.append(
            shardview(new_size, new_start, _base_offset(distribution[i]), new_axis_map, new_steps)
        )
    return new_global_size, np.array(new_dist)


# creates distribution reduced along a set of axes;  keeps 1 element for each division along each axis for local reductions
def reduce_axes(size, dist, axes):
    return reduce_axes_internal(size, dist, tuple(axes))

# creates distribution reduced along all axes;  keeps 1 element for each division along each axis for local reductions
def reduce_all_axes(size, dist):
    return reduce_axes_internal(size, dist, tuple(range(len(size))))

@numba.njit(cache=True)
def reduce_axes_internal(size, dist, axes):
    rdist = clean_dist(dist)
    bdist = clean_dist(dist)
    rsz = size
    for j in axes:
        divs = list( set([_start(dist[i])[j] for i in range(len(dist)) if not is_empty(dist[i])]) )
        for i in range(len(rdist)):
            if rdist[i, 1, j] in divs:
                rdist[i, 1, j] = divs.index(rdist[i, 1, j])
            rdist[i, 0, j] = 1
            bdist[i, 2, j] = -1
        rsz = UT.tuple_setitem(rsz, j, len(divs))
    return rsz, rdist, bdist


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
    divisions[:, 0, :] = 1
    divisions[:, 1, :] = 0
    # Now put everything on worker 0.
    divisions[node, 0, :] = starts
    divisions[node, 1, :] = ends


def make_uni_dist_from_shape(num_workers, node, shape):
    divisions = np.empty((num_workers, 2, len(shape)), dtype=np.int64)
    starts = np.zeros(len(shape), dtype=np.int64)
    ends = np.array(list(shape), dtype=np.int64)
    # the ends are inclusive, not one past the last index
    ends -= 1
    make_uni_dist(divisions, node, starts, ends)
    return divisions
