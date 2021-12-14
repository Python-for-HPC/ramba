"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import numba
import functools
import numpy as np
import math
import operator
from ramba.common import *

# from numba import int64
# from numba.experimental import jitclass

# spec = [
#        ('size', int64[:]),
#        ('index_start', int64[:]),
#        ('base_offset', int64[:]),
#        ('axis_map', int64[:])
#    ]


# simple class to specify a shard or view
# Here size is the k-dimensional size of the view/shard
#   index_start specifies the beginning of the global index range for this shard/view portion
#   base_offset is the start position in the base container
#   axis_map maps view axes to base container axes (or -1 for broadcasted axes)
#   Note: size, index_start, and axis_map are of length k; base_offset may have some other length
# A distribution is a list of shards -- one element per remote worker, covering the range of the dist. array
# @jitclass(spec)
class shardview:
    def __init__(self, size, index_start=None, base_offset=None, axis_map=None):
        # def __init__(self, size, index_start, base_offset, axis_map):
        if np.any(size < 1):
            size = size * 0
        self._size = size
        self._index_start = size * 0 if index_start is None else index_start
        self._base_offset = size * 0 if base_offset is None else base_offset
        self._axis_map = np.arange(size.shape[0]) if axis_map is None else axis_map
        # assert((len(self._size)==len(self._index_start)) and (len(self._size)==len(self._base_offset)))
        assert (len(self._size) == len(self._index_start)) and (
            len(self._size) == len(self._axis_map)
        )

    def __repr__(self):
        return (
            "[start="
            + repr(self._index_start)
            + " size="
            + repr(self._size)
            + " base_offset="
            + repr(self._base_offset)
            + " axis_map="
            + repr(self._axis_map)
            + "]"
        )

    # def __getstate__(self):
    #    return {'index_start':self._index_start, 'size':self._size, 'base_offset':self._base_offset, 'axis_map':self._axis_map}

    def __eq__(self, other):
        raise Exception("Should use is_eq or dist_is_eq")

    @property
    def _stop(self):
        return self._index_start + self._size

    @property
    def _start(self):
        return self._index_start


def get_start(sv):
    return sv._start


def get_size(sv):
    return sv._size


def get_base_offset(sv):
    return sv._base_offset


def get_axis_map(sv):
    return sv._axis_map


def get_stop(sv):
    return sv._stop


def is_eq(sv, other):
    if isinstance(other, shardview):
        return (
            all(sv._size == other._size)
            and all(sv._index_start == other._index_start)
            and all(sv._base_offset == other._base_offset)
            and all(sv._axis_map == other._axis_map)
        )
    return False


def is_empty(sv):
    return any(sv._size == 0)


def is_compat(sv, other):
    if isinstance(other, shardview):
        return all(sv._size == other._size) and all(
            sv._index_start == other._index_start
        )
    return False


def overlaps(sv, other):
    return all(
        np.logical_or(
            np.logical_and(sv._start <= other._start, other._start < sv._stop),
            np.logical_and(other._start <= sv._start, sv._start < other._stop),
        )
    )


def contains(sv, other):
    return (not is_empty(other)) and all(
        np.logical_and(sv._start <= other._start, other._stop <= sv._stop)
    )


def clean_range(sv):  # remove offset, axis_map
    return shardview(sv._size, sv._index_start)


def base_to_index(sv, base):
    assert len(base) == len(sv._base_offset)
    offset = [base[i] - sv._base_offset[i] for i in range(len(sv._base_offset))]
    return tuple(
        [
            sv._index_start[i] + (0 if sv._axis_map[i] < 0 else offset[sv._axis_map[i]])
            for i in range(len(sv._size))
        ]
    )


def index_to_base(sv, index):
    assert len(index) == len(sv._size)
    offset = [index[i] - sv._index_start[i] for i in range(len(sv._size))]
    invmap = [-1] * len(sv._base_offset)
    for i, v in enumerate(sv._axis_map):
        if v >= 0:
            invmap[v] = i
    return tuple(
        [
            sv._base_offset[i] + (0 if invmap[i] < 0 else offset[invmap[i]])
            for i in range(len(sv._base_offset))
        ]
    )


def slice_to_local(sv, sl):
    assert len(sl) == len(sv._size)
    s = index_to_base(sv, [x.start for x in sl])
    e = index_to_base(sv, [x.stop for x in sl])
    e = [
        x if x != 0 else None for x in e
    ]  # special case to let border computation work with neg offset
    return tuple([slice(s[i], e[i]) for i in range(len(s))])


def div_to_local(sv, div):
    s = index_to_base(sv, div[0])
    e = index_to_base(sv, div[1] + 1)
    e = [
        x if x != 0 else None for x in e
    ]  # special case to let border computation work with neg offset
    # print (div,s,e)
    return tuple([slice(s[i], e[i]) for i in range(len(s))])


def to_slice(sv):
    s = sv._index_start
    e = s + sv._size
    return tuple([slice(s[i], e[i]) for i in range(len(s))])


def to_base_slice(sv):
    s = sv._base_offset
    e = np.ones(len(sv._base_offset), dtype=int)
    for i, v in enumerate(sv._axis_map):
        if v >= 0:
            e[v] = sv._size[i]
    e += s
    return tuple([slice(s[i], e[i]) for i in range(len(s))])


def has_index(sv, index):
    return (sv._index_start <= index).all() and (
        index < (sv._index_start + sv._size)
    ).all()


def mapslice(sv, sl):
    assert len(sl) == len(sv._size)
    s = np.array(
        [
            min(max(sl[i].start, sv._index_start[i]), sv._index_start[i] + sv._size[i])
            for i in range(len(sl))
        ]
    )
    e = np.array(
        [
            min(max(sl[i].stop, sv._index_start[i]), sv._index_start[i] + sv._size[i])
            for i in range(len(sl))
        ]
    )
    si = s - np.array([sd.start for sd in sl])
    return shardview(e - s, si, np.array(index_to_base(sv, s)), sv._axis_map)


def intersect(sv, sl):
    assert len(sl._size) == len(sv._size)
    s = np.array(
        [
            min(max(sl._index_start[i], sv._index_start[i]), sv._stop[i])
            for i in range(len(sl._size))
        ]
    )
    e = np.array(
        [
            min(max(sl._stop[i], sv._index_start[i]), sv._stop[i])
            for i in range(len(sl._size))
        ]
    )
    return shardview(e - s, s, axis_map=sv._axis_map)


# get a view of array (e.g. piece of a bcontainer) based on this shardview
# output is an np array with shape same as shardview size
# Will broadcast along additional dimensions as needed
def array_to_view(sv, arr):
    # sanity check
    # print ("axis map, size arr, offset",sv._axis_map,arr.shape,sv._base_offset)
    # assert(len(arr.shape)==len(sv._base_offset))
    # special case 0 size
    if any([v == 0 for v in arr.shape]):
        return np.zeros(tuple([0] * len(sv._size)))
    # special case do nothing
    if (
        len(sv._size) == len(arr.shape)
        and len(sv._axis_map) == len(sv._size)
        and all(
            [
                arr.shape[i] == sv._size[i] and sv._axis_map[i] == i
                for i in range(len(sv._size))
            ]
        )
    ):
        return arr
    # compute slice of array needed for output (and sanity check expected shape of array)
    #   expected size is 1 in any broadcasted dimensions, and no more than shard size in others
    sl = [0] * len(sv._base_offset)
    # sl = [ slice(None) if v==0 else 0 for v in arr.shape ]
    shp = [1] * len(sv._base_offset)
    # shp = [ min(1,v) for v in arr.shape ]
    for i, v in enumerate(sv._axis_map):
        if v >= 0:
            sl[v] = slice(None)
            shp[v] = min(sv._size[i], arr.shape[v])
    sl = tuple(sl)
    shp = tuple(shp)
    # print ("axis map, size arr, expected:",sv._axis_map,arr.shape,shp)
    assert shp == arr.shape
    arr2 = arr[sl]
    if not isinstance(arr2, (np.ndarray)):
        arr2 = np.array([arr2])  # in case we get single element
    # add additonal axes as needed (numpy broadcast)
    newdims = [sv._size[i] for i, v in enumerate(sv._axis_map) if v < 0]
    if len(newdims) > 0:
        newdims = tuple(newdims) + arr2.shape
        arr2 = np.broadcast_to(arr2, newdims)
    # get new mapping from arr2 axes to output axesa
    sortmap = sorted(list(sv._axis_map))
    if all(sv._axis_map == sortmap):  # nothing more to do
        return arr2
    outmap = []
    for i, v in enumerate(sv._axis_map):
        j = sortmap.index(v)
        sortmap[j] = -2
        outmap.append(j)
    # move axes and return
    # print ("outmap", outmap)
    outarr = np.moveaxis(arr2, outmap, [i for i in range(len(sv._axis_map))])
    return outarr


def to_division(sv):
    return np.array([sv._index_start, sv._index_start + sv._size - 1])


def shape_to_div(shape):
    res = np.zeros((2, len(shape)))
    for i in range(len(shape)):
        res[1, i] = shape[i] - 1
    return res


# still need?
def slice_to_fortran(sl):
    ret = list(sl)
    ret.reverse()
    return tuple(ret)


def clean_dist(dist):
    return [clean_range(s) for s in dist]


def get_range_splits(s1, s2):
    assert len(s1._size) == len(s2._size)

    def get_splits(r, v, s=[], e=[]):
        if len(r) == 0:
            s = np.array(s)
            e = np.array(e)
            v.append(shardview(e - s, s))
            return
        r0 = r[0]
        for i in range(len(r0) - 1):
            get_splits(r[1:], v, s + [r0[i]], e + [r0[i + 1]])

    axis_ranges = [
        sorted(set([s1._start[i], s1._stop[i], s2._start[i], s2._stop[i]]))
        for i in range(len(s1._start))
    ]
    all_splits = []
    get_splits(axis_ranges, all_splits)
    s1_splits = [s for s in all_splits if contains(s1, s)]
    s2_splits = [s for s in all_splits if contains(s2, s)]
    return s1_splits, s2_splits


def compatible_distributions(d1, d2):
    assert len(d1) == len(d2)
    return all([is_compat(d1[i], d2[i]) for i in range(len(d1))])


def dist_is_eq(d1, d2):
    return len(d1) == len(d2) and all([is_eq(d1[i], d2[i]) for i in range(len(d1))])


def slice_distribution(sl, dist):
    return [mapslice(d, sl) for d in dist]


def find_index(dist, index):
    for i in range(len(dist)):
        if has_index(dist[i], index):
            return i
    return None


def divisions_to_distribution(divs, base_offset=None, axis_map=None):
    dprint(4, "Divisions to convert:", divs, divs.shape, type(divs))
    divshape = divs.shape
    return [
        shardview(
            size=divs[i][1] - divs[i][0] + 1,
            index_start=divs[i][0],
            base_offset=base_offset[i] if base_offset is not None else None,
            axis_map=axis_map,
        )
        for i in range(divshape[0])
    ]


def division_to_shape(divs):
    assert isinstance(divs, np.ndarray)
    divshape = divs.shape
    assert len(divshape) == 2
    return tuple(divs[1, :] - divs[0, :] + 1)


def distribution_to_divisions(dist):
    return np.array([[d._index_start, d._index_start + d._size - 1] for d in dist])


def default_distribution(size):
    num_dim = len(size)
    starts = np.zeros(num_dim, dtype=np.int64)
    ends = np.array(list(size), dtype=np.int64)
    # the ends are inclusive, not one past the last index
    ends -= 1
    divisions = np.empty((num_workers, 2, num_dim), dtype=np.int64)
    if do_not_distribute(size):
        make_uni_dist(divisions, 0, starts, ends)
    else:
        compute_regular_schedule(size, divisions)
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
    new_dims = len(size) - len(distribution[0]._size)
    dprint(4, "shardview::broadcast", distribution, broadcasted_dims, size, new_dims)
    new_axis_map = np.array(
        [
            -1 if broadcasted_dims[j] else distribution[0]._axis_map[j - new_dims]
            for j in range(len(size))
        ]
    )
    ret = []
    for i in range(len(distribution)):
        new_size = np.array(
            [
                size[j] if broadcasted_dims[j] else distribution[i]._size[j - new_dims]
                for j in range(len(size))
            ]
        )
        new_start = np.array(
            [
                0 if broadcasted_dims[j] else distribution[i]._index_start[j - new_dims]
                for j in range(len(size))
            ]
        )
        # new_offset = np.array([0 if broadcasted_dims[j] else distribution[i]._base_offset[j - new_dims] for j in range(len(size))])
        # ret.append(shardview(new_size, new_start, new_offset))
        ret.append(
            shardview(new_size, new_start, distribution[i]._base_offset, new_axis_map)
        )
    return ret


# re-orders and/or removes axes.  newmap is a list that specifies which axis maps to the index.  Removes axes if lenght less than curent shardview dimensionaily.  Elements of newmap must be unique, and in range
def remap_axis(size, distribution, newmap):
    old_ndims = len(distribution[0]._size)
    old_map = distribution[0]._axis_map
    assert (
        len(newmap) <= old_ndims
        and all([0 <= v and v < old_ndims for v in newmap])
        and len(newmap) == len(set(newmap))
    )
    new_axis_map = np.array([old_map[i] for i in newmap])
    new_global_size = tuple([size[i] for i in newmap])
    new_dist = []
    for i in range(len(distribution)):
        new_size = np.array([distribution[i]._size[j] for j in newmap])
        new_start = np.array([distribution[i]._index_start[j] for j in newmap])
        new_dist.append(
            shardview(new_size, new_start, distribution[i]._base_offset, new_axis_map)
        )
    return new_global_size, new_dist


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
