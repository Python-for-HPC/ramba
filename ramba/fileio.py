"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# import multiprocessing as mp
# import threading
import numpy as np


loaders = {}


class loader:
    def __init__(self, ftype, is_dist, aliases=[]):
        self.ftype = ftype
        self.is_dist = is_dist
        self.getinfo = globals()[ftype + "_getinfo"]
        self.read = globals()[ftype + "_read"]
        self.readall = globals()[ftype + "_readall"]
        loaders[ftype] = self
        for i in aliases:
            loaders[i] = self


def get_load_handler(fname, ftype=None):
    if ftype is None:
        ftype = fname.split("/")[-1].split(".")[-1].lower()
    if ftype not in loaders:
        print("Error:  unknown file type", ftype)
        ftype = "hdf5"
    return loaders[ftype]


try:
    import h5py

    def hdf5_get_part(fname, nm):
        f = h5py.File(fname, "r")
        if nm is None or nm == "":
            return f
        l = nm.split("/")
        p = f
        for i in l:
            p = p[i]
        return p

    def hdf5_getinfo(fname, arr_path):
        p = hdf5_get_part(fname, arr_path)
        return p.shape, p.dtype

    def hdf5_read(fname, arr, src_index, arr_path, dst_index=None):
        p = hdf5_get_part(fname, arr_path)
        print("hdf5 read direct", p.shape, src_index)
        if dst_index is None:
            p.read_direct(arr, src_index)
        else:
            p.read_direct(arr, src_index, dst_index)

    def hdf5_readall(fname, arr_path):
        p = hdf5_get_part(fname, arr_path)
        return p[:]

    loader("hdf5", True, aliases=["h5ad"])
except:
    print("No HDF5 support")

try:
    from PIL import Image

    # TODO: should handle frames, depths other than 8-bit
    def pil_getinfo(fname):
        img = Image.open(fname)
        h = img.height
        w = img.width
        c = len(img.getbands())
        shape = (c, h, w)
        dtype = np.uint8
        return shape, dtype

    def pil_read():  # distributed, partial loads not supported
        pass

    def pil_readall(fname):
        img = Image.open(fname)
        arr = np.array(img)
        if arr.ndim > 2:
            arr = np.transpose(arr, (2, 0, 1))  # convert from HxWxC to CxHxW
        return arr

    loader("pil", False, aliases=["jpg", "jpeg", "png", "tif", "tiff"])

except:
    print("No PIL support")
