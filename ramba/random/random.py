"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import ramba
import numpy as np
import numba


def seed(x):
    ramba.remote_exec_all("seed", x)


class Generator:
    def random(size=None, dtype=np.float64, out=None, **kwargs):
        if size is None:
            rng = np.random.default_rng()
            assert out is None
            return rng.random(size, dtype=dtype, out=out)
        else:
            kwargs["dtype"] = dtype
            # We don't handle this case yet.
            assert out is None
            # def impl(bcontainer, dim_lens, starts):
            #    rng = np.random.default_rng()
            #    rng.random(size=dim_lens, out=bcontainer)
            # return ramba.init_array(size, ramba.Filler(impl, mode=ramba.Filler.WHOLE_ARRAY_INPLACE, do_compile=False), **kwargs)
            def impl(bcontainer, dim_lens, starts):
                for i in numba.pndindex(dim_lens):
                    bcontainer[i] = np.random.random()

            return ramba.init_array(
                size,
                ramba.Filler(
                    impl, mode=ramba.Filler.WHOLE_ARRAY_INPLACE, do_compile=True
                ),
                **kwargs
            )

    def normal(loc=0.0, scale=1.0, size=None, dtype=np.float64, out=None, **kwargs):
        if size is None:
            rng = np.random.default_rng()
            assert out is None
            return rng.normal(loc=loc, scale=scale, size=size)
        else:
            kwargs["dtype"] = dtype
            # We don't handle this case yet.
            assert out is None

            def impl(dim_lens, starts):
                rng = np.random.default_rng()
                return rng.normal(loc=loc, scale=scale, size=dim_lens)

            return ramba.init_array(
                size,
                ramba.Filler(impl, mode=ramba.Filler.WHOLE_ARRAY_NEW, do_compile=False),
                **kwargs
            )


def default_rng():
    return Generator()


def random(size=None, **kwargs):
    if size is None:
        return np.random.random()
    else:
        # def impl(dim_lens, starts):
        #    return np.random.random(dim_lens)
        # return ramba.init_array(size, ramba.Filler(impl, mode=ramba.Filler.WHOLE_ARRAY_NEW, do_compile=True), **kwargs)
        def impl(bcontainer, dim_lens, starts):
            for i in numba.pndindex(dim_lens):
                bcontainer[i] = np.random.random()

        return ramba.init_array(
            size,
            ramba.Filler(impl, mode=ramba.Filler.WHOLE_ARRAY_INPLACE, do_compile=True),
            **kwargs
        )


def randn(*args, **kwargs):
    if len(args) == 0:
        return np.random.randn()
    else:
        # def impl(dim_lens, starts):
        #    return np.random.randn(*dim_lens)
        # return ramba.init_array(args, ramba.Filler(impl, mode=ramba.Filler.WHOLE_ARRAY_NEW, do_compile=False), **kwargs)
        ##return ramba.init_array(args, ramba.Filler(impl, per_element=False, do_compile=True), **kwargs)
        def impl(bcontainer, dim_lens, starts):
            for i in numba.pndindex(dim_lens):
                bcontainer[i] = np.random.randn()

        return ramba.init_array(
            args,
            ramba.Filler(impl, mode=ramba.Filler.WHOLE_ARRAY_INPLACE, do_compile=True),
            **kwargs
        )


def uniform(low=0.0, high=1.0, size=None, **kwargs):
    if size is None:
        return np.random.uniform(low=low, high=high)
    else:
        # def impl(bcontainer, dim_lens, starts):
        #    bcontainer[:] = np.random.uniform(low, high, size=dim_lens)
        def impl(bcontainer, dim_lens, starts):
            for i in numba.pndindex(dim_lens):
                bcontainer[i] = np.random.uniform(low, high)

        return ramba.init_array(
            size,
            ramba.Filler(impl, mode=ramba.Filler.WHOLE_ARRAY_INPLACE, do_compile=True),
            **kwargs
        )


class RandomState:
    def __init__(self, *args):
        self.args = args

    def __getattr__(self, attr):
        def ramba_rs_attr(*args, **kwargs):
            if "size" in kwargs:
                ramba.dprint(1, "RandomState generate distributed array")
                rs = np.random.RandomState()
                size = kwargs["size"]
                del kwargs["size"]

                def impl(bcontainer, dim_lens, starts):
                    bcontainer[:] = getattr(rs, attr)(*args, size=dim_lens)

                return ramba.init_array(
                    size,
                    ramba.Filler(
                        impl, mode=ramba.Filler.WHOLE_ARRAY_INPLACE, do_compile=False
                    ),
                )
            else:
                return getattr(np.random.RandomState(), attr)(*args, **kwargs)

        return ramba_rs_attr
