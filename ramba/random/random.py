"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import ramba
import numpy as np

def seed(x):
    [ramba.remote_states[i].seed.remote(x+i) for i in range(len(ramba.remote_states))]

def random(size):
    return ramba.init_array(size, lambda x: np.random.random())

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
                return ramba.init_array(size, lambda x: getattr(rs, attr)(*args, **kwargs))
            else:
                return getattr(np.random.RandomState(), attr)(*args, **kwargs)

        return ramba_rs_attr
