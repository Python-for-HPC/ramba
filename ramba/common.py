
"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import os
import sys
import numpy as np

regular_schedule = True
distribute_min_size = 100
#USE_RAY=True
USE_RAY=False

fast_partial_matmul = True
fast_reduction = True

float32 = np.float32
float64 = np.float64
complex128 = np.complex128
int64 = np.int64

if sys.version_info >= (3, 3):
    from time import perf_counter as timer
else:
    from timeit import default_timer as timer

# If RAMBA_DEBUG environment variable set to True, will print detailed debugging messages.
ndebug = int(os.environ.get('RAMBA_DEBUG', "0"))
if ndebug != 0:
    debug = True
else:
    debug = False

# If RAMBA_RESHAPE_COPY environment variable set to non-zero then reshape calls forward to reshape_copy.
nreshape_forwarding = int(os.environ.get('RAMBA_RESHAPE_COPY', "0"))
if nreshape_forwarding != 0:
    reshape_forwarding = True
else:
    reshape_forwarding = False

# If RAMBA_TIMING environment variable set to True, will print detailed timing messages.
ntiming = int(os.environ.get('RAMBA_TIMING', "0"))
if ntiming != 0:
    timing = True
else:
    timing = False

timing_debug_worker = int(os.environ.get('RAMBA_TIMING_WORKER',"0"))


def tprint(level, *args):
    if ntiming >= level:
        print(*args)
        sys.stdout.flush()

def dprint(level, *args):
    if ndebug >= level:
        print(*args)
        sys.stdout.flush()

num_workers = int(os.environ.get('RAMBA_WORKERS', "4")) # number of machines
num_threads = int(os.environ.get('RAMBA_NUM_THREADS', '1')) # number of threads per worker

# RAMBA_BIG_DATA environment variable MUST be set to 1 if the application will use arrays
# larger than 2**32 in size.
ramba_big_data = int(os.environ.get('RAMBA_BIG_DATA', "0"))
if ramba_big_data != 0:
    ramba_big_data = True
else:
    ramba_big_data = False

def do_not_distribute(size):
    return np.prod(size) < distribute_min_size

def get_common_state():
    return (num_workers, num_threads, timing, ntiming, timing_debug_worker, ndebug)

def set_common_state(st):
    global num_workers
    global num_threads
    global timing
    global ntiming
    global ndebug
    num_workers, num_threads, timing, ntiming, timing_debug_worker, ndebug = st

