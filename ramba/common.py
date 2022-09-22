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
import functools
import math
import operator
#import uuid
import copy as libcopy

distribute_min_size = 100
NUM_WORKERS_FOR_BCAST = 100

non_dist_mode = int(os.environ.get("RAMBA_NON_DIST", "0"))

if non_dist_mode != 0:
    USE_NON_DIST = True
    num_workers = 1
    USE_MPI = False
    default_bcast = "1"
    USE_ZMQ = False
    numa_zones = os.environ.get("RAMBA_NUMA_ZONES", None)  # override detected numa zones
    num_nodes = 1

    def in_driver():
        return True
else:
    USE_NON_DIST = False

if not USE_NON_DIST:
    try:
        from mpi4py import MPI
    
        comm = MPI.COMM_WORLD
        nranks = comm.Get_size()
        rank = comm.Get_rank()
        assert nranks > 1
        USE_MPI = True
        USE_MPI_CW = int(os.environ.get("RAMBA_USE_CW", "0")) != 0
        if USE_MPI_CW:
            if rank == nranks - 1:
                print("Using MPI with", nranks - 1, "workers, 1 driver")
            num_workers = int(os.environ.get("RAMBA_WORKERS", "-1"))
            if num_workers != -1 and rank == 0:
                print("RAMBA_WORKERS setting ignored.")
            num_workers = nranks - 1
        else:
            if rank == 0:
                print("Using MPI with", nranks, "workers")
            num_workers = int(os.environ.get("RAMBA_WORKERS", "-1"))
            if num_workers != -1 and rank == 0:
                print("RAMBA_WORKERS setting ignored.")
            num_workers = nranks
        numa_zones = "DISABLE"  # let MPI handle numa stuff before process starts
        # print ("MPI rank", rank, os.uname()[1])
        USE_ZMQ = int(os.environ.get("RAMBA_USE_ZMQ", "0")) != 0
        default_bcast = None if USE_ZMQ else "1"
    
        # number of threads per worker
        num_threads = int(os.environ.get('RAMBA_NUM_THREADS', '1'))
    
        # number of nodes
        import socket
        nodename = socket.gethostname()
        #print(nodename, rank)
        #allnodes = set(comm.allgather(nodename))
        allnodes = comm.allgather(nodename)
        if USE_MPI_CW:
            allnodes = set(allnodes[:-1])     # Don't include driver process node (in case it is different)
            def in_driver():
                comm = MPI.COMM_WORLD
                rank = comm.Get_rank()
                return rank==num_workers
        else:
            allnodes = set(allnodes)
            def in_driver():
                return False
        #if rank==0: print(allnodes)
        num_nodes = len(allnodes)
    
    
    except:
        USE_MPI = False
        USE_MPI_CW = False
        USE_ZMQ = int(os.environ.get("RAMBA_USE_ZMQ", "1")) != 0
        default_bcast = None


# USE_RAY_CALLS=True
# USE_RAY_CALLS=False
USE_RAY_CALLS = int(os.environ.get("RAMBA_USE_RAY_CALLS", "0")) != 0

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
ndebug = int(os.environ.get("RAMBA_DEBUG", "0"))
if ndebug != 0:
    debug = True
else:
    debug = False

# If RAMBA_SHOW_CODE environment variable is set to non-sero, then display remote code functions run
debug_showcode = int(os.environ.get("RAMBA_SHOW_CODE", "0"))
debug_showcode = debug_showcode!=0

# If RAMBA_RESHAPE_COPY environment variable set to non-zero then reshape calls forward to reshape_copy.
nreshape_forwarding = int(os.environ.get("RAMBA_RESHAPE_COPY", "0"))
if nreshape_forwarding != 0:
    reshape_forwarding = True
else:
    reshape_forwarding = False

# If RAMBA_TIMING environment variable set to True, will print detailed timing messages.
ntiming = int(os.environ.get("RAMBA_TIMING", "0"))
if ntiming != 0:
    timing = True
else:
    timing = False

timing_debug_worker = int(os.environ.get("RAMBA_TIMING_WORKER", "0"))


def tprint(level, *args):
    if ntiming >= level:
        print(*args)
        sys.stdout.flush()


def dprint(level, *args, flush=True):
    if ndebug >= level:
        print(*args)
        if flush:
            sys.stdout.flush()


if not USE_MPI and not USE_NON_DIST:
    num_workers = int(os.environ.get("RAMBA_WORKERS", "4"))  # number of machines
    numa_zones = os.environ.get("RAMBA_NUMA_ZONES", None)  # override detected numa zones

num_threads = int(os.environ.get("RAMBA_NUM_THREADS", "1"))  # number of threads per worker
hint_ip = os.environ.get("RAMBA_IP_HINT", None)  # IP address used to hint which interface to bind queues

if default_bcast is None:
    default_bcast = "1" if num_workers > NUM_WORKERS_FOR_BCAST else "0"
USE_BCAST = int(os.environ.get("RAMBA_USE_BCAST", default_bcast)) != 0


# RAMBA_BIG_DATA environment variable MUST be set to 1 if the application will use arrays
# larger than 2**32 in size.
ramba_big_data = int(os.environ.get("RAMBA_BIG_DATA", "0"))
if ramba_big_data != 0:
    ramba_big_data = True
else:
    ramba_big_data = False


def do_not_distribute(size):
    return np.prod(size) < distribute_min_size


def get_common_state():
    return (
        num_workers,
        num_threads,
        timing,
        ntiming,
        timing_debug_worker,
        ndebug,
        hint_ip,
        numa_zones,
        USE_ZMQ,
        USE_BCAST,
        USE_RAY_CALLS,
        ramba_big_data,
    )


def set_common_state2(st):
    global num_workers
    global num_threads
    global timing
    global ntiming
    global ndebug
    global timing_debug_worker
    global hint_ip
    global numa_zones
    global USE_ZMQ
    global USE_BCAST
    global USE_RAY_CALLS
    global ramba_big_data
    (
        num_workers,
        num_threads,
        timing,
        ntiming,
        timing_debug_worker,
        ndebug,
        hint_ip,
        numa_zones,
        USE_ZMQ,
        USE_BCAST,
        USE_RAY_CALLS,
        ramba_big_data,
    ) = st

def set_common_state(st,gl):
    (
        gl['num_workers'],
        gl['num_threads'],
        gl['timing'],
        gl['ntiming'],
        gl['timing_debug_worker'],
        gl['ndebug'],
        gl['hint_ip'],
        gl['numa_zones'],
        gl['USE_ZMQ'],
        gl['USE_BCAST'],
        gl['USE_RAY_CALLS'],
        gl['ramba_big_data'],
    ) = st
    set_common_state2(st)


# -------------------------------------------------------------------------
# Code to calculate regular divisions of work across an array's dimensions.
# -------------------------------------------------------------------------
def compute_regular_schedule(size, divisions, dims_do_not_distribute=[]):
    divisions[:] = compute_regular_schedule_internal(
        num_workers, size, tuple(dims_do_not_distribute)
    )


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


def create_divisions(divisions, size, best):
    divshape = divisions.shape
    assert(divshape[2] == len(best))
    num_dim = len(size)
    main_divs = [0] * num_dim
    for j in range(num_dim):
        main_divs[j], _, _, _ = get_div_sizes(size[j], best[j])

    index_map = list(range(num_dim))

    def crsi_div(divisions, best, main_divs, index, min_worker, max_worker, size, index_map):
        if index >= len(best):
            return

        mapped_index = index_map[index]
        total_workers = max_worker - min_worker + 1
        chunks_here = total_workers // best[mapped_index]
        last = -1

        dprint(3, "crsi_div:", "best", best, "main_divs", main_divs, "index", index, "min_worker", min_worker, "max_worker", max_worker, "size", size, "chunks_here", chunks_here, "total_workers", total_workers)
        for i in range(min_worker, max_worker + 1, chunks_here):
            num_left = size[mapped_index] - last
            this_div = num_left // ((max_worker + 1 - i) // chunks_here)
            for j in range(chunks_here):
                divisions[i+j,0,mapped_index] = last + 1
                divisions[i+j,1,mapped_index] = last + this_div
                if divisions[i+j,1,mapped_index] > size[mapped_index]:
                    divisions[i+j,1,mapped_index] = size[mapped_index]
            last += this_div
            #last += main_divs[index]
            crsi_div(divisions, best, main_divs, index + 1, i, i + chunks_here - 1, size, index_map)

    crsi_div(divisions, best, main_divs, 0, 0, divshape[0] - 1, np.array(size) - 1, index_map)


def compute_multi_partition(
    num_workers, constraint_dict, flex_array_set, nonflexdist_info, mode="nodesurface"
):
    # flexdist_info is map of array id to (array, divisions, dim_factors)
    flexdist_info = {k:(x, np.empty((num_workers, 2, x.ndim), dtype=np.int64), get_dim_factors(num_workers, x.ndim)) for k,x in flex_array_set.items()}

    non_flex_symbols = {}
    # Get the set of symbols that come from already constructed arrays.
    # Make sure if more than one constructured array has a certain factor
    # associated with a symbol that they all match.
    for aconst in constraint_dict.values():
    #for aconst in constraint.constraint_dict:
        arr = aconst[0]()
        if arr is None:
            continue
        if not arr.bdarray.flex_dist:
            const_array = aconst[1]
            for i, symbol in enumerate(const_array):
                if symbol <= 0:
                    continue
                if symbol in non_flex_symbols:
                    other_div_across_symbol = nonflexdist_info[id(arr)][1][i]
                    if other_div_across_symbol != non_flex_symbols[symbol]:
                        # Two nonflex dist arrays with same symbol but their
                        # dimensions not split the same way.
                        return None
                else:
                    # mapping from symbol to divisions across that dimension
                    non_flex_symbols[symbol] = nonflexdist_info[id(arr)][1][i]

    # Filter out factorizations that split a dimension that some constraint
    # says should not be split.
    # Using the above, filter out partitionings that don't use the same split
    # as an already created array.
    for aconst in constraint_dict.values():
        arr = aconst[0]()
        if arr is None:
            continue
        const_array = aconst[1]
        if id(arr) not in flexdist_info:
            continue
        for i, symbol in enumerate(const_array):
            flexdist_info_for_arr = flexdist_info[id(arr)]
            dprint(3, "prefilter:", id(arr), const_array, flexdist_info_for_arr)
            if symbol == -1:
                flexdist_info[id(arr)] = (arr, aconst[1], list(filter(lambda x: x[i] == 1, flexdist_info_for_arr[2])))
                dprint(3, "postfilter:", id(arr), const_array, flexdist_info[id(arr)])
            elif symbol > 0:
                if symbol in non_flex_symbols:
                    dprint(3, "symbol in non_flex_symbols")
                    flexdist_info[id(arr)] = (arr, aconst[1], list(filter(lambda x: x[i] == non_flex_symbols[symbol], flexdist_info_for_arr[2])))
                    dprint(3, "postfilter:", id(arr), const_array, flexdist_info[id(arr)])

    dprint(3, "values:", flexdist_info.values())
    for v in flexdist_info.values():
        dprint(3, "v:", id(v[0]), v, type(v))
    # If any of array dim_factors are now empty then no solution possible.
    if any([len(v[2]) == 0 for v in flexdist_info.values()]):
        return None

    solutions = set()
    arr_factor_current = []
    arr_factor_solutions = []
    def recursive_eliminator(flexdist_items, current_index, solutions, current_solution, constraint_set, arr_factor_current, arr_factor_solutions):
        if current_index >= len(flexdist_items):
            dprint(3, "adding solution", frozenset(current_solution.items()))
            solutions.add(frozenset(current_solution.items()))
            cur_sum = 0
            result = []
            for test in arr_factor_current:
                best, best_value = compute_regular_schedule_core(num_workers, ((test[1]),), test[0].shape, ())
                cur_sum += best_value
                result.append((test[0], best))
                dprint(3, "test:", test[1], test[0].shape, best_value)
            result.append(cur_sum)
            arr_factor_solutions.append(result)
            return

        arr, _, arr_dim_factors = flexdist_items[current_index]
        dprint(3, "recursive:", current_index, arr, id(arr), arr_dim_factors, constraint_set, current_solution, flexdist_items)
        arr_constraint = constraint_set[id(arr)]
        carr, arr_constraint = arr_constraint
        carr = carr()
        assert carr is arr
        for one_dim_factor in arr_dim_factors:
            new_solution = libcopy.copy(current_solution)
            factors_good = True
            for i, factor in enumerate(one_dim_factor):
                isymbol = arr_constraint[i]
                if isymbol > 0:
                    if isymbol in new_solution:
                        if new_solution[isymbol] != factor:
                            factors_good = False
                            break
                    else:
                        new_solution[isymbol] = factor
            dprint(3, "in arr_dim_factors loop:", one_dim_factor, factors_good)
            if factors_good:
                arr_factor_current.append((arr, one_dim_factor))
                recursive_eliminator(flexdist_items, current_index + 1, solutions, new_solution, constraint_set, arr_factor_current, arr_factor_solutions)
                arr_factor_current.pop()

    recursive_eliminator(list(flexdist_info.values()), 0, solutions, {}, constraint_dict, arr_factor_current, arr_factor_solutions)
    dprint(3, "solutions:", solutions)
    for i in arr_factor_solutions:
        dprint(3, i, i[0], id(i[0]))
    best_solution = min(arr_factor_solutions, key = lambda x: x[-1])
    return best_solution[:-1]


@functools.lru_cache(maxsize=None)
def compute_regular_schedule_core(
    num_workers, the_factors, size, dims_do_not_distribute, mode="nodesurface"
):
    num_dim = len(size)
    divisions = np.empty((num_workers, 2, num_dim), dtype=np.int64)
    best = None
    best_value = math.inf

    largest = [0] * num_dim
    smallest = [0] * num_dim
    dprint(4, "core:", the_factors, type(the_factors))

    for factored in the_factors:
        not_possible = False
        for i in range(num_dim):
            dprint(4, "factored:", factored, type(factored))
            if factored[i] != 1 and i in dims_do_not_distribute:
                not_possible = True
                break
            if factored[i] > size[i]:
                not_possible = True
                break
            if mode == "ratio":
                _, _, largest[i], smallest[i] = get_div_sizes(size[i], factored[i])
        if not_possible:
            continue

        if mode == "ratio":
            ratio = np.prod(largest) / np.prod(smallest)
            if ratio < best_value:
                best_value = ratio
                best = factored
        elif mode == "surface":
            # We are trying to minimize the size of the portions of the blocks that are touching.
            # For 2D, this is the blocks' perimeters.  For 3D, it is surface area, etc.
            # We're really interested in the internal sharing between blocks but we
            # note that I + E = W where I is the interior shared border, E is the
            # exterior unshared border and W is the surface area of all the blocks.
            # We can just compute W since E is a constant given the input array size.
            # Moreover, since all the blocks are about the same, we can just compute
            # for one block and don't have to explicitly count duplicate edges or
            # faces that are symmetric because multiplying by a constant won't change
            # the relative order of the possible partitions.

            surface_area = 0
            # List of the dimensions of each block on average.
            blockSize = [size[i] / factored[i] for i in range(len(factored))]

            for i in range(num_dim):
                # Saves the average length of the block for the current dimension.
                temp = blockSize[i]
                # Ignores the current dimension so that the surface area of the
                # N-dimensional face (edge for 2D, face for 3D, etc.) can be calculated.
                blockSize[i] = 1
                # Adds the calculated size of piece of the block to what has been
                # calculated so far. At the end of the loop, the whole representation
                # of a block surface area will be known.
                surface_area += np.prod(blockSize)
                # Restore the length of the current dimension for future iterations.
                blockSize[i] = temp

            # If the N-dimensional surface area of this factorization is the best
            # thus far then remember it.
            if surface_area < best_value:
                best_value = surface_area
                best = factored
        elif mode == "nodesurface":
            # Here we primarily optimize for the surface area between nodes.
            # Only if there are ties do we look at the intra-node surface.

            surface_area = 0
            # List of the dimensions of each block on average.
            blockSize = [size[i]/factored[i] for i in range(len(factored))]
            create_divisions(divisions, size, factored)
            wpn = workers_per_node()
            #print (3, wpn, divisions)

            for j in range(num_workers):
                for i in range(num_dim):
                    index_to_find = divisions[j, 0, :].copy()
                    index_to_find[i] = divisions[j, 1, i] + 1
                    owning_worker = find_owning_worker(divisions, index_to_find)
                    #print(factored, j, i, owning_worker, index_to_find, owning_worker is not None and j // wpn != owning_worker // wpn)
                    #if owning_worker is not None and j // wpn != owning_worker // wpn:
                    # Add surface area if different node; add smaller fraction if same node
                    if owning_worker is not None:
                        # Saves the average length of the block for the current dimension.
                        temp = blockSize[i]
                        # Ignores the current dimension so that the surface area of the
                        # N-dimensional face (edge for 2D, face for 3D, etc.) can be calculated.
                        blockSize[i] = 1
                        # Adds the calculated size of piece of the block to what has been
                        # calculated so far. At the end of the loop, the whole representation
                        # of a block surface area will be known.
                        if  j // wpn != owning_worker // wpn:
                            surface_area += np.prod(blockSize)
                        else:
                            surface_area += np.prod(blockSize)*0.1
                        # Restore the length of the current dimension for future iterations.
                        blockSize[i] = temp

            # If the N-dimensional surface area of this factorization is the best
            # thus far then remember it.
            if surface_area < best_value:
                best_value = surface_area
                best = factored

    if mode == "surface" or mode == "nodesurface":
        dprint(3, "Best:", best, "Smallest surface area:", best_value)

    assert(best is not None)
    return best, best_value


@functools.lru_cache(maxsize=None)
def compute_regular_schedule_internal(
    num_workers, size, dims_do_not_distribute, mode="nodesurface"
):
    num_dim = len(size)
    divisions = np.empty((num_workers, 2, num_dim), dtype=np.int64)
    # Get the combinations of the prime factorization of the number of workers.
    the_factors = get_dim_factors(num_workers, num_dim)
    best, _ = compute_regular_schedule_core(num_workers, frozenset(the_factors), size, dims_do_not_distribute, mode=mode)
    create_divisions(divisions, size, best)
    return divisions


def find_owning_worker(divisions, index):
    for i in range(divisions.shape[0]):
        if np.all(index >= divisions[i,0,:]) and np.all(index <= divisions[i,1,:]):
            return i
    return None


def exps_to_factor(factors, exps):
    rest = 1
    for i in range(len(factors)):
        rest *= factors[i] ** exps[i]
    return rest


def gen_ind_factor_internal(
    fset, factors, exps, remaining_len, thus_far, index, part_exps
):
    if index >= len(exps):
        rest = exps_to_factor(factors, part_exps)
        new_thus = thus_far + [rest]
        gen_ind_factors(
            fset,
            factors,
            list(map(operator.sub, exps, part_exps)),
            remaining_len - 1,
            new_thus,
        )
    else:
        for i in range(exps[index] + 1):
            part_exps[index] = i
            gen_ind_factor_internal(
                fset, factors, exps, remaining_len, thus_far, index + 1, part_exps
            )


def gen_ind_factors(fset, factors, exps, remaining_len, thus_far):
    if remaining_len == 1:
        rest = exps_to_factor(factors, exps)
        complete_factors = thus_far + [rest]
        fset.add(tuple(complete_factors))
    else:
        gen_ind_factor_internal(
            fset, factors, exps, remaining_len, thus_far, 0, [0] * len(exps)
        )


def gen_dim_factor_dict(factors, exps):
    dim_factor = {1: set(), 2: set(), 3: set(), 4: set()}

    gen_ind_factors(dim_factor[1], factors, exps, 1, [])
    gen_ind_factors(dim_factor[2], factors, exps, 2, [])
    gen_ind_factors(dim_factor[3], factors, exps, 3, [])
    gen_ind_factors(dim_factor[4], factors, exps, 4, [])

    return dim_factor

@functools.lru_cache(maxsize=None)
def get_prime_factors(num_workers):
    return gen_prime_factors(num_workers)

@functools.lru_cache(maxsize=None)
def get_dim_factor_dict(num_workers):
    num_worker_factors, num_worker_exps = get_prime_factors(num_workers)
    dim_factor_dict = gen_dim_factor_dict(num_worker_factors, num_worker_exps)
    return dim_factor_dict

def get_dim_factors(num_workers, num_dim):
    dfdict = get_dim_factor_dict(num_workers)
    if num_dim not in dfdict:
        dfdict[num_dim] = set()
        num_worker_factors, num_worker_exps = get_prime_factors(num_workers)
        gen_ind_factors(dfdict[num_dim], num_worker_factors, num_worker_exps, num_dim, [])
    return dfdict[num_dim]

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


if not USE_MPI and not USE_NON_DIST:
    import ray

    #num_workers = int(os.environ.get('RAMBA_WORKERS', "4")) # number of machines
    num_workers = int(os.environ.get('RAMBA_WORKERS', "-1")) # number of machines
    # number of threads per worker
    num_threads = int(os.environ.get('RAMBA_NUM_THREADS', '-1'))
    num_nodes = 0

    def ray_init():
        if ray.is_initialized():
            return False

        ray_address = os.getenv("ray_address")
        ray_redis_pass = os.getenv("ray_redis_password")
        if ray_address==None:
            ray_address = "auto"
        if ray_redis_pass==None:
            ray_redis_pass = ""
        try:
            ray.init(address=ray_address, _redis_password=ray_redis_pass)
        except:
            print("Failed to connect to existing cluster; starting local Ray")
            import uuid
            ray.init(_redis_password=str(uuid.uuid4()))
        assert ray.is_initialized() == True
        import time
        time.sleep(1)

        global num_workers
        global num_threads
        global num_nodes

        cores = int(ray.available_resources()['CPU'])
        num_nodes = len(list(filter(lambda x: x.startswith("node"), ray.available_resources().keys())))
        dprint(2, "Ray initialized;  available cores:", cores, num_nodes)
        # Default to use all cores.
        if num_workers == -1:
            if num_threads == -1:
                cores_per_node = cores // num_nodes
                core_factors = get_dim_factors(cores_per_node, 2)
                dprint(2, "core_factors:", core_factors)
                closest = 0
                for core_factor in core_factors:
                    total_workers = core_factor[0] * num_nodes
                    if total_workers > closest and total_workers < 50:
                        closest = total_workers
                        num_workers = total_workers
                        num_threads = core_factor[1]
            else:
                num_workers = cores // num_threads
        elif num_threads == -1:
            num_threads = cores // num_workers
        else:
            # If user manually gave more workers than there are cores,
            # then reduce to only use the max core count.
            if num_workers > cores:
                num_workers = cores
            if num_workers * num_threads > cores:
                print("Ramba oversubscription: {} workers times {} threads per worker exceeds the Ray cluster core count of {}".format(num_workers, num_threads, cores))
        dprint(2, "workers:", num_workers)
        dprint(2, "threads per worker:", num_threads)

        return True

    ray_first_init = ray_init()
    numa_zones = os.environ.get('RAMBA_NUMA_ZONES', None) # override detected numa zones

    def in_driver():
        return ray_first_init

hint_ip = os.environ.get('RAMBA_IP_HINT', None)    # IP address used to hint which interface to bind queues


def workers_per_node():
    return num_workers // num_nodes



