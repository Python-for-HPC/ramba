"""
Copyright 2021 Intel Corporation

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

from ramba.common import *

import os

if USE_MPI:
    import mpi4py
else:
    import ray
import numba
import types
import inspect
import tokenize
from io import BytesIO
import sys
import parser
import uuid
import weakref
import operator
import copy as libcopy
import pickle as pickle

if pickle.HIGHEST_PROTOCOL < 5:
    import pickle5 as pickle
import cloudpickle
import threading

if USE_ZMQ:
    import ramba.ramba_queue_zmq as ramba_queue
elif USE_MPI:
    import ramba.ramba_queue_mpi as ramba_queue
else:
    import ramba.ramba_queue as ramba_queue
# import ramba.shardview as shardview
import ramba.shardview_array as shardview
import ramba.numa as numa
import asyncio
import functools
import math
import traceback
import numba.cpython.unsafe.tuple as UT
import atexit
import ramba.random as random
import socket
import numbers

import ramba.fileio as fileio

try:
    import dpctl

    dpctl_present = True
except Exception:
    dpctl_present = False

dprint(3, "dpctl present =", dpctl_present)

if dpctl_present:
    if dpctl.has_gpu_queues():
        gpu_present = True
    else:
        gpu_present = False
else:
    gpu_present = False

dprint(3, "Ramba will use GPU =", gpu_present)

if ndebug >= 3 and gpu_present:
    os.environ["SYCL_PI_TRACE"] = "-1"
    pass

if ndebug >= 5:
    os.environ["NUMBA_DEBUG_ARRAY_OPT"] = "2"

import numpy as np

if gpu_present:
    import dpctl.dptensor.numpy_usm_shared as gnp
else:
    import numpy as gnp

from numba import prange

# Get a reference to the current module to be used for dynamic code generation.
ramba_module = __import__(__name__)
dprint(3, "ramba_module:", ramba_module, type(ramba_module))


if not USE_MPI:
    reexport = ["get", "put", "wait"]
    istmt = "from ray import {}".format(",".join(reexport))
    exec(istmt)


class Filler:
    PER_ELEMENT = 0
    WHOLE_ARRAY_NEW = 1
    WHOLE_ARRAY_INPLACE = 2

    def __init__(self, func, mode=PER_ELEMENT, do_compile=False):
        self.func = func
        self.mode = mode
        self.do_compile = do_compile


class FillerFunc:
    def __init__(self, func):
        self.func = func

    def __hash__(self):
        if isinstance(self.func, numba.core.registry.CPUDispatcher):
            ret = hash(self.func)
        else:
            ret = hash(self.func.__code__.co_code)

        return ret

    def __eq__(self, other):
        return (self.func.__code__.co_code == other.func.__code__.co_code and
                self.func.__code__.co_argcount == other.func.__code__.co_argcount and
                self.func.__code__.co_consts == other.func.__code__.co_consts and
                self.func.__code__.co_freevars == other.func.__code__.co_freevars and
                self.func.__code__.co_names == other.func.__code__.co_names)


@functools.lru_cache()
def get_fm(func: FillerFunc, parallel):
    real_func = func.func
    dprint(2, "get_fm for function", real_func.__name__, "parallel =", parallel)
    assert isinstance(real_func, types.FunctionType)
    return numba.njit(parallel=parallel)(real_func)


class FunctionMetadata:
    def __init__(self, func, dargs, dkwargs, no_global_cache=False, parallel=True):
        self.func = func
        self.dargs = dargs
        self.dkwargs = dkwargs
        self.numba_args = dkwargs.get("numba", {})
        dprint(2, "FunctionMetadata __init__ numba_args:", self.numba_args, id(self))
        self.numba_func = None
        self.numba_pfunc = None
        self.nfunc = {}
        self.npfunc = {}
        self.ngfunc = {}
        self.no_global_cache = no_global_cache
        self.parallel = parallel
        dprint(2, "FunctionMetadata finished")

    def __call__(self, *args, **kwargs):
        """Get the types of the arguments.  See if we attempted to compile that
        set of arguments with Numba parallel=True or Numba parallel=False.  We
        always try Numba parallel=True first and then Numba parallel=False.  If
        either of those ever fail to compile or run then we record that failure
        in npfunc or nfunc so that we don't try the failed case again.  If
        neither of those work then fall back to Python.
        """
        dprint(2, "FunctionMetadata::__call__", self.func.__name__)
        dprint(
            4,
            "FunctionMetadata::__call__ args",
            args,
            kwargs,
            self.numba_args,
        )
        atypes = tuple([type(x) for x in args])
        dprint(2, "types:", atypes)
        try_again = True
        count = 0
        args_for_numba = []
        for arg in args:
            if inspect.isfunction(arg):
                args_for_numba.append(numba.extending.register_jitable(arg))
                #args_for_numba.append(numba.njit(arg))
            else:
                args_for_numba.append(arg)

        if not self.numba_pfunc:
            if len(self.numba_args) == 0 and not self.no_global_cache:
                self.numba_pfunc = get_fm(FillerFunc(self.func), True)
                self.numba_func = get_fm(FillerFunc(self.func), False)
            else:
                self.numba_pfunc = numba.njit(parallel=True, **self.numba_args)(
                    self.func
                )
                self.numba_func = numba.njit(**self.numba_args)(self.func)

        if gpu_present and self.parallel:
            dprint(1, "using gpu context")

            with dpctl.device_context("level0:gpu"):
                while try_again and count < 2:
                    count += 1
                    try_again = False
                    if self.ngfunc.get(atypes, True):
                        try:
                            ret = self.numba_pfunc(*args_for_numba, **kwargs)
                            self.ngfunc[atypes] = True
                            return ret
                        except numba.core.errors.TypingError as te:
                            tetxt = str(te)
                            tesplit = tetxt.splitlines()
                            for teline in tesplit:
                                if (
                                    "Untyped global name" in teline
                                    and "ramba.StencilMetadata" in teline
                                ):
                                    try_again = True
                                    # Name of global that is of type ramba.StencilMetadata
                                    tes = teline[21:].split()[0][:-2]
                                    outer_globals = self.func.__globals__
                                    outer_locals = {}
                                    etes = eval(tes, outer_globals, outer_locals)
                                    etes.compile()  # Converts to a Numba StencilFunc
                                    outer_globals[
                                        tes
                                    ] = (
                                        etes.sfunc
                                    )  # Rewrite the global to the Numba StencilFunc
                                    self.numba_pfunc = numba.njit(
                                        parallel=True, **self.numba_args
                                    )(self.func)
                                    self.numba_func = numba.njit(**self.numba_args)(
                                        self.func
                                    )
                            if not try_again:
                                self.ngfunc[atypes] = False
                                dprint(
                                    1,
                                    "Numba GPU ParallelAccelerator attempt failed for",
                                    self.func,
                                )
                        except Exception:
                            self.ngfunc[atypes] = False
                            dprint(
                                1,
                                "Numba GPU ParallelAccelerator attempt failed for",
                                self.func,
                            )

        while try_again and count < 2 and self.parallel:
        #while num_threads > 1 and try_again and count < 2:
            count += 1
            try_again = False
            if self.npfunc.get(atypes, True):
                try:
                    ret = self.numba_pfunc(*args_for_numba, **kwargs)
                    self.npfunc[atypes] = True
                    return ret
                except numba.core.errors.TypingError as te:
                    tetxt = str(te)
                    tesplit = tetxt.splitlines()
                    for teline in tesplit:
                        if (
                            "Untyped global name" in teline
                            and "ramba.StencilMetadata" in teline
                        ):
                            try_again = True
                            # Name of global that is of type ramba.StencilMetadata
                            tes = teline[21:].split()[0][:-2]
                            outer_globals = self.func.__globals__
                            outer_locals = {}
                            etes = eval(tes, outer_globals, outer_locals)
                            etes.compile()  # Converts to a Numba StencilFunc
                            outer_globals[
                                tes
                            ] = (
                                etes.sfunc
                            )  # Rewrite the global to the Numba StencilFunc
                            self.numba_pfunc = numba.njit(
                                parallel=True, **self.numba_args
                            )(self.func)
                            self.numba_func = numba.njit(**self.numba_args)(self.func)
                    if not try_again:
                        self.npfunc[atypes] = False
                        dprint(
                            1,
                            "Numba ParallelAccelerator attempt failed for",
                            self.func,
                            atypes,
                        )
                except Exception:
                    if ndebug >= 2:
                        traceback.print_exc()
                    self.npfunc[atypes] = False
                    dprint(
                        1,
                        "Numba ParallelAccelerator attempt failed for",
                        self.func,
                        atypes,
                    )
                    try:
                        dprint(1, inspect.getsource(self.func))
                    except Exception:
                        pass

        if self.nfunc.get(atypes, True):
            try:
                dprint(3, "pre-running", self.func)
                ret = self.numba_func(*args_for_numba, **kwargs)
                self.nfunc[atypes] = True
                dprint(3, "Numba attempt succeeded.")
                return ret
            except numba.core.errors.TypingError as te:
                print("Ramba TypingError:", te, type(te))
                self.npfunc[atypes] = False
                dprint(1, "Numba attempt failed for", self.func, atypes)
            except Exception:
                self.nfunc[atypes] = False
                dprint(1, "Numba attempt failed for", self.func, atypes)
                raise

        return self.func(*args, **kwargs)


class StencilMetadata:
    stencilmap = {}

    def __init__(self, func, dargs, dkwargs):
        self.func = func
        self.dargs = dargs
        self.dkwargs = dkwargs
        self.hashname = "stencilmeta_" + str(hash(func))
        self.sfunc = None
        self.neighborhood = None
        dprint(2, "StencilMetadata finished")

    def compile_local(self):
        numba_args = self.dkwargs.get("numba", {})
        return numba.stencil(**numba_args)(self.func)

    def compile(self, override_args={}):
        ldict = {}
        gdict = globals()
        # if not self.sfunc:
        if self.hashname not in gdict:
            dprint(2, "Compiling stencil")
            numba_args = self.dkwargs.get("numba", {})
            dprint(2, "Default args:", numba_args)
            numba_args.update(override_args)
            dprint(2, "Override args:", numba_args)
            self.sfunc = numba.stencil(**numba_args)(self.func)
            gdict[self.hashname] = self.sfunc
            # make wrapper to ensure stencil is called from a parallel njit region
            fsrc = inspect.getsource(self.func)
            dprint(3, "fsrc:\n", fsrc)
            fsrc_tokens = fsrc.split("\n")
            varlist = fsrc_tokens[1][
                fsrc_tokens[1].find("(") + 1 : fsrc_tokens[1].find(")")
            ]
            # code = "@numba.njit(parallel=True)\n"
            code = "def " + self.hashname + "_wrap(" + varlist + ",out=None):\n"
            code += "    if out is not None:\n"
            code += "        return " + self.hashname + "(" + varlist + ",out=out)\n"
            code += "    else:\n"
            code += "        return " + self.hashname + "(" + varlist + ")\n"
            # print(code)
            exec(code, gdict, ldict)
            # self.sfunc_wrap = ldict[self.hashname+"_wrap"]
            self.sfunc_wrap = FunctionMetadata(ldict[self.hashname + "_wrap"], [], {})
            # self.sfunc_wrap = numba.njit(parallel=True)(ldict[self.hashname+"_wrap"])
            gdict[self.hashname + "_wrap"] = self.sfunc_wrap
        else:
            self.sfunc = gdict[self.hashname]
            self.sfunc_wrap = gdict[self.hashname + "_wrap"]
        return self.sfunc_wrap

    def __call__(self, *args, **kwargs):
        """Get the types of the arguments.  See if we attempted to compile that
        set of arguments with Numba parallel=True or Numba parallel=False.  We
        always try Numba parallel=True first and then Numba parallel=False.  If
        either of those ever fail to compile or run then we record that failure
        in npfunc or nfunc so that we don't try the failed case again.  If
        neither of those work then fall back to Python.
        """
        dprint(2, "StencilMetadata::__call__", self.func.__name__, args, kwargs)
        self.compile()
        return self.sfunc(*args, **kwargs)


def stencil(*args, **kwargs):
    dprint(1, "stencil:", args, kwargs)

    def make_wrapper(func, args, kwargs):
        dprint(2, "stencil make_wrapper:", args, kwargs, func, type(func))
        if len(args) > 0:
            raise ValueError(
                "ramba.stencil only supports ray and numba dictionary keyword arguments."
            )
        for kwa in kwargs:
            if kwa not in ["ray", "numba"]:
                raise ValueError(
                    "ramba.stencil only supports ray and numba dictionary keyword arguments."
                )
            if not isinstance(kwargs[kwa], dict):
                raise ValueError(
                    "ramba.stencil only supports ray and numba dictionary keyword arguments."
                )

        assert inspect.isfunction(func)

        sm = StencilMetadata(func, args, kwargs)
        return sm

    if len(args) > 0:
        dprint(2, "stencil args0 type:", args[0], type(args[0]))
        func = args[0]
        return make_wrapper(func, (), {})

    def rdec(func):
        dprint(2, "stencil rdec:", func, type(func))
        return make_wrapper(func, args, kwargs)

    return rdec


# **************************************
# Main functions to support Ray+Numba *
# **************************************
if not USE_MPI:

    def remote(*args, **kwargs):
        dprint(2, "remote:", args, kwargs)

        def make_wrapper(func, args, kwargs):
            dprint(2, "remote make_wrapper:", args, kwargs)
            if len(args) > 0:
                raise ValueError(
                    "ramba.remote only supports ray and numba dictionary keyword arguments."
                )
            for kwa in kwargs:
                if kwa not in ["ray", "numba"]:
                    raise ValueError(
                        "ramba.remote only supports ray and numba dictionary keyword arguments."
                    )
                if not isinstance(kwargs[kwa], dict):
                    raise ValueError(
                        "ramba.remote only supports ray and numba dictionary keyword arguments."
                    )

            # ray_init()  # should already have been initied

            if isinstance(func, type):
                dprint(2, "remote args:", args, kwargs)
                dprint(2, "remote func is type")
                if "ray" in kwargs and len(kwargs["ray"]) > 0:
                    rtype = ray.remote(**(kwargs["ray"]))(func)
                    dprint(2, "remote rtype with args:", rtype, type(rtype))
                else:
                    rtype = ray.remote(func)
                    dprint(2, "remote rtype:", rtype, type(rtype))
                return rtype
            elif inspect.isfunction(func):
                fname = func.__name__
                # Use a unique name for each FunctionMetadata to avoid global name conflicts.
                fmname = "FunctionMetadataFor" + fname
                fm = FunctionMetadata(func, args, kwargs)
                ftext = "def {fname}(*args, **kwargs):\n    return {fmname}(*args, **kwargs)\n".format(
                    fname=fname, fmname=fmname
                )
                ldict = {}
                gdict = globals()
                gdict[fmname] = fm
                exec(ftext, gdict, ldict)
                dprint(2, "remote make_wrapper ldict:", ldict)
                if "ray" in kwargs and len(kwargs["ray"]) > 0:
                    rfunc = ray.remote(**(kwargs["ray"]))(ldict[fname])
                    dprint(2, "remote rfunc with args:", rfunc, type(rfunc))
                else:
                    rfunc = ray.remote(ldict[fname])
                    dprint(2, "remote rfunc:", rfunc, type(rfunc))
                return rfunc

        if len(args) > 0:
            dprint(2, "remote args0 type:", args[0], type(args[0]))
            func = args[0]
            return make_wrapper(func, (), {})

        def rdec(func):
            dprint(2, "remote rdec:", func, type(func))
            return make_wrapper(func, args, kwargs)

        return rdec

    remote_exec = False
    egmf_cache = {}

    """This function will facilitate the new deobjectified function code (in text
    form) being transferred to the remote.  Thus, the exec of that code into
    existence happens on the remote."""

    def exec_generic_member_function(
        func_name, deobj_func_name, func_as_str, numba_args, *args, **kwargs
    ):
        global egmf_cache
        if func_name not in egmf_cache:
            dprint(2, "func_name not in cache", numba_args)

            ldict = {}
            gdict = {}
            exec(func_as_str, gdict, ldict)
            ret_func = ldict[deobj_func_name]
            egmf_cache[func_name] = FunctionMetadata(ret_func, (), numba_args)

        func = egmf_cache[func_name]
        return func(*args, **kwargs)

    def jit(*args, **kwargs):
        dprint(2, "jit:", args, kwargs)
        outer_globals = inspect.currentframe().f_back.f_globals
        outer_locals = inspect.currentframe().f_back.f_locals

        def make_deobj_func(fname, inst_vars, other_args, fsrc_tokens):
            indent_line = fsrc_tokens[0]
            indent = indent_line[: -len(indent_line.lstrip())]
            in_nested = False

            # Form a comma-separated string of all the class instance variables in the deobjectified function.
            joined_inst_vars = ",".join(inst_vars.values())

            fsrc_with_return_fix = []
            # Fix existing return statements to also return new instance variable values.
            for line in fsrc_tokens:
                cur_indent = len(line) - len(line.lstrip())
                if in_nested and cur_indent == nested_indent:
                    in_nested = False

                if line.lstrip().startswith("def"):
                    in_nested = True
                    nested_indent = cur_indent
                    # Add line unmodified.
                    fsrc_with_return_fix.append(line)
                elif line.lstrip().startswith("return") and not in_nested:
                    # New line will have same indentation as the original.
                    new_line = line[: -len(line.lstrip())]
                    new_line += "return "
                    # Find everything that comes after "return " in the line.
                    post_return = line.lstrip()[7:]
                    dprint(2, "post_return:", post_return)
                    new_line += "((" + post_return + ")," + joined_inst_vars + ")"
                    fsrc_with_return_fix.append(new_line)
                else:
                    # Add line unmodified.
                    fsrc_with_return_fix.append(line)
            # Add the return of the instance vars if the last line is an implicit return.
            last_line = indent + "return "
            last_line += "((None)," + joined_inst_vars + ")"
            fsrc_with_return_fix.append(last_line)

            # Join all the function source lines into one string.
            fsrc_rest = "\n".join(fsrc_with_return_fix)
            dprint(2, "fsrc_rest:")
            dprint(2, fsrc_rest)

            # Sort by decreasing length so that shorter replacements don't cause larger
            # replacements to be missed.
            sorted_inst_vars = sorted(
                inst_vars, key=lambda k: len(inst_vars[k]), reverse=True
            )
            dprint(2, "sorted_inst_vars:", sorted_inst_vars, type(sorted_inst_vars))
            for siv in sorted_inst_vars:
                fsrc_rest = fsrc_rest.replace("self." + siv, inst_vars[siv])
            dprint(2, "fsrc_rest post_replace:")
            dprint(2, fsrc_rest)

            deobj_func_name = "ramba_deobj_func_" + fname.replace(".", "_")
            deobj_func = (
                "def "
                + deobj_func_name
                + "("
                + joined_inst_vars
                + ("," if len(joined_inst_vars) > 0 and len(other_args) > 0 else "")
                + other_args
                + "):\n"
            )
            if debug:
                deobj_func += indent + 'print("Running deobj func")\n'
            deobj_func += fsrc_rest + "\n"
            return deobj_func, deobj_func_name

        def exec_deobj_func(func_as_str, deobj_func_name, kwargs):
            ldict = {}  # outer_locals
            gdict = outer_globals
            dprint(2, "exec_deobj_func: " + str(gdict.keys()))
            # dprint(2, "frames: "+str(inspect.getouterframes(inspect.currentframe())))
            exec(func_as_str, gdict, ldict)
            ret_func = ldict[deobj_func_name]
            dprint(2, "ldict: ", ldict.keys())
            dprint(2, id(ret_func.__module__))
            dprint(2, ret_func.__module__)
            # Wrap the target function with FunctionMetadata to handle Numba compilation.
            return FunctionMetadata(ret_func, (), kwargs)

        def make_wrapper(func, args, kwargs):
            dprint(2, "jit make_wrapper:", args, kwargs)
            if len(args) > 0:
                raise ValueError(
                    "ramba.remote only supports ray and numba dictionary keyword arguments."
                )
            for kwa in kwargs:
                if kwa not in ["ray", "numba"]:
                    raise ValueError(
                        "ramba.remote only supports ray and numba dictionary keyword arguments."
                    )
                if not isinstance(kwargs[kwa], dict):
                    raise ValueError(
                        "ramba.remote only supports ray and numba dictionary keyword arguments."
                    )

            ray_init()

            numba_args = kwargs.get("numba", {})

            if inspect.isfunction(func):
                fname = func.__name__
                fqname = func.__qualname__
                fsrc = inspect.getsource(func)
                modname = func.__module__
                fmod = sys.modules[modname]
                dprint(2, "fname:", fname, fqname)
                dprint(2, "fsrc:", fsrc)
                dprint(2, "modname:", modname)
                dprint(2, "fmod:", fmod)
                fsrc_tokens = fsrc.split("\n")
                dprint(2, "fsrc_tokens:", fsrc_tokens)
                # Get the non-self arguments to the function....everything after "self" and before ")"
                other_args = fsrc_tokens[1][fsrc_tokens[1].find("self") + 4 :]
                if other_args.find(",") >= 0:
                    other_args = other_args[other_args.find(",") + 1 :]
                other_args = other_args[: other_args.find(")")]
                dprint(2, "other_args:", other_args)
                bio_fscr = BytesIO(fsrc.encode("utf-8")).readline
                tokens = tokenize.tokenize(bio_fscr)
                inst_vars = {}
                tlist = [(t, v) for t, v, _, _, _ in tokens]
                dprint(2, "token list:", tlist)
                for i in range(len(tlist) - 2):
                    if tlist[i][1] == "self" and tlist[i + 1][1] == ".":
                        field = tlist[i + 2][1]
                        inst_vars[field] = "ramba_deobjectified_field_" + field
                dprint(2, "inst_vars:", inst_vars)

                whole_func = fsrc_tokens[1:]
                indent_len = len(whole_func[0]) - len(whole_func[0].lstrip())
                whole_func = "\n".join(x[indent_len:] for x in whole_func)
                dprint(2, "whole_func")
                dprint(2, whole_func)
                st = parser.suite(whole_func)
                dprint(2, "st:", st, type(st))
                stlist = parser.st2list(st)
                dprint(2, "stlist:", stlist, type(stlist))
                # Construct global helper function to do the work from the body of the incoming function.
                deobj_func_txt, deobj_func_name = make_deobj_func(
                    fqname, inst_vars, other_args, fsrc_tokens[2:]
                )
                dprint(2, "full deobj_func:")
                dprint(2, deobj_func_txt)
                if not remote_exec:
                    # This creates the deobj_func from text and wraps it in FunctionMetadata
                    # so that Numba is invoked.
                    deobj_func = exec_deobj_func(
                        deobj_func_txt, deobj_func_name, kwargs
                    )

                new_func = (
                    "def "
                    + fname
                    + "(self"
                    + ("," if len(other_args) > 0 else "")
                    + other_args
                    + "):\n"
                )
                #            new_func += "    print(\"Running new_func.\")\n"
                if debug:
                    for k in inst_vars:
                        new_func += '    print("' + k + ':", self.' + k + ")\n"
                new_func += "    rv = None\n"
                new_func += "    try:\n"
                joined_inst_vars = ",".join(["self." + x for x in inst_vars.keys()])
                dprint(2, "joined_inst_vars:", joined_inst_vars)
                inst_and_others = (
                    joined_inst_vars
                    + ("," if len(joined_inst_vars) > 0 and len(other_args) > 0 else "")
                    + other_args
                )
                ret_and_inst = "rv," + joined_inst_vars
                if remote_exec:
                    new_func += (
                        "        "
                        + ret_and_inst
                        + ' = curmod.exec_generic_member_function("'
                        + fname
                        + '","'
                        + deobj_func_name
                        + '","""'
                        + deobj_func_txt
                        + '""","'
                        + str(numba_args)
                        + '",'
                        + inst_and_others
                        + ")\n"
                    )
                else:
                    new_func += (
                        "        "
                        + ret_and_inst
                        + " = "
                        + deobj_func_name
                        + "("
                        + inst_and_others
                        + ")\n"
                    )
                new_func += "    except AttributeError as ae:\n"
                new_func += '        print("AttributeError:", ae.args)\n'
                new_func += "    except NameError as ae:\n"
                new_func += '        print("NameError:", ae.args)\n'
                new_func += "    except Exception:\n"
                new_func += '        print("error calling deobj", sys.exc_info()[0])\n'
                new_func += "    return rv\n"
                dprint(2, "new_func:", new_func)
                ldict = {}
                gdict = globals()
                if remote_exec:
                    gdict["curmod"] = ramba_module
                else:
                    gdict[deobj_func_name] = deobj_func  # numba.njit(deobj_func)
                exec(new_func, gdict, ldict)
                nfunc = ldict[fname]
                dprint(
                    2,
                    "nfunc:",
                    nfunc,
                    type(nfunc),
                    nfunc.__qualname__,
                    nfunc.__module__,
                )
                return nfunc
            else:
                dprint(2, "Object of type", func, "is not handled by ramba.jit.")
                assert 0

        if len(args) > 0:
            dprint(2, "jit::args0 type:", args[0], type(args[0]))
            if isinstance(args[0], types.FunctionType):
                func = args[0]
                return make_wrapper(func, (), {})

        def rdec(func):
            dprint(2, "jit::rdec:", func, type(func))
            return make_wrapper(func, args, kwargs)

        return rdec


# *******************
# Distributed Arrays
# *******************

######### Barrier code ##############

if not USE_MPI:

    @ray.remote(num_cpus=0)
    class BarrierActor:
        def __init__(self, count):
            self.current = 0
            self.count = count
            self.lock = threading.Lock()
            self.cndvar = threading.Condition(lock=self.lock)

        def barrier(self):
            self.cndvar.acquire()
            self.current += 1
            if self.current == self.count:
                self.current = 0
                self.cndvar.notify_all()
            else:
                try:
                    self.cndvar.wait(timeout=5)
                except Exception:
                    exp = sys.exc_info()[0]
                    print("cndvar.wait exception", exp, type(exp))
                    pass
            self.cndvar.release()

    try:
        ramba_spmd_barrier = ray.get_actor("RambaSpmdBarrier")
    except Exception:
        ramba_spmd_barrier = BarrierActor.options(
            name="RambaSpmdBarrier", max_concurrency=num_workers
        ).remote(num_workers)

    def barrier():
        ray.get(ramba_spmd_barrier.barrier.remote())


### TODO: Need non-Ray version for MPI

######### End Barrier code ##############

# --------------- Global variables to hold timings of parts of Ramba -------------
# The dict value {0:(0,0)} is explained as follows:
# The first 0 is an identifier.  If it is 0 then it corresponds to the overall time for
# that key entry.  Other key values indicate sub-parts of that time.
# In the (0,0) tuple, the first zero is a counter of how many times have been recorded
# and the second 0 is the total time.
time_dict = {}
"""
time_dict = {
    "matmul_b_c_not_dist": {0: (0, 0)},
    "matmul_c_not_dist_a_b_dist_match": {0: (0, 0)},
    "matmul_c_not_dist_a_b_dist_non_match": {0: (0, 0)},
    "matmul_general": {0: (0, 0)},
}
"""

if in_driver() and numba.version_info.short >= (0, 53):
    global compile_recorder
    compile_recorder = numba.core.event.RecordingListener()
    numba.core.event.register("numba:compile", compile_recorder)


def reset_timing():
    for k, v in time_dict.items():
        time_dict[k] = {0: (0, 0)}
    if numba.version_info.short >= (0, 53):
        compile_recorder.buffer = []
    remote_exec_all("reset_compile_stats")
    remote_exec_all("reset_def_ops_stats")


def get_timing(details=False):
    if numba.version_info.short >= (0, 53):
        driver_summary = rec_buf_summary(compile_recorder.buffer)
        stats = remote_call_all("get_compile_stats")
        remote_maximum = functools.reduce(lambda a, b: a if a[1] > b[1] else b, stats)
        time_dict["numba_compile_time"] = {
            0: tuple(map(operator.add, driver_summary, remote_maximum))
        }

    stats = remote_call_all("get_def_ops_stats")
    remote_maximum = functools.reduce(lambda a, b: a if a[1] > b[1] else b, stats)
    time_dict["remote_deferred_ops"] = {0: remote_maximum}

    if details:
        return time_dict
    else:
        return {k: time_dict[k][0] for k in time_dict.keys()}


def get_timing_str(details=False):
    timings = get_timing(details=details)
    res = ""
    if details:
        for k, v in timings.items():
            res += k + ": " + str(v[0][1]) + "s(" + str(v[0][0]) + ")\n"
            for tk, tv in v.items():
                if tk != 0:
                    res += "    " + tk + ": " + str(tv[1]) + "s(" + str(tv[0]) + ")\n"
    else:
        for k, v in timings.items():
            res += k + ": " + str(v[1]) + "s(" + str(v[0]) + ") "
    return res


def add_time(time_name, val):
    if time_name not in time_dict:
        time_dict[time_name] = {0: (0, 0)}
    tindex = time_dict[time_name]
    assert isinstance(tindex, dict)
    cur_val = tindex[0]
    assert isinstance(cur_val, tuple)
    tindex[0] = (cur_val[0] + 1, cur_val[1] + val)


def add_sub_time(time_name, sub_name, val):
    tindex = time_dict[time_name]
    if sub_name not in tindex:
        tindex[sub_name] = (0, 0)
    cur_val = tindex[sub_name]
    tindex[sub_name] = (cur_val[0] + 1, cur_val[1] + val)


# --------------- Global variables to hold timings of parts of Ramba -------------

def get_advindex_dim(index):
    for i in range(len(index)):
        if isinstance(index[i], np.ndarray):
            return i
    assert 0


def distindex_internal(dist, dim, accum):
    if dim >= len(shardview.get_size(dist)):
        yield tuple(accum)
    else:
        for i in range(
            shardview.get_start(dist)[dim],
            shardview.get_start(dist)[dim] + shardview.get_size(dist)[dim],
        ):
            yield from distindex_internal(dist, dim + 1, accum + [i])


def distindex(dist):
    yield from distindex_internal(dist, 0, [])

# class that represents the base distributed array (set of bcontainers on remotes)
# this has a unique GID, and a distribution to specify the actual partition
# multiple ndarrays (arrays and slices) can refer to the same bdarray
# bdarrays do not provide arithmetic operators, and should not be used directly
class bdarray:
    gid_map = (
        weakref.WeakValueDictionary()
    )  # global map from gid to weak references of bdarrays

    def __init__(self, shape, distribution, gid, pad, fdist, dtype):
        self.shape = shape
        self.gid = gid
        self.pad = pad
        self.distribution = distribution
        self.nd_set = weakref.WeakSet()
        self.remote_constructed = False
        self.flex_dist = fdist
        assert gid not in self.gid_map
        self.gid_map[gid] = self
        self.dtype = dtype

    def __del__(self):
        dprint(2, "Deleting bdarray", self.gid, self, "refcount is", len(self.nd_set), self.remote_constructed, id(self))
        if self.remote_constructed:  # check remote constructed flag
            deferred_op.del_remote_array(self.gid)

    @staticmethod
    def atexit():
        def alternate_del(self):
            dprint(2, "Deleting (at exit) bdarray", self.gid, self, "refcount is", len(self.nd_set), self.remote_constructed)

        dprint(2,"at exit -- disabling del handing")
        bdarray.__del__ = alternate_del


    def add_nd(self, nd):
        pass
        #self.nd_set.add(nd)

    @classmethod
    def assign_bdarray(
        cls,
        nd,
        shape,
        gid=None,
        distribution=None,
        pad=0,
        flexible_dist=False,
        dtype=None,
        **kwargs
    ):
        if gid is None:
            gid = uuid.uuid4()
        if gid not in cls.gid_map:
            if dtype is None:
                dtype = np.float64  # default
            if shape == ():
                distribution = np.ndarray(shape, dtype=dtype)
            else:
                distribution = (
                    shardview.default_distribution(shape, **kwargs)
                    if distribution is None
                    else libcopy.deepcopy(distribution)
                )
                # clear offsets (since this is a new array)
                for i in distribution:
                    tmp = shardview.get_base_offset(i)
                    tmp *= 0
            bd = cls(shape, distribution, gid, pad, flexible_dist, dtype)
        else:
            bd = cls.gid_map[gid]
        bd.add_nd(nd)
        dprint(2, "Assigning ndarray to bdarray", gid, "refcount is", len(bd.nd_set))
        dprint(3, "Distribution:", bd.distribution, shape)
        if len(shape) != 0 and not isinstance(bd.distribution, np.ndarray):
            dprint(
                4, "Divisions:", shardview.distribution_to_divisions(bd.distribution)
            )
        return bd

    @classmethod
    def get_by_gid(cls, gid):
        return cls.gid_map[gid]

    @classmethod
    def valid_gid(cls, gid):
        dprint(3, "bdarray: testing existence of gid", gid,": ",(gid in cls.gid_map))
        return gid in cls.gid_map


atexit.register(bdarray.atexit)

"""
# Hmm -- does not seem to be used
def make_padded_shard(core, boundary):
    cshape = core.shape
    assert len(cshape) == len(boundary)
    new_size = [cshape[i] + boundary[i] for i in range(len(cshape))]
"""


class LocalNdarray:
    def __init__(
        self,
        gid,
        remote,
        subspace,
        dim_lens,
        whole,
        border,
        whole_space,
        from_border,
        to_border,
        dtype,
        creation_mode=Filler.PER_ELEMENT,
    ):
        self.gid = gid
        self.remote = remote
        self.subspace = subspace
        self.dim_lens = dim_lens
        self.core_slice = tuple([slice(0, x) for x in dim_lens])
        self.whole = whole
        self.border = border
        self.allocated_size = tuple([x + 2 * border for x in dim_lens])
        self.whole_space = whole_space
        self.from_border = from_border
        self.to_border = to_border
        self.dtype = dtype
        if self.dtype is not None:
            gnpargs = {"dtype": self.dtype}
        else:
            gnpargs = {}

        if border == 0 and creation_mode == Filler.WHOLE_ARRAY_NEW and not gpu_present:
            self.bcontainer = None
        else:
            if debug:
                self.bcontainer = gnp.zeros(self.allocated_size, **gnpargs)
            else:
                self.bcontainer = gnp.empty(self.allocated_size, **gnpargs)

    def init_like(self, gid, dtype=None):
        return LocalNdarray(
            gid,
            self.remote,
            self.subspace,
            self.dim_lens,
            self.whole,
            self.border,
            self.whole_space,
            self.from_border,
            self.to_border,
            self.dtype if dtype is None else dtype,
        )

    def getlocal(self, width=None):
        if isinstance(width, int):
            neighborhood = width
        else:
            neighborhood = 0

        if neighborhood <= self.border:
            return self.bcontainer
        else:
            print("Dynamic addition of border not yet supported!")
            assert 0

        """
        neighborhood = None  # bypass this for now
        if neighborhood is not None:
            boundary_needed = []
            assert(len(self.global_space) == len(self.whole_size))
            assert(len(self.global_space) == len(neighborhood))
            for i in range(len(self.global_space)):
                this_dim = self.global_space[i]
                extra = 0
                if this_dim[0] != 0:
                    extra -= min(0, neighborhood[i][0])
                if this_dim[1] != self.whole_size[i]:
                    extra += max(0, neighborhood[i][1])
                bounded_needed.append(extra)
        else:
            return self.bcontainer
        """

    def getborder(self, stencil_uuid=None):
        if self.to_border is None:
            print("getborder with no pre-computed overlaps")
            # what to do here?
            assert 0
        else:
            dprint(
                2,
                "getborder:",
                self.remote.worker_num,
                len(self.to_border),
                len(self.from_border),
                self.remote,
                type(self.remote),
                ndebug,
            )
            for remote_id, region in self.to_border.items():
                the_slice = tuple(
                    [
                        slice(region[0][x], region[1][x] + 1)
                        for x in range(region.shape[1])
                    ]
                )
                # local_slice = slice_to_local(the_slice, self.subspace)
                local_slice = shardview.slice_to_local(self.subspace, the_slice)
                dprint(
                    2,
                    "Sending Data:",
                    self.remote.worker_num,
                    remote_id,
                    the_slice,
                    local_slice,
                )
                self.remote.comm_queues[remote_id].put(
                    (the_slice, self.bcontainer[local_slice], stencil_uuid)
                )
            for _ in range(len(self.from_border)):
                try:
                    if stencil_uuid is not None:
                        incoming_slice, incoming_data, ruuid = self.remote.comm_queues[
                            self.remote.worker_num
                        ].get(gfilter=lambda x: x[2] == stencil_uuid, timeout=5)
                    else:
                        incoming_slice, incoming_data, ruuid = self.remote.comm_queues[
                            self.remote.worker_num
                        ].get(timeout=5)
                except Exception:
                    print("some exception!", sys.exc_info()[0])
                    assert 0
                # local_slice = slice_to_local(incoming_slice, self.subspace)
                local_slice = shardview.slice_to_local(self.subspace, incoming_slice)
                dprint(
                    2,
                    "Receiving Data:",
                    self.remote.worker_num,
                    incoming_slice,
                    local_slice,
                    self.bcontainer.shape,
                    incoming_data.shape,
                    self.bcontainer[local_slice].size,
                    ruuid,
                )
                self.bcontainer[local_slice] = incoming_data

    @property
    def shape(self):
        return self.whole

    def getglobal(self):
        return self.subspace

    def get_partial_view(self, slice_index, shard, global_index=True, remap_view=True):
        local_slice = (
            shardview.slice_to_local(shard, slice_index)
            if global_index
            else slice_index
        )
        arr = self.bcontainer[local_slice]
        if remap_view:
            arr = shardview.array_to_view(shard, arr)
        # print("get_partial_view:", slice_index, local_slice, shard, self.bcontainer.shape, arr, type(arr), arr.dtype)
        arr.flags.writeable = True  # Force enable writes -- TODO:  Need to check if safe!! May want to set only if ndarray is not readonly
        return arr

    def get_view(self, shard):
        return self.get_partial_view(
            shardview.to_base_slice(shard), shard, global_index=False
        )


# ------------------------------------------------------------------
# Support LocalNdarray usage in Numba.
# ------------------------------------------------------------------
class NumbaLocalNdarray(numba.types.Type):
    def __init__(self, dtype, ndim):
        try:
            self.dtype = numba.np.numpy_support.from_dtype(dtype)
        except NotImplementedError:
            raise ValueError("Unsupported array dtype: %s" % (dtype,))

        self.ndim = ndim
        dprint(
            3,
            "NumbaLocalNdarray:__init__:",
            dtype,
            type(dtype),
            self.dtype,
            type(self.dtype),
        )
        super(NumbaLocalNdarray, self).__init__(name="LocalNdarray")

    @property
    def key(self):
        return self.dtype, self.ndim

    def __repr__(self):
        return "NumbaLocalNdarray(" + str(self.dtype) + "," + str(self.ndim) + ")"


@numba.extending.typeof_impl.register(LocalNdarray)
def typeof_LocalNdarray(val, c):
    dprint(3, "typeof_LocalNdarray", val, type(val))
    return NumbaLocalNdarray(val.bcontainer.dtype, val.bcontainer.ndim)


@numba.extending.register_model(NumbaLocalNdarray)
class NumbaLocalNdarrayModel(numba.extending.models.StructModel):
    def __init__(self, dmm, fe_type):
        dprint(
            3,
            "NumbaLocalNdarrayModel:",
            fe_type,
            type(fe_type),
            fe_type.dtype,
            type(fe_type.dtype),
        )
        members = [
            ("subspace", numba.types.Array(numba.types.int64, 2, "C")),
            ("bcontainer", numba.types.Array(fe_type.dtype, fe_type.ndim, "C")),
        ]
        numba.extending.models.StructModel.__init__(self, dmm, fe_type, members)


numba.extending.make_attribute_wrapper(NumbaLocalNdarray, "bcontainer", "bcontainer")
numba.extending.make_attribute_wrapper(NumbaLocalNdarray, "subspace", "subspace")

# Convert from Python LocalNdarray to Numba format.
@numba.extending.unbox(NumbaLocalNdarray)
def unbox_LocalNdarray(typ, obj, c):
    dprint(
        3,
        "unbox",
        typ,
        type(typ),
        typ.dtype,
        type(typ.dtype),
        numba.core.typing.typeof.typeof(typ.dtype),
    )
    numba_localndarray = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder
    )

    subspace_obj = c.pyapi.object_getattr_string(obj, "subspace")
    subspace_unboxed = numba.core.boxing.unbox_array(
        numba.types.Array(numba.types.int64, 2, "C"), subspace_obj, c
    )
    numba_localndarray.subspace = subspace_unboxed.value

    bcontainer_obj = c.pyapi.object_getattr_string(obj, "bcontainer")
    bcontainer_unboxed = numba.core.boxing.unbox_array(
        numba.types.Array(typ.dtype, typ.ndim, "C"), bcontainer_obj, c
    )
    numba_localndarray.bcontainer = bcontainer_unboxed.value

    c.pyapi.decref(subspace_obj)
    c.pyapi.decref(bcontainer_obj)

    is_error = numba.core.cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return numba.core.pythonapi.NativeValue(
        numba_localndarray._getvalue(), is_error=is_error
    )


@numba.extending.box(NumbaLocalNdarray)
def box_LocalNdarray(typ, val, c):
    dprint(3, "box_LocalNdarray")
    numba_localndarray = numba.core.cgutils.create_struct_proxy(typ)(
        c.context, c.builder
    )
    classobj = c.pyapi.unserialize(c.pyapi.serialize_object(LocalNdarray))
    assert 0
    return ret


@numba.extending.overload_method(NumbaLocalNdarray, "getlocal")
def localndarray_getlocal(localndarray):
    def getter(localndarray):
        return localndarray.bcontainer

    return getter


@numba.extending.overload_method(NumbaLocalNdarray, "getglobal")
def localndarray_getglobal(localndarray):
    def getter(localndarray):
        return localndarray.subspace

    return getter


# ------------------------------------------------------------------
# Support LocalNdarray usage in Numba.
# ------------------------------------------------------------------


def none_if_zero(a, b):
    x = a - b
    if x == 0:
        return None
    return x


def get_remote_ranges(distribution, required_division):
    divisions = shardview.distribution_to_divisions(distribution)
    dprint(
        2,
        "get_remote_ranges: distribution\n",
        distribution,
        "\ndivisions\n",
        divisions,
        "\nrequired_division\n",
        required_division,
    )

    ret = []
    for node_num in range(len(divisions)):
        sview = divisions[node_num]
        bi = shardview.block_intersection(sview, required_division)
        if bi is not None:
            ret.append((node_num, bi, sview))

    return ret


def division_non_empty(x):
    xshape = x.shape
    if len(xshape) == 3:
        for i in range(xshape[0]):
            if np.any(x[i][0] > x[i][1]):
                return False
        return True
    elif len(xshape) == 2:
        return not np.any(x[0] > x[1])
    else:
        assert 0


@numba.njit(parallel=True, cache=True)
def reduce_list(x):
    res = np.zeros(x[0].shape)
    for i in numba.prange(len(x)):
        res += x[i]
    return res


@functools.lru_cache()
def get_do_fill(filler: FillerFunc, num_dim):
    filler = filler.func
    dprint(2, "get_do_fill", filler, num_dim)
    # FIX ME: What if njit(filler) fails during compile or runtime?
    njfiller = (
        filler
        if isinstance(filler, numba.core.registry.CPUDispatcher)
        else numba.extending.register_jitable(filler)
        #else numba.njit(filler)
    )

    if num_dim > 1:

        def do_fill(A, sz, starts):
            for i in numba.pndindex(sz):
                arg = sz
                # Construct the global index.
                for j in range(len(starts)):
                    arg = UT.tuple_setitem(arg, j, i[j] + starts[j])
                A[i] = njfiller(arg)

        return FunctionMetadata(do_fill, [], {}, no_global_cache=True)
    else:

        def do_fill(A, sz, starts):
            for i in numba.prange(sz[0]):
                A[i] = njfiller((i + starts[0],))

        return FunctionMetadata(do_fill, [], {}, no_global_cache=True)


@functools.lru_cache()
def get_do_fill_non_tuple(filler: FillerFunc, num_dim):
    filler = filler.func
    dprint(2, "get_do_fill_non_tuple", filler)
    njfiller = (
        filler
        if isinstance(filler, numba.core.registry.CPUDispatcher)
        else numba.extending.register_jitable(filler)
        #else numba.njit(filler)
    )

    if num_dim > 1:

        def do_fill(A, sz, starts):
            for i in numba.pndindex(sz):
                arg = sz
                # Construct the global index.
                for j in range(len(starts)):
                    arg = UT.tuple_setitem(arg, j, i[j] + starts[j])
                A[i] = njfiller(*arg)

        return FunctionMetadata(do_fill, [], {}, no_global_cache=True)
    else:

        def do_fill(A, sz, starts):
            for i in numba.prange(sz[0]):
                A[i] = njfiller(i + starts[0])

        return FunctionMetadata(do_fill, [], {}, no_global_cache=True)


smap_func = 0

@functools.lru_cache()
def get_smap_fill(filler: FillerFunc, num_dim, ramba_array_args, parallel=True):
    filler = filler.func
    dprint(2, "get_smap_fill", filler, num_dim, ramba_array_args)
    #njfiller = filler
    njfiller = (
        filler
        if isinstance(filler, numba.core.registry.CPUDispatcher)
        #else numba.extending.register_jitable(filler)
        else numba.extending.register_jitable(inline="always")(filler)
        #else numba.njit(filler)
    )

    global smap_func

    fname = f"smap_fill{smap_func}"
    fillername = f"njfiller{smap_func}"
    smap_func += 1
    arg_names = ",".join([f"arg{i}" for i in range(len(ramba_array_args))])
    fill_txt  = f"def {fname}(A, sz, {arg_names}):\n"
    arg_list = [ f"arg{idx}[i]" if ramba_array_args[idx] else f"arg{idx}" for idx in range(len(ramba_array_args)) ]

    if num_dim > 1:
        fill_txt += "    for i in numba.pndindex(sz[0]):\n"
    else:
        fill_txt += "    for i in numba.prange(sz[0]):\n"

    fill_txt += f"        A[i] = {fillername}(" + ",".join(arg_list) + ")\n"
    dprint(2, "fill_txt:")
    dprint(2, fill_txt)
    ldict = {}
    gdict = globals()
    gdict[fillername] = njfiller
    exec(fill_txt, gdict, ldict)
    return FunctionMetadata(ldict[fname], [], {}, no_global_cache=True, parallel=parallel)


@functools.lru_cache()
def get_smap_fill_index(filler: FillerFunc, num_dim, ramba_array_args, parallel=True):
    filler = filler.func
    dprint(2, "get_smap_fill_index", filler, type(filler), num_dim, ramba_array_args)
    njfiller = (
        filler
        if isinstance(filler, numba.core.registry.CPUDispatcher)
        #else numba.extending.register_jitable(filler)
        else numba.extending.register_jitable(inline="always")(filler)
        #else numba.njit(filler)
    )

    global smap_func

    fname = f"smap_fill_index{smap_func}"
    fillername = f"njfiller{smap_func}"
    smap_func += 1
    arg_names = ",".join([f"arg{i}" for i in range(len(ramba_array_args))])
    fill_txt  = f"def {fname}(A, sz, starts, {arg_names}):\n"

    if num_dim > 1:

        fill_txt += "    for i in numba.pndindex(sz):\n"
        fill_txt += "        si = ("
        for nd in range(num_dim):
            fill_txt += f"i[{nd}] + starts[{nd}]"
            if nd < num_dim - 1:
                fill_txt += ","
        fill_txt += ")\n"

    else:

        fill_txt += "    for i in numba.prange(sz[0]):\n"
        fill_txt += "        si = i + starts[0]\n"

    arg_list = [ f"arg{idx}[i]" if ramba_array_args[idx] else f"arg{idx}" for idx in range(len(ramba_array_args)) ]
    fill_txt += f"        A[i] = {fillername}(si, " + ",".join(arg_list) + ")\n"
    dprint(2, "fill_txt:", num_dim)
    dprint(2, fill_txt)
    ldict = {}
    gdict = globals()
    gdict[fillername] = njfiller
    exec(fill_txt, gdict, ldict)
    return FunctionMetadata(ldict[fname], [], {}, no_global_cache=True, parallel=parallel)


@functools.lru_cache()
def get_sreduce_fill(filler: FillerFunc, reducer: FillerFunc, num_dim, ramba_array_args, parallel=True):
    filler = filler.func
    reducer = reducer.func
    dprint(2, "get_sreduce_fill", filler, reducer, num_dim, ramba_array_args)
    njfiller = (
        filler
        if isinstance(filler, numba.core.registry.CPUDispatcher)
        #else numba.extending.register_jitable(filler)
        else numba.extending.register_jitable(inline="always")(filler)
        #else numba.njit(filler)
    )
    njreducer = (
        reducer
        if isinstance(reducer, numba.core.registry.CPUDispatcher)
        else numba.extending.register_jitable(reducer)
        #else numba.njit(reducer)
    )

    global smap_func

    fname = f"sreduce_fill{smap_func}"
    fillername = f"njfiller{smap_func}"
    reducername = f"njreducer{smap_func}"
    smap_func += 1
    arg_names = ",".join([f"arg{i}" for i in range(len(ramba_array_args))])
    fill_txt  = f"def {fname}(sz, identity, {arg_names}):\n"
    fill_txt +=  "    result = identity\n"
    arg_list = [ f"arg{idx}[i]" if ramba_array_args[idx] else f"arg{idx}" for idx in range(len(ramba_array_args)) ]

    if num_dim > 1:
        fill_txt +=  "    for i in numba.pndindex(sz):\n"
    else:
        fill_txt +=  "    for i in numba.prange(sz[0]):\n"

    fill_txt += f"        fres = {fillername}(" + ",".join(arg_list) + ")\n"
    fill_txt += f"        result = {reducername}(result, fres)\n"
    fill_txt +=  "    return result\n"
    dprint(2, "fill_txt:", num_dim)
    dprint(2, fill_txt)
    ldict = {}
    gdict = globals()
    gdict[fillername] = njfiller
    gdict[reducername] = njreducer
    exec(fill_txt, gdict, ldict)
    return FunctionMetadata(ldict[fname], [], {}, no_global_cache=True, parallel=parallel)


@functools.lru_cache()
def get_sreduce_fill_index(filler: FillerFunc, reducer: FillerFunc, num_dim, ramba_array_args, parallel=True):
    filler = filler.func
    reducer = reducer.func
    dprint(2, "get_sreduce_fill_index", filler, reducer, num_dim, ramba_array_args)
    njfiller = (
        filler
        if isinstance(filler, numba.core.registry.CPUDispatcher)
        #else numba.extending.register_jitable(filler)
        else numba.extending.register_jitable(inline="always")(filler)
        #else numba.njit(filler)
    )
    njreducer = (
        reducer
        if isinstance(reducer, numba.core.registry.CPUDispatcher)
        else numba.extending.register_jitable(reducer)
        #else numba.extending.register_jitable(inline="always")(reducer)
        #else numba.njit(reducer)
    )

    global smap_func

    fname = f"sreduce_fill_index{smap_func}"
    fillername = f"njfiller{smap_func}"
    reducername = f"njreducer{smap_func}"
    smap_func += 1
    arg_names = ",".join([f"arg{i}" for i in range(len(ramba_array_args))])
    fill_txt  = f"def {fname}(sz, starts, identity, {arg_names}):\n"
    fill_txt +=  "    result = identity\n"

    if num_dim > 1:
        fill_txt +=  "    for i in numba.pndindex(sz):\n"
        fill_txt +=  "        si = ("
        for nd in range(num_dim):
            fill_txt += f"i[{nd}] + starts[{nd}]"
            if nd < num_dim - 1:
                fill_txt += ","
        fill_txt +=  ")\n"
    else:
        fill_txt +=  "    for i in numba.prange(sz[0]):\n"
        fill_txt +=  "        si = i + starts[0]\n"

    arg_list = [ f"arg{idx}[i]" if ramba_array_args[idx] else f"arg{idx}" for idx in range(len(ramba_array_args)) ]
    fill_txt += f"        fres = {fillername}(si, " + ",".join(arg_list) + ")\n"
    fill_txt += f"        result = {reducername}(result, fres)\n"
    fill_txt +=  "    return result\n"
    dprint(2, "fill_txt:", num_dim)
    dprint(2, fill_txt)
    ldict = {}
    gdict = globals()
    gdict[fillername] = njfiller
    gdict[reducername] = njreducer
    exec(fill_txt, gdict, ldict)
    return FunctionMetadata(ldict[fname], [], {}, no_global_cache=True, parallel=parallel)


def rec_buf_summary(rec_buf):
    total = 0
    depth = 0
    num_compiled = 0
    for rec in rec_buf:
        if rec[1].is_start:
            num_compiled += 1
            if depth == 0:
                start = rec[0]
                # print("rec_buf_summary starting", start)
            depth += 1
        elif rec[1].is_end:
            depth -= 1
            if depth == 0:
                total += rec[0] - start
                #print("rec_buf_summary ending", rec, rec[0], rec[0] - start, total, rec[1].data["dispatcher"])

    return (num_compiled, total)


# The following support functions are for ZMQ with BCAST -- this helps create a 2 level control and response tree
def workers_by_node(comm_queues):
    w = {}
    for i, q in enumerate(comm_queues):
        if q.ip not in w:
            w[q.ip] = []
        w[q.ip].append(i)
    return w


def get_children(i, comm_queues):
    if not USE_RAY_CALLS and USE_ZMQ and USE_BCAST:
        w = workers_by_node(comm_queues)
        nw = w[comm_queues[i].ip]
        aggr = nw[0]
        nw = nw[1:]
        return aggr == i, nw
    return False, []


def get_aggregators(comm_queues):
    if not USE_RAY_CALLS and USE_ZMQ and USE_BCAST:
        w = workers_by_node(comm_queues)
        aggregators = [x[0] for _, x in w.items()]
        return aggregators
    return []


def unpickle_args(args):
    for idx, value in enumerate(args):
        if isinstance(value, bytes):
            args[idx] = func_loads(value)

def external_set_common_state(st):
    set_common_state(st, globals())

class RemoteState:
    def __init__(self, worker_num, common_state):
        external_set_common_state(common_state)
        z = numa.set_affinity(worker_num, numa_zones)
        dprint(
            1,
            "Worker:",
            worker_num,
            socket.gethostname(),
            num_workers,
            z,
            num_threads,
            ndebug,
            ntiming,
            timing,
        )
        self.numpy_map = {}
        self.worker_num = worker_num
        # ensure numba is imported in remote contexts
        x = numba.config.NUMBA_DEFAULT_NUM_THREADS
        numba.set_num_threads(num_threads)
        self.my_comm_queue = None  # ramba_queue.Queue()
        self.my_control_queue = None  # ramba_queue.Queue()
        self.my_rvq = None
        self.up_rvq = None
        self.tlast = timer()

        if numba.version_info.short >= (0, 53):
            self.compile_recorder = numba.core.event.RecordingListener()
            numba.core.event.register("numba:compile", self.compile_recorder)
        self.deferred_ops_time = 0
        self.deferred_ops_count = 0

    def set_comm_queues(self, comm_queues, control_queues):
        self.comm_queues = comm_queues
        self.comm_queues[self.worker_num] = self.my_comm_queue
        self.control_queues = control_queues
        self.control_queues[self.worker_num] = self.my_control_queue
        self.is_aggregator, self.children = get_children(
            self.worker_num, self.comm_queues
        )
        return self.comm_queues

    def get_comm_queue(self):
        if self.my_comm_queue is None:
            self.my_comm_queue = ramba_queue.Queue(hint_ip=hint_ip, tag=0)
        return self.my_comm_queue

    def get_control_queue(self):
        if self.my_control_queue is None:
            self.my_control_queue = ramba_queue.Queue(hint_ip=hint_ip, tag=1)
        return self.my_control_queue

    def set_up_rvq(self, rvq):
        self.up_rvq = rvq

    def destroy_array(self, gid):
        #if gid in self.numpy_map:
        self.numpy_map.pop(gid)

    def create_array(
        self,
        gid,
        subspace,
        whole,
        filler,
        local_border,
        dtype,
        whole_space,
        from_border,
        to_border,
        filler_tuple_arg=True,
    ):
        dprint(
            3, "RemoteState::create_array:", gid, subspace, filler, local_border, dtype,
            "\ndtype of subspace", subspace.dtype,
        )
        num_dim = len(shardview.get_size(subspace))
        dim_lens = tuple(shardview.get_size(subspace))
        starts = tuple(shardview.get_start(subspace))
        dprint(
            3,
            "num_dim:",
            num_dim,
            dim_lens,
            type(dim_lens),
            dim_lens[0],
            type(dim_lens[0]),
            starts,
        )
        # Per-element fillers is the default.  This can be updated through passing a Filler object as filler.
        # If there is no filler then setting this mode will assure that LocalNdarray allocates the array.
        mode = Filler.PER_ELEMENT

        if filler:
            filler = func_loads(filler)
        if filler:
            # per_element = True
            do_compile = True
            if isinstance(filler, Filler):
                mode = filler.mode
                do_compile = filler.do_compile
                filler = filler.func
            dprint(3, "Has filler", mode, do_compile, filler_tuple_arg)

        lnd = LocalNdarray(
            gid,
            self,
            subspace,
            dim_lens,
            whole,
            local_border,
            whole_space,
            from_border,
            to_border,
            dtype,
            creation_mode=mode,
        )
        self.numpy_map[gid] = lnd
        new_bcontainer = lnd.bcontainer
        if filler:
            if mode == Filler.PER_ELEMENT:

                def do_per_element_non_compiled(new_bcontainer, dim_lens, starts):
                    for i in np.ndindex(dim_lens):
                        # Construct the global index.
                        arg = tuple(map(operator.add, i, starts))
                        if filler_tuple_arg:
                            new_bcontainer[i] = filler(arg)
                        else:
                            new_bcontainer[i] = filler(*arg)

                if do_compile:
                    try:
                        if filler_tuple_arg:
                            do_fill = get_do_fill(FillerFunc(filler), num_dim)
                        else:
                            do_fill = get_do_fill_non_tuple(FillerFunc(filler), num_dim)

                        do_fill(new_bcontainer, dim_lens, starts)
                    except Exception:
                        dprint(1, "Some exception running filler.", sys.exc_info()[0])
                        if ndebug >= 2:
                            traceback.print_exc()
                        do_per_element_non_compiled(new_bcontainer, dim_lens, starts)
                else:
                    do_per_element_non_compiled(new_bcontainer, dim_lens, starts)
            elif mode == Filler.WHOLE_ARRAY_NEW:
                if do_compile:
                    try:
                        njfiller = (
                            filler
                            if isinstance(filler, numba.core.registry.CPUDispatcher)
                            else FunctionMetadata(filler, (), {})
                        )
                        filler_res = njfiller(dim_lens, starts)
                    except Exception:
                        dprint(1, "Some exception running filler.", sys.exc_info()[0])
                        traceback.print_exc()
                        filler_res = filler(dim_lens, starts)
                else:
                    filler_res = filler(dim_lens, starts)
                if lnd.bcontainer is None:
                    lnd.bcontainer = filler_res
                else:
                    lnd.bcontainer[:] = filler_res
            elif mode == Filler.WHOLE_ARRAY_INPLACE:
                if do_compile:
                    try:
                        njfiller = (
                            filler
                            if isinstance(filler, numba.core.registry.CPUDispatcher)
                            else FunctionMetadata(filler, (), {})
                        )
                        njfiller(new_bcontainer, dim_lens, starts)
                    except Exception:
                        dprint(1, "Some exception running filler.", sys.exc_info()[0])
                        traceback.print_exc()
                        filler(new_bcontainer, dim_lens, starts)
                else:
                    filler(new_bcontainer, dim_lens, starts)
            else:
                assert 0

    def copy(self, new_gid, old_gid, border, from_border, to_border):
        old_lnd = self.numpy_map[old_gid]
        lnd = LocalNdarray(
            new_gid,
            self,
            old_lnd.subspace,
            old_lnd.dim_lens,
            old_lnd.whole,
            border,
            old_lnd.whole_space,
            from_border,
            to_border,
            old_lnd.dtype,
        )
        self.numpy_map[new_gid] = lnd
        new_bcontainer = lnd.bcontainer
        new_bcontainer[lnd.core_slice] = old_lnd.bcontainer[lnd.core_slice]

    def triu(self, new_gid, old_gid, k, border, from_border, to_border):
        old_lnd = self.numpy_map[old_gid]
        lnd = LocalNdarray(
            new_gid,
            self,
            old_lnd.subspace,
            old_lnd.dim_lens,
            old_lnd.whole,
            border,
            old_lnd.whole_space,
            from_border,
            to_border,
            old_lnd.dtype,
        )
        self.numpy_map[new_gid] = lnd
        new_bcontainer = lnd.bcontainer
        starts = tuple(shardview.get_start(lnd.subspace))
        # We should have a Numba function here to parallelize.
        new_bcontainer[lnd.core_slice] = np.triu(
            old_lnd.bcontainer[lnd.core_slice], k - (starts[1] - starts[0])
        )

    def setitem1(self, to_gid, from_gid):
        to_lnd = self.numpy_map[to_gid]
        from_lnd = self.numpy_map[from_gid]
        to_lnd.bcontainer[to_lnd.core_slice] = from_lnd.bcontainer[to_lnd.core_slice]

    def push_array(
        self,
        gid,
        subspace,
        whole,
        pushed_subarray,
        border,
        whole_space,
        from_border,
        to_border,
        dtype,
    ):
        num_dim = len(shardview.get_size(subspace))
        dim_lens = tuple(shardview.get_size(subspace))
        lnd = LocalNdarray(
            gid,
            self,
            subspace,
            dim_lens,
            whole,
            border,
            whole_space,
            from_border,
            to_border,
            dtype,
        )
        self.numpy_map[gid] = lnd
        new_bcontainer = lnd.bcontainer
        if dim_lens != pushed_subarray.shape:
            print("dim_lens != pushed_subarray.shape", dim_lens, pushed_subarray.shape)
        new_bcontainer[lnd.core_slice] = pushed_subarray

    def get_array(self, gid):
        #        print("get_array:", gid, self.numpy_map.keys())
        lnd = self.numpy_map[gid]
        return lnd.bcontainer[lnd.core_slice]

    def get_partial_array(self, gid, slice_index):
        return self.numpy_map[gid].bcontainer[slice_index]

    # returns a view corresponding to the whole shard
    # output dimensionality is always same as bcontainer
    def get_view(self, gid, shard):
        return self.get_partial_view(
            gid, shardview.to_base_slice(shard), shard, global_index=False
        )

    # returns a view corresponding to the slice_index based on given shardview
    # slice_index may be globally indexed (same num dims as view), or locally indexed (same num dims as bcontainer)
    # output dimensionality is always same as bcontainer
    def get_partial_view(
        self, gid, slice_index, shard, global_index=True, remap_view=True
    ):
        lnd = self.numpy_map[gid]
        return lnd.get_partial_view(
            slice_index, shard, global_index=global_index, remap_view=remap_view
        )
        # local_slice = shard.slice_to_local(slice_index) if global_index else slice_index
        ##print("get_partial_view:",self.worker_num, slice_index, local_slice, shard, lnd.bcontainer.shape)
        # arr = lnd.bcontainer[local_slice]
        # if remap_view: arr = shard.array_to_view(arr)
        # return arr

    def getitem_global(self, gid, global_index, shard):
        lnd = self.numpy_map[gid]
        return lnd.bcontainer[shardview.index_to_base(shard, global_index)]

    def array_unaryop(self, lhs_gid, out_gid, dist, out_dist, op, axis, dtype):
        unaryop = getattr(self.get_view(lhs_gid, dist[self.worker_num]), op)
        if out_gid is not None:
            outarr = self.get_view(out_gid, out_dist[self.worker_num])
            sl = tuple(0 if i == axis else slice(None) for i in range(outarr.ndim))
            outarr[sl] = unaryop(axis=axis, dtype=dtype)
        else:
            #    print("befor unaryop array_unary_op", timer())
            ret = unaryop(axis=axis, dtype=dtype)  # works for some simple reductions
            #    print("after unaryop array_unary_op", timer())
            return ret

    # TODO: should use get_view
    def smap(self, out_gid, first_gid, args, func, dtype, parallel):
        func = func_loads(func)
        first = self.numpy_map[first_gid]
        lnd = first.init_like(out_gid, dtype=dtype)
        self.numpy_map[out_gid] = lnd
        new_bcontainer = lnd.bcontainer
        unpickle_args(args)
        # uuids = list(filter(args, lambda x: isinstance(x, uuid.UUID)))
        # bcontainers = [self.numpy_map[x][0] for x in uuids]

        if True:
            ramba_array_args = [isinstance(x, uuid.UUID) for x in args]
            do_fill = get_smap_fill(FillerFunc(func), len(first.dim_lens), tuple(ramba_array_args), parallel=parallel)
            #fargs = tuple([self.numpy_map[x].get_view() if isinstance(x, uuid.UUID) else x for x in args])
            fargs = tuple([self.numpy_map[x].bcontainer if isinstance(x, uuid.UUID) else x for x in args])
            do_fill(new_bcontainer, first.dim_lens, *fargs)
        else:
            for index in np.ndindex(first.dim_lens):
                fargs = [
                    self.numpy_map[x].bcontainer[index] if isinstance(x, uuid.UUID) else x
                    for x in args
                ]
                new_bcontainer[index] = func(*fargs)

    # TODO: should use get_view
    def smap_index(self, out_gid, first_gid, args, func, dtype, parallel):
        func = func_loads(func)
        first = self.numpy_map[first_gid]
        self.numpy_map[out_gid] = first.init_like(out_gid, dtype=dtype)
        new_bcontainer = self.numpy_map[out_gid].bcontainer
        starts = tuple(shardview.get_start(first.subspace))
        unpickle_args(args)

        if True:
            ramba_array_args = [isinstance(x, uuid.UUID) for x in args]
            do_fill = get_smap_fill_index(FillerFunc(func), len(first.dim_lens), tuple(ramba_array_args), parallel=parallel)
            fargs = tuple([self.numpy_map[x].bcontainer if isinstance(x, uuid.UUID) else x for x in args])
            do_fill(new_bcontainer, first.dim_lens, starts, *fargs)
        else:
            for index in np.ndindex(first.dim_lens):
                index_arg = tuple(map(operator.add, index, starts))
                # index_arg = tuple(shardview.base_to_index(first.subspace,index))
                fargs = [
                    self.numpy_map[x].bcontainer[index] if isinstance(x, uuid.UUID) else x
                    for x in args
                ]
                new_bcontainer[index] = func(index_arg, *fargs)

    # TODO: should use get_view
    def sreduce(self, first_gid, args, func, reducer, reducer_driver, identity, a_send_recv, parallel):
        start_time = timer()
        func = func_loads(func)
        reducer = func_loads(reducer)
        reducer_driver = func_loads(reducer_driver)
        first = self.numpy_map[first_gid]
        result = None
        unpickle_args(args)
        #assert len(args) == 1
        if True:
            ramba_array_args = [isinstance(x, uuid.UUID) for x in args]
            do_fill = get_sreduce_fill(FillerFunc(func), FillerFunc(reducer), len(first.dim_lens), tuple(ramba_array_args), parallel=parallel)
            def fix_args(x):
                if isinstance(x, uuid.UUID):
                    return self.numpy_map[x].bcontainer
                elif callable(x):
                    res = x()
                    return res
                else:
                    return x
            fargs = tuple([fix_args(x) for x in args])
            """
            for farg in fargs:
                print("farg:", farg, type(farg))
                if isinstance(farg, np.ndarray):
                    print("shape:", farg.shape)
            """
            res = do_fill(first.dim_lens, identity, *fargs)
            #print("worker do_fill res:", res, res[0].shape, res[1].shape)
            #return res
            after_map_time = timer()

            # Distributed and Parallel Reduction
            max_worker = num_workers
            while max_worker > 1:
                # midpoint is the index of the first worker that needs to send
                midpoint = (max_worker + 1) // 2
                # this worker will send
                if self.worker_num >= midpoint:
                    send_to = self.worker_num - midpoint
                    self.comm_queues[send_to].put((a_send_recv, res))
                    break
                else:
                    # If the remaining workers is odd and this worker is the last one then it won't receive
                    # any communication or needs to do any compute but just goes to the next iteration where
                    # it will send.
                    if not (
                        max_worker % 2 == 1 and self.worker_num == midpoint - 1
                    ):
                        try:
                            incoming_uuid, incoming_res = self.comm_queues[
                                    self.worker_num
                            ].get(gfilter=lambda x: x[0] == a_send_recv, timeout=5)
                        except Exception:
                            print("some exception!", sys.exc_info()[0])
                            assert 0
                        res = reducer_driver(res, incoming_res)

                max_worker = midpoint

            after_reduce_time = timer()
            tprint(2, "sreduce map time:", self.worker_num, after_map_time - start_time)
            tprint(2, "sreduce distributed reduction time:", self.worker_num, after_reduce_time - after_map_time)

            if self.worker_num == 0:
                return res
            else:
                return None
        else:
            for index in np.ndindex(first.dim_lens):
                fargs = [
                    self.numpy_map[x].bcontainer[index] if isinstance(x, uuid.UUID) else x
                    for x in args
                ]
                if result is None:
                    result = func(*fargs)
                else:
                    result = reducer(result, func(*fargs))
            return result

    def sreduce_index(self, first_gid, args, func, reducer, reducer_driver, identity, a_send_recv, parallel):
        start_time = timer()
        func = func_loads(func)
        reducer = func_loads(reducer)
        reducer_driver = func_loads(reducer_driver)
        first = self.numpy_map[first_gid]
        result = None
        starts = tuple(shardview.get_start(first.subspace))
        unpickle_args(args)
        if callable(identity):
            identity = identity()
        #assert len(args) == 1
        if True:
            ramba_array_args = [isinstance(x, uuid.UUID) for x in args]
            do_fill = get_sreduce_fill_index(FillerFunc(func), FillerFunc(reducer), len(first.dim_lens), tuple(ramba_array_args), parallel=parallel)
            def fix_args(x):
                if isinstance(x, uuid.UUID):
                    return self.numpy_map[x].bcontainer
                elif callable(x):
                    res = x()
                    return res
                else:
                    return x
            fargs = tuple([fix_args(x) for x in args])
            """
            for farg in fargs:
                print("farg:", farg, type(farg))
                if isinstance(farg, np.ndarray):
                    print("shape:", farg.shape)
            """
            res = do_fill(first.dim_lens, starts, identity, *fargs)
            #print("worker do_fill res:", res, res.shape)
            after_map_time = timer()

            # Distributed and Parallel Reduction
            max_worker = num_workers
            while max_worker > 1:
                # midpoint is the index of the first worker that needs to send
                midpoint = (max_worker + 1) // 2
                # this worker will send
                if self.worker_num >= midpoint:
                    send_to = self.worker_num - midpoint
                    self.comm_queues[send_to].put((a_send_recv, res))
                    break
                else:
                    # If the remaining workers is odd and this worker is the last one then it won't receive
                    # any communication or needs to do any compute but just goes to the next iteration where
                    # it will send.
                    if not (
                        max_worker % 2 == 1 and self.worker_num == midpoint - 1
                    ):
                        try:
                            incoming_uuid, incoming_res = self.comm_queues[
                                    self.worker_num
                            ].get(gfilter=lambda x: x[0] == a_send_recv, timeout=5)
                        except Exception:
                            print("some exception!", sys.exc_info()[0])
                            assert 0
                        res = reducer_driver(res, incoming_res)

                max_worker = midpoint

            after_reduce_time = timer()
            tprint(2, "sreduce_index map time:", self.worker_num, after_map_time - start_time)
            tprint(2, "sreduce_index distributed reduction time:", self.worker_num, after_reduce_time - after_map_time)

            if self.worker_num == 0:
                return res
            else:
                return None
        else:
            for index in np.ndindex(first.dim_lens):
                # Make the global index.
                index_arg = tuple(map(operator.add, index, starts))
                fargs = [
                    self.numpy_map[x].bcontainer[index] if isinstance(x, uuid.UUID) else x
                    for x in args
                ]
                if result is None:
                    result = func(index_arg, *fargs)
                else:
                    result = reducer(result, func(index_arg, *fargs))
            return result

    def reshape(
        self,
        out_gid,
        out_size,
        out_distribution,
        arr_gid,
        arr_size,
        arr_distribution,
        send_recv,
    ):
        out_lnd = self.numpy_map[out_gid]
        arr_lnd = self.numpy_map[arr_gid]

        worker_data = [[] for _ in range(num_workers)]
        out_cumul = np.array(
            [int(np.prod(out_size[i + 1 :])) for i in range(len(out_size))]
        )
        arr_cumul = np.array(
            [int(np.prod(arr_size[i + 1 :])) for i in range(len(arr_size))]
        )
        dprint(
            2,
            "reshape",
            self.worker_num,
            out_cumul,
            arr_cumul,
            arr_lnd.subspace,
            arr_lnd.whole,
            shardview.distribution_to_divisions(arr_lnd.whole_space)[self.worker_num],
            arr_distribution,
        )

        def to_flat(index):
            return sum(index * arr_cumul)

        def to_out_index(flat):
            out_index = []
            for i in range(len(out_cumul)):
                out_index.append(flat // out_cumul[i])
                flat = flat % out_cumul[i]
            return tuple(out_index)

        for index in distindex(arr_lnd.subspace):
            flat = to_flat(index)
            out_index = to_out_index(flat)
            out_worker = shardview.find_index(out_distribution, out_index)
            base_index = shardview.index_to_base(
                arr_distribution[self.worker_num], index
            )
            if out_worker != self.worker_num:
                worker_data[out_worker].append(arr_lnd.bcontainer[base_index])
            else:
                out_base_index = shardview.index_to_base(
                    out_distribution[self.worker_num], out_index
                )
                out_lnd.bcontainer[out_base_index] = arr_lnd.bcontainer[base_index]

        for i in range(num_workers):
            if i != self.worker_num:
                self.comm_queues[i].put(
                    (send_recv, self.worker_num, np.array(worker_data[i]))
                )

        for i in range(num_workers):
            if i != self.worker_num:
                (
                    incoming_send_recv_uuid,
                    from_worker_num,
                    incoming_data,
                ) = self.comm_queues[self.worker_num].get(
                    gfilter=lambda x: x[0] == send_recv, timeout=5
                )
                j = 0
                for index in distindex(arr_lnd.whole_space[from_worker_num]):
                    flat = to_flat(index)
                    out_index = to_out_index(flat)
                    out_worker = shardview.find_index(out_distribution, out_index)
                    if out_worker == self.worker_num:
                        base_index = shardview.index_to_base(
                            out_distribution[self.worker_num], out_index
                        )
                        out_lnd.bcontainer[base_index] = incoming_data[j]
                        j += 1

    def matmul(
        self,
        out_gid,
        out_size,
        out_distribution,
        a_gid,
        a_size,
        a_distribution,
        b_gid,
        b_size,
        b_distribution,
        bextend,
        a_send_recv,
        b_send_recv,
    ):
        start_worker_matmul = timer()
        dprint(3, "remote matmul", self.worker_num, out_gid, a_gid, b_gid)
        # print( "remote matmul", out_gid, a_gid, a_distribution, b_gid, b_distribution)
        def send_recv(
            thearray, array_distribution, from_range, to_range, send_recv_gid
        ):
            ret = []
            send_stats = []
            recv_stats = []

            for to_worker, from_worker, block_intersection, sview in to_range:
                if from_worker != to_worker:
                    start_time = timer()
                    local_slice = shardview.div_to_local(
                        array_distribution, block_intersection
                    )
                    dprint(
                        4,
                        "send matmul:",
                        to_worker,
                        from_worker,
                        block_intersection,
                        sview,
                        local_slice,
                        thearray.subspace,
                    )
                    data = self.get_partial_view(
                        thearray.gid,
                        local_slice,
                        array_distribution,
                        global_index=False,
                    )
                    self.comm_queues[to_worker].put(
                        (send_recv_gid, data, block_intersection, sview)
                    )
                    end_time = timer()
                    # print("sending",local_slice,data.shape,block_intersection,array_distribution)
                    send_stats.append((end_time - start_time, data.size))

            for to_worker, from_worker, block_intersection, sview in from_range:
                if from_worker == to_worker:
                    local_slice = shardview.div_to_local(
                        array_distribution, block_intersection
                    )
                    dprint(
                        4,
                        "from same matmul:",
                        to_worker,
                        from_worker,
                        block_intersection,
                        sview,
                        local_slice,
                    )
                    ret.append(
                        (
                            block_intersection,
                            self.get_partial_view(
                                thearray.gid,
                                local_slice,
                                array_distribution,
                                global_index=False,
                            ),
                        )
                    )
                else:
                    dprint(
                        4,
                        "from different matmul:",
                        to_worker,
                        from_worker,
                        block_intersection,
                        sview,
                    )
                    try:
                        start_time = timer()
                        (
                            incoming_send_recv_gid,
                            incoming_data,
                            in_block_intersection,
                            in_sview,
                        ) = self.comm_queues[to_worker].get(
                            gfilter=lambda x: x[0] == send_recv_gid, timeout=5
                        )
                        assert incoming_send_recv_gid == send_recv_gid
                        dprint(
                            4,
                            "receive matmul:",
                            incoming_data.shape,
                            in_block_intersection,
                        )
                        ret.append((in_block_intersection, incoming_data))
                        end_time = timer()
                        recv_stats.append((end_time - start_time, incoming_data.size))
                    except Exception:
                        print("some exception!", sys.exc_info()[0])
                        assert 0
            # print ("ret: ", ret)
            return ret, send_stats, recv_stats

        class FakeLocal:
            def __init__(self, shape, worker_num, bcontainer=None):
                self.whole = shape
                self.whole_space = shardview.divisions_to_distribution(
                    shardview.make_uni_dist_from_shape(num_workers, worker_num, shape)
                )
                if bcontainer is not None:
                    self.bcontainer = bcontainer
                else:
                    self.bcontainer = np.zeros(shape)
                self.core_slice = tuple(
                    [slice(None, None, None) for _ in range(len(shape))]
                )

            def get_partial_view(self, the_slice, dist, global_index=True):
                # assert(global_index == False)
                return self.bcontainer[the_slice]

            def get_view(self, shard):
                #print(shard, self.core_slice)
                return self.get_partial_view(
                    shardview.to_base_slice(shard), shard, global_index=False
                )

        a = self.numpy_map[a_gid]
        adiv = shardview.to_division(a_distribution[self.worker_num])
        if isinstance(b_gid, uuid.UUID):
            b = self.numpy_map[b_gid]
            bdiv = shardview.to_division(b_distribution[self.worker_num])
            # b_slice = tuple([slice(None,None)]*len(b.dim_lens))
            b_slice = tuple(
                [slice(bdiv[0][i], bdiv[1][i] + 1) for i in range(len(b.dim_lens))]
            )
        elif isinstance(b_gid, np.ndarray):
            b = FakeLocal(b_gid.shape, self.worker_num, bcontainer=b_gid)
            bdiv = shardview.to_division(b.whole_space[self.worker_num])
            b_distribution = b.whole_space
            if bextend:
                b.core_slice = (slice(adiv[0, 1], adiv[1, 1] + 1),)
                b_slice = (slice(adiv[0, 1], adiv[1, 1] + 1),)
            else:
                b.core_slice = (
                    slice(adiv[0, 1], adiv[1, 1] + 1),
                    slice(0, b_gid.shape[1]),
                )
                b_slice = (slice(adiv[0, 1], adiv[1, 1] + 1), slice(0, b_gid.shape[1]))
        else:
            assert 0

        if isinstance(out_gid, uuid.UUID):
            clocal = self.numpy_map[out_gid]
            # This is too much....should only do the slice portion.
            clocal.bcontainer.fill(0)
            cdiv = shardview.to_division(out_distribution[self.worker_num])
            # cdiv = shardview.to_division(clocal.subspace)
            partial_numpy = None
            cslice = self.get_view(out_gid, out_distribution[self.worker_num])
        elif isinstance(out_gid, tuple):
            clocal = FakeLocal(out_gid, self.worker_num)
            cdiv = shardview.to_division(clocal.whole_space[self.worker_num])
            partial_numpy = clocal.bcontainer
            # print("pre alocal:", self.worker_num, a_distribution[self.worker_num], type(a_distribution[self.worker_num]), a_distribution[self.worker_num].to_base_slice())
            # alocal = a.get_partial_view(shardview.to_base_slice(a_distribution[self.worker_num]), a_distribution[self.worker_num], global_index=False)
            alocal = a.get_view(a_distribution[self.worker_num])
            # print("pre blocal:", self.worker_num, b_distribution[self.worker_num], type(b_distribution[self.worker_num]), b_distribution[self.worker_num].to_base_slice())
            # blocal = b.get_partial_view(shardview.to_base_slice(b_distribution[self.worker_num]), b_distribution[self.worker_num], global_index=False)
            # blocal = b.get_view(b_distribution[self.worker_num])
            blocal = b.get_partial_view(b_slice, b_distribution[self.worker_num])
            nothing_to_do = all(x == 0 for x in alocal.shape) or all(
                x == 0 for x in blocal.shape
            )
            if not nothing_to_do and alocal.shape[1] != blocal.shape[0]:
                print("mismatch", self.worker_num, alocal.shape, blocal.shape)
                print(
                    "a_distribution:",
                    a_distribution,
                    "\n",
                    shardview.distribution_to_divisions(a_distribution),
                )
                print(
                    "b_distribution:",
                    b_distribution,
                    "\n",
                    shardview.distribution_to_divisions(b_distribution),
                )
                assert 0

            exec_start_time = timer()
            if not nothing_to_do:
                if bextend:
                    cslice_struct = (slice(adiv[0, 0], adiv[1, 0] + 1),)
                else:
                    cslice_struct = (
                        slice(adiv[0, 0], adiv[1, 0] + 1),
                        slice(0, out_gid[1]),
                    )
                clocal.bcontainer[cslice_struct] = alocal @ blocal
                # clocal.bcontainer[cslice_struct] = np.dot(alocal, blocal)
            #                print("special dot:", alocal.shape, blocal.shape)
            exec_end_time = timer()

            start_reduction_time = timer()
            if fast_reduction:
                # Distributed and Parallel Reduction
                max_worker = num_workers
                while max_worker > 1:
                    # midpoint is the index of the first worker that needs to send
                    midpoint = (max_worker + 1) // 2
                    # this worker will send
                    if self.worker_num >= midpoint:
                        send_to = self.worker_num - midpoint
                        # print("parreduction", max_worker, self.worker_num, "sending to", send_to)
                        self.comm_queues[send_to].put((a_send_recv, partial_numpy))
                        break
                    else:
                        # If the remaining workers is odd and this worker is the last one then it won't receive
                        # any communication or needs to do any compute but just goes to the next iteration where
                        # it will send.
                        if not (
                            max_worker % 2 == 1 and self.worker_num == midpoint - 1
                        ):
                            # print("parreduction", max_worker, self.worker_num, "receiving")
                            try:
                                incoming_uuid, incoming_partial = self.comm_queues[
                                    self.worker_num
                                ].get(gfilter=lambda x: x[0] == a_send_recv, timeout=5)
                            except Exception:
                                print("some exception!", sys.exc_info()[0])
                                assert 0
                            partial_numpy += incoming_partial

                    max_worker = midpoint

                if self.worker_num == 0:
                    # final_res = self.numpy_map[out_size]
                    # final_res.bcontainer[:] = partial_numpy
                    final_res = self.get_view(
                        out_size, out_distribution[self.worker_num]
                    )
                    final_res[:] = partial_numpy
                partial_numpy = 0

            end_worker_matmul = timer()
            return (
                self.worker_num,
                end_worker_matmul - start_worker_matmul,
                0,
                end_worker_matmul - start_reduction_time,
                0,
                0,
                exec_end_time - exec_start_time,
                [],
                [],
                [],
                [],
                partial_numpy,
            )
        else:
            assert 0

        cstartrow = cdiv[0, 0]
        cendrow = cdiv[1, 0]
        if not bextend:
            cstartcol = cdiv[0, 1]
            cendcol = cdiv[1, 1]
        else:
            cstartcol = None
            cendcol = None

        dprint(
            4,
            "remote matmul a subspace:",
            self.worker_num,
            a_distribution[self.worker_num],
            adiv,
        )
        dprint(
            4,
            "remote matmul b subspace:",
            self.worker_num,
            b_distribution[self.worker_num],
            bdiv,
        )
        dprint(
            4,
            "remote matmul c whole:",
            self.worker_num,
            clocal.whole,
            cdiv,
            cstartrow,
            cstartcol,
            out_distribution,
            clocal.whole_space,
        )

        # Compute which c indices need the chunk of a that this worker has.
        if bextend:
            send_a_indices = np.array([[adiv[0, 0]], [adiv[1, 0]]])
        else:
            send_a_indices = np.array(
                [[adiv[0, 0], 0], [adiv[1, 0], clocal.whole[1] - 1]]
            )
        dprint(4, "remote matmul send_a_indices:", self.worker_num, send_a_indices)
        # Compute which c indices need the chunk of b that this worker has.
        if bextend:
            send_b_indices = np.array([[bdiv[0, 0]], [bdiv[1, 0]]])
        else:
            send_b_indices = np.array(
                [[0, bdiv[0, 1]], [clocal.whole[0] - 1, bdiv[1, 1]]]
            )
        dprint(4, "remote matmul send_b_indices:", self.worker_num, send_b_indices)
        # Compute which a indices are needed by this worker.
        from_a_indices = np.array([[cstartrow, 0], [cendrow, a_size[1] - 1]])
        dprint(4, "remote matmul from_a_indices:", self.worker_num, from_a_indices)
        # Compute which a indices are needed by this worker.
        if bextend:
            from_b_indices = np.array([[0], [b_size[0] - 1]])
        else:
            from_b_indices = np.array([[0, cstartcol], [b_size[0] - 1, cendcol]])
        dprint(4, "remote matmul from_b_indices:", self.worker_num, from_b_indices)

        send_a_ranges = [
            [
                rr[0],
                self.worker_num,
                np.array([[rr[1][0, 0], adiv[0, 1]], [rr[1][1, 0], adiv[1, 1]]]),
                None,
            ]
            for rr in get_remote_ranges(clocal.whole_space, send_a_indices)
        ]
        if bextend:
            # send_b_ranges = [[i, self.worker_num, bdiv, None] for i in range(len(clocal.whole_space))]
            send_b_ranges = []
            for i in range(len(clocal.whole_space)):
                if not shardview.is_empty(clocal.whole_space[i]):
                    send_b_ranges.append([i, self.worker_num, bdiv, None])
        else:
            send_b_ranges = [
                [
                    rr[0],
                    self.worker_num,
                    np.array([[bdiv[0, 0], rr[1][0, 1]], [bdiv[1, 0], rr[1][1, 1]]]),
                    None,
                ]
                for rr in get_remote_ranges(clocal.whole_space, send_b_indices)
            ]

        def non_empty(x):
            divs = x[2]
            return not np.any(divs[0] > divs[1])

        send_a_ranges = list(filter(non_empty, send_a_ranges))
        send_b_ranges = list(filter(non_empty, send_b_ranges))

        # If this node computes nothing (meaning cdiv is empty) then it should receive nothing.
        if division_non_empty(cdiv):
            from_a_rr = list(
                filter(non_empty, get_remote_ranges(a_distribution, from_a_indices))
            )
            from_b_rr = list(
                filter(non_empty, get_remote_ranges(b_distribution, from_b_indices))
            )
        else:
            from_a_rr = []
            from_b_rr = []

        dprint(4, "remote matmul send_a_ranges:", self.worker_num, send_a_ranges)
        dprint(4, "remote matmul send_b_ranges:", self.worker_num, send_b_ranges)
        from_a_ranges = [[self.worker_num] + list(rr) for rr in from_a_rr]
        from_b_ranges = [[self.worker_num] + list(rr) for rr in from_b_rr]
        dprint(4, "remote matmul from_a_ranges:", self.worker_num, from_a_ranges)
        dprint(4, "remote matmul from_b_ranges:", self.worker_num, from_b_ranges)
        after_compute_comm = timer()

        aranges, a_send_stats, a_recv_stats = send_recv(
            a,
            a_distribution[self.worker_num],
            from_a_ranges,
            send_a_ranges,
            a_send_recv,
        )
        branges, b_send_stats, b_recv_stats = send_recv(
            b,
            b_distribution[self.worker_num],
            from_b_ranges,
            send_b_ranges,
            b_send_recv,
        )
        after_communication = timer()

        for adata in aranges:
            adataarray = adata[1]
            if shardview.division_to_shape(adata[0]) != adataarray.shape:
                print(
                    "shard wrong size!",
                    adataarray.shape,
                    adata[0],
                    shardview.division_to_shape(adata[0]),
                    self.worker_num,
                )
                assert 0
            for bdata in branges:
                bdataarray = bdata[1]
                if shardview.division_to_shape(bdata[0]) != bdataarray.shape:
                    print(
                        "shard wrong size!",
                        bdataarray.shape,
                        bdata[0],
                        shardview.division_to_shape(bdata[0]),
                        self.worker_num,
                    )
                    assert 0

                dprint(
                    4,
                    "adata, bdata:",
                    adata[0],
                    adataarray.shape,
                    bdata[0],
                    bdataarray.shape,
                )
                astartrow = adata[0][0, 0]
                aendrow = adata[0][1, 0] + 1
                astartcol = adata[0][0, 1]
                aendcol = adata[0][1, 1] + 1

                if bdata[0].shape[1] == 1:
                    bstartrow = bdata[0][0, 0]
                    bendrow = bdata[0][1, 0] + 1
                    bstartcol = 0
                    bendcol = 1
                else:
                    bstartrow = bdata[0][0, 0]
                    bendrow = bdata[0][1, 0] + 1
                    bstartcol = bdata[0][0, 1]
                    bendcol = bdata[0][1, 1] + 1

                kstart = max(astartcol, bstartrow)
                kend = min(aendcol, bendrow)

                if kend > kstart:
                    try:
                        if bextend:
                            d = cslice[astartrow - cstartrow : aendrow - cstartrow]
                            # d = clocal.bcontainer[astartrow-cstartrow:aendrow-cstartrow]
                        else:
                            d = cslice[
                                astartrow - cstartrow : aendrow - cstartrow,
                                bstartcol - cstartcol : bendcol - cstartcol,
                            ]
                            # d = clocal.bcontainer[astartrow-cstartrow:aendrow-cstartrow,bstartcol-cstartcol:bendcol-cstartcol]
                    except Exception:
                        print("subfail", cremote, sys.exc_info()[0])

                    try:
                        ktotal = kend - kstart
                        if bextend:
                            ashifted = adataarray[
                                :,
                                kstart
                                - adata[0][0, 1] : kstart
                                - adata[0][0, 1]
                                + ktotal,
                            ]
                            bshifted = bdataarray[
                                kstart
                                - bdata[0][0, 0] : kstart
                                - bdata[0][0, 0]
                                + ktotal
                            ]
                            d += ashifted @ bshifted
                            # d += np.dot(ashifted, bshifted)
                        #                            print("general dot:", ashifted.shape, bshifted.shape)
                        else:
                            ashifted = adataarray[
                                :,
                                kstart
                                - adata[0][0, 1] : kstart
                                - adata[0][0, 1]
                                + ktotal,
                            ]
                            bshifted = bdataarray[
                                kstart
                                - bdata[0][0, 0] : kstart
                                - bdata[0][0, 0]
                                + ktotal,
                                :,
                            ]
                            d += ashifted @ bshifted
                            # d += np.dot(ashifted, bshifted)
                    #                            print("general dot:", ashifted.shape, bshifted.shape)
                    except Exception:
                        print(self.worker_num, sys.exc_info()[0])
                        print(
                            "shifted shapes exception:",
                            adataarray.shape,
                            bdataarray.shape,
                            ashifted.shape,
                            bshifted.shape,
                            "kstart",
                            kstart,
                            "kend",
                            kend,
                            "ktotal",
                            ktotal,
                            adata[0][0, 1],
                            bdata[0][0, 0],
                            adata[0][0, 1] - kstart,
                            bdata[0][0, 0] - kstart,
                        )
                        print(
                            "cross-ranges:",
                            self.worker_num,
                            clocal.whole,
                            adata,
                            bdata,
                            "astartrow",
                            astartrow,
                            aendrow,
                            "bstartcol",
                            bstartcol,
                            bendcol,
                            "kstart",
                            kstart,
                            kend,
                            "cstart",
                            cstartrow,
                            cstartcol,
                            "indices",
                            astartrow - cstartrow,
                            aendrow - cstartrow,
                            bstartcol - cstartcol,
                            bendcol - bstartcol,
                        )
                        print(
                            "worker:",
                            self.worker_num,
                            adataarray.shape,
                            bdataarray.shape,
                        )
                        raise

        end_worker_matmul = timer()
        return (
            self.worker_num,
            end_worker_matmul - start_worker_matmul,
            after_compute_comm - start_worker_matmul,
            after_communication - after_compute_comm,
            len(aranges),
            len(branges),
            end_worker_matmul - after_communication,
            a_send_stats,
            a_recv_stats,
            b_send_stats,
            b_recv_stats,
            partial_numpy,
        )

    """

    # TODO: should use get_view
    def matmul(self, out_gid, a_gid, a_distribution, b_gid, b_distribution, bextend, a_send_recv, b_send_recv):
        async def send_shard(the_remote, thearray, array_distribution, to_worker, from_worker, block_intersection, sview, send_recv_gid):
            if from_worker != to_worker:
                local_slice = array_distribution.div_to_local(block_intersection)
                dprint(4, "send matmul:", to_worker, from_worker, block_intersection, sview, local_slice, thearray.subspace)
                data = the_remote.get_partial_view(thearray.gid, local_slice, array_distribution, global_index=False)
                await the_remote.comm_queues[to_worker].put_async((send_recv_gid, data, block_intersection, sview))
                #print("sending",local_slice,data.shape,block_intersection,array_distribution)

        async def recv_shard(the_remote, thearray, array_distribution, to_worker, from_worker, block_intersection, sview, send_recv_gid_a, send_recv_gid_b, received_a, received_b, ready, for_a):
            async def push_ready_pairs(received_a, received_b, ready, last_received_a):
                if last_received_a:
                    the_one = received_a[-1]
                    for bval in received_b:
                        await ready.put((the_one, bval))
                else:
                    the_one = received_b[-1]
                    for aval in received_a:
                        await ready.put((aval, the_one))

            if from_worker == to_worker:
                local_slice = array_distribution.div_to_local(block_intersection)
                dprint(4, "from same matmul:", to_worker, from_worker, block_intersection, sview, local_slice)
                if for_a:
                    received_a.append((block_intersection, the_remote.get_partial_view(thearray.gid, local_slice, array_distribution, global_index=False)))
                    await push_ready_pairs(received_a, received_b, ready, True)
                else:
                    received_b.append((block_intersection, the_remote.get_partial_view(thearray.gid, local_slice, array_distribution, global_index=False)))
                    await push_ready_pairs(received_a, received_b, ready, False)
            else:
                dprint(4, "from different matmul:", to_worker, from_worker, block_intersection, sview)
                try:
                    incoming_send_recv_gid, incoming_data, in_block_intersection, in_sview = await the_remote.comm_queues[to_worker].get_async(timeout = 5)
                    dprint(4, "receive matmul:", incoming_data.shape, in_block_intersection)
                    if incoming_send_recv_gid == send_recv_gid_a:
                        received_a.append((in_block_intersection, incoming_data))
                        await push_ready_pairs(received_a, received_b, ready, True)
                    else:
                        received_b.append((in_block_intersection, incoming_data))
                        await push_ready_pairs(received_a, received_b, ready, False)
                except Exception:
                    print("some exception!", sys.exc_info()[0])
                    assert(0)

        async def executor(ready, clocal, cstartrow, cstartcol):
            val = await ready.get()
            adata = val[0]
            bdata = val[1]

            adataarray = adata[1]
            if shardview.division_to_shape(adata[0]) != adataarray.shape:
                print("shard wrong size!", adataarray.shape, adata[0], shardview.division_to_shape(adata[0]), self.worker_num)
                assert(0)

            bdataarray = bdata[1]
            if shardview.division_to_shape(bdata[0]) != bdataarray.shape:
                print("shard wrong size!", bdataarray.shape, bdata[0], shardview.division_to_shape(bdata[0]), self.worker_num)
                assert(0)

            dprint(4, "adata, bdata:", adata[0], adataarray.shape, bdata[0], bdataarray.shape)
            astartrow = adata[0][0,0]
            aendrow = adata[0][1,0] + 1
            astartcol = adata[0][0,1]
            aendcol = adata[0][1,1] + 1

            if bdata[0].shape[1] == 1:
                bstartrow = bdata[0][0,0]
                bendrow = bdata[0][1,0] + 1
                bstartcol = 0
                bendcol = 1
            else:
                bstartrow = bdata[0][0,0]
                bendrow = bdata[0][1,0] + 1
                bstartcol = bdata[0][0,1]
                bendcol = bdata[0][1,1] + 1

            kstart = max(astartcol, bstartrow)
            kend = min(aendcol, bendrow)

            if kend > kstart:
                try:
                    d = clocal.bcontainer[astartrow-cstartrow:aendrow-cstartrow,bstartcol-cstartcol:bendcol-cstartcol]
                except Exception:
                    print("subfail", sys.exc_info()[0])

                try:
                    ktotal = kend - kstart
                    if bdata[0].shape[1] == 1:
                        ashifted = adataarray[:,kstart-adata[0][0,1]:kstart-adata[0][0,1]+ktotal]
                        bshifted = bdataarray[kstart-bdata[0][0,0]:kstart-bdata[0][0,0]+ktotal]
                        dotres = np.dot(ashifted, bshifted)
                        d += np.expand_dims(dotres, 1)
                    else:
                        ashifted = adataarray[:,kstart-adata[0][0,1]:kstart-adata[0][0,1]+ktotal]
                        bshifted = bdataarray[kstart-bdata[0][0,0]:kstart-bdata[0][0,0]+ktotal,:]
                        d += np.dot(ashifted, bshifted)
                except Exception:
                    print(self.worker_num, sys.exc_info()[0])
                    print("shifted shapes exception:", adataarray.shape, bdataarray.shape, ashifted.shape, bshifted.shape, "kstart", kstart, "kend", kend, "ktotal", ktotal, adata[0][0,1], bdata[0][0,0], adata[0][0,1]-kstart, bdata[0][0,0]-kstart)
                    print("cross-ranges:", self.worker_num, clocal.whole, adata, bdata, "astartrow", astartrow, aendrow, "bstartcol", bstartcol, bendcol, "kstart", kstart, kend, "cstart", cstartrow, cstartcol, "indices", astartrow-cstartrow, aendrow-cstartrow, bstartcol-cstartcol, bendcol-bstartcol)
                    print("worker:", self.worker_num, adataarray.shape, bdataarray.shape)
                    raise

        async def run():
            start_worker_matmul = timer()
            dprint(3, "remote matmul", out_gid, a_gid, b_gid)
            # To hold the combinations of adata and bdata ready to execute.
            ready = asyncio.Queue()
            # Holds the shards of adata and bdata received thus far.
            received_a = []
            received_b = []
            #print( "remote matmul", out_gid, a_gid, a_distribution, b_gid, b_distribution)

            a = self.numpy_map[a_gid]
            b = self.numpy_map[b_gid]
            clocal = self.numpy_map[out_gid]
            adiv = a_distribution[self.worker_num].to_division()
            bdiv = b_distribution[self.worker_num].to_division()
            cdiv = clocal.subspace.to_division()
            cstartrow = cdiv[0,0]
            cstartcol = cdiv[0,1]
            cendrow = cdiv[1,0]
            cendcol = cdiv[1,1]

            dprint(4, "remote matmul a subspace:", self.worker_num, a_distribution[self.worker_num], adiv)
            dprint(4, "remote matmul b subspace:", self.worker_num, b_distribution[self.worker_num], bdiv)
            dprint(4, "remote matmul c whole:", self.worker_num, clocal.whole, cdiv, cstartrow, cstartcol)

            # Compute which c indices need the chunk of a that this worker has.
            send_a_indices = np.array([[adiv[0,0], 0], [adiv[1,0], clocal.whole[1] - 1]])
            dprint(4, "remote matmul send_a_indices:", self.worker_num, send_a_indices)
            # Compute which c indices need the chunk of b that this worker has.
            if bextend:
                send_b_indices = np.array([[bdiv[0,0]], [bdiv[1,0]]])
            else:
                send_b_indices = np.array([[0, bdiv[0,1]], [clocal.whole[0] - 1, bdiv[1,1]]])
            dprint(4, "remote matmul send_b_indices:", self.worker_num, send_b_indices)
            # Compute which a indices are needed by this worker.
            from_a_indices = np.array([[cstartrow, 0], [cendrow, a.whole[1]]])
            dprint(4, "remote matmul from_a_indices:", self.worker_num, from_a_indices)
            # Compute which a indices are needed by this worker.
            if bextend:
                from_b_indices = np.array([[0], [b.whole[0]]])
            else:
                from_b_indices = np.array([[0, cstartcol], [b.whole[0], cendcol]])
            dprint(4, "remote matmul from_b_indices:", self.worker_num, from_b_indices)

            send_a_ranges = [[rr[0], self.worker_num, np.array([[rr[1][0,0],adiv[0,1]],[rr[1][1,0],adiv[1,1]]]), None] for rr in get_remote_ranges(clocal.whole_space, send_a_indices)]
            if bextend:
                send_b_ranges = [[i, self.worker_num, bdiv, None] for i in range(len(clocal.whole_space))]
            else:
                send_b_ranges = [[rr[0], self.worker_num, np.array([[bdiv[0,0], rr[1][0,1]],[bdiv[1,0], rr[1][1,1]]]), None] for rr in get_remote_ranges(clocal.whole_space, send_b_indices)]

            dprint(4, "remote matmul send_a_ranges:", self.worker_num, send_a_ranges)
            dprint(4, "remote matmul send_b_ranges:", self.worker_num, send_b_ranges)
            from_a_ranges = [[self.worker_num] + list(rr) for rr in get_remote_ranges(a_distribution, from_a_indices)]
            from_b_ranges = [[self.worker_num] + list(rr) for rr in get_remote_ranges(b_distribution, from_b_indices)]
            dprint(4, "remote matmul from_a_ranges:", self.worker_num, from_a_ranges)
            dprint(4, "remote matmul from_b_ranges:", self.worker_num, from_b_ranges)
            after_compute_comm = timer()

            senders = []
            receivers = []
            executors = []

            for to_worker, from_worker, block_intersection, sview in send_a_ranges:
                senders.append(asyncio.create_task(send_shard(self, a, a_distribution[self.worker_num], to_worker, from_worker, block_intersection, sview, a_send_recv)))
            for to_worker, from_worker, block_intersection, sview in send_b_ranges:
                senders.append(asyncio.create_task(send_shard(self, b, b_distribution[self.worker_num], to_worker, from_worker, block_intersection, sview, b_send_recv)))


            for to_worker, from_worker, block_intersection, sview in from_a_ranges:
                receivers.append(asyncio.create_task(recv_shard(self, a, a_distribution[self.worker_num], to_worker, from_worker, block_intersection, sview, a_send_recv, b_send_recv, received_a, received_b, ready, True)))
            for to_worker, from_worker, block_intersection, sview in from_b_ranges:
                receivers.append(asyncio.create_task(recv_shard(self, b, b_distribution[self.worker_num], to_worker, from_worker, block_intersection, sview, a_send_recv, b_send_recv, received_a, received_b, ready, False)))

            after_communication = timer()

            for _ in range(len(from_a_ranges) * len(from_b_ranges)):
                executors.append(asyncio.create_task(executor(ready, clocal, cstartrow, cstartcol)))

            await asyncio.gather(*senders)
            await asyncio.gather(*senders)
            await asyncio.gather(*executors)

            end_worker_matmul = timer()
            return (end_worker_matmul - start_worker_matmul, after_compute_comm - start_worker_matmul, after_communication - after_compute_comm, len(received_a), len(received_b), end_worker_matmul - after_communication)

        return asyncio.run(run())
    """

    def push_pull_copy(self, out_gid, from_ranges, to_ranges, send_recv_uuid):
        dprint(1, "push_pull_copy:", out_gid)
        out = self.numpy_map[out_gid]
        local_send = 0

        def write_to_out(out, incoming_data, map_out_combined, region):
            bires = map_out_combined[0] + region
            global_slice = tuple(
                [slice(bires[0][x], bires[1][x] + 1) for x in range(bires.shape[1])]
            )
            local_slice = shardview.slice_to_local(out.subspace, global_slice)
            dprint(
                2,
                "Receiving Data:",
                self.worker_num,
                map_out_combined,
                out.bcontainer.shape,
                incoming_data.shape,
                bires,
                global_slice,
                local_slice,
                region,
            )
            out.bcontainer[local_slice] = incoming_data

        for gid, remote_id, region, map_out_combined in to_ranges:
            other_local = self.numpy_map[gid]
            the_slice = tuple(
                [slice(region[0][x], region[1][x] + 1) for x in range(region.shape[1])]
            )
            local_slice = shardview.slice_to_local(other_local.subspace, the_slice)
            dprint(
                2,
                "Sending Data:",
                other_local.remote.worker_num,
                remote_id,
                the_slice,
                local_slice,
                map_out_combined,
            )
            if other_local.remote.worker_num == remote_id:
                local_send += 1
                write_to_out(
                    out, other_local.bcontainer[local_slice], map_out_combined, region
                )
            else:
                self.comm_queues[remote_id].put(
                    (
                        send_recv_uuid,
                        other_local.bcontainer[local_slice],
                        map_out_combined,
                        region,
                    )
                )

        for _ in range(len(from_ranges) - local_send):
            try:
                get_result = self.comm_queues[self.worker_num].get(
                    gfilter=lambda x: x[0] == send_recv_uuid, timeout=5
                )
                in_send_recv_uuid, incoming_data, map_out_combined, region = get_result
                # incoming_data, map_out_combined, region = self.comm_queues[self.worker_num].get(timeout = 5)
            except Exception:
                print("some exception!", sys.exc_info()[0])
                print("get_result:", get_result)
                assert 0
            write_to_out(out, incoming_data, map_out_combined, region)

    def sstencil(
        self, stencil_op_uuid, out_gid, neighborhood, first_gid, args, func, create_flag
    ):
        sstencil_start = timer()

        func = func_loads(func)
        first = self.numpy_map[first_gid]
        if create_flag:
            lnd = first.init_like(out_gid)
            self.numpy_map[out_gid] = lnd
        else:
            lnd = self.numpy_map[out_gid]
        new_bcontainer = lnd.bcontainer
        for i, x in enumerate(args):
            if isinstance(x, uuid.UUID):
                lnd_arg = self.numpy_map[x]
                lnd_arg.getborder(str(stencil_op_uuid) + str(i))
        #        print("sstencil:", first.remote.worker_num, func, type(func), first.subspace, first.bcontainer.shape, neighborhood)
        assert isinstance(func, StencilMetadata)
        # Convert from StencilMetadata to Numba StencilFunc
        fargs = [
            self.numpy_map[x].bcontainer if isinstance(x, uuid.UUID) else x
            for x in args
        ]

        worker_neighborhood = [
            (
                min(0, int(shardview.get_start(lnd.subspace)[x] + neighborhood[x][0])),
                max(
                    2 * lnd.border,
                    int(
                        shardview.get_start(lnd.subspace)[x]
                        + shardview.get_size(lnd.subspace)[x]
                        - lnd.whole[x]
                        + 2 * lnd.border
                        + neighborhood[x][1]
                    ),
                ),
            )
            for x in range(len(lnd.dim_lens))
        ]

        before_compile = timer()

        sfunc = func.compile({"neighborhood": tuple(worker_neighborhood)})
        #        print("sstencil fargs:", first.remote.worker_num, sfunc, type(sfunc), worker_neighborhood, "\n", fargs)
        after_compile = timer()

        if create_flag:
            sout = sfunc(*fargs)
            new_bcontainer[first.core_slice] = sout[first.core_slice]
        else:
            sfunc(*fargs, out=new_bcontainer)

        exec_time = timer()
        add_time("sstencil_prep", before_compile - sstencil_start)
        add_time("sstencil_compile", after_compile - before_compile)
        add_time("sstencil_exec", exec_time - after_compile)
        return (
            before_compile - sstencil_start,
            after_compile - before_compile,
            exec_time - after_compile,
            exec_time - sstencil_start
        )

    #        print("sstencil sout:", first.remote.worker_num, sout)
    #        new_bcontainer[first.core_slice] = sout[first.core_slice]
    #        new_bcontainer[:] = sfunc(*fargs, out=new_bcontainer)
    #        print("sstencil done", first.remote.worker_num, "\n", new_bcontainer)

    # TODO: should use get_view
    def scumulative_local(self, out_gid, in_gid, func):
        func = func_loads(func)
        in_bcontainer = self.numpy_map[in_gid]
        self.numpy_map[out_gid] = in_bcontainer.init_like(out_gid)
        new_bcontainer = self.numpy_map[out_gid].bcontainer
        in_array = in_bcontainer.bcontainer
        new_bcontainer[0] = in_array[0]
        for index in range(1, in_bcontainer.dim_lens[0]):
            new_bcontainer[index] = func(new_bcontainer[index - 1], in_array[index])

    # TODO: should use get_view
    def scumulative_final(self, array_gid, boundary_value, func):
        func = func_loads(func)
        in_bcontainer = self.numpy_map[array_gid]
        in_array = in_bcontainer.bcontainer
        boundary_value = boundary_value[0]
        for index in range(in_bcontainer.dim_lens[0]):
            in_array[index] = func(in_array[index], boundary_value)

    ## still needed?
    def array_binop(self, lhs_gid, rhs_gid, out_gid, op):
        lhs = self.numpy_map[lhs_gid]
        if isinstance(rhs_gid, uuid.UUID):
            rhs = self.numpy_map[rhs_gid].bcontainer[lhs.core_slice]
        else:
            rhs = rhs_gid
        binop = getattr(lhs.bcontainer[lhs.core_slice], op)
        if out_gid is not None:
            lnd = lhs.init_like(out_gid)
            self.numpy_map[out_gid] = lnd
            new_bcontainer = lnd.bcontainer
            new_bcontainer[lnd.core_slice] = binop(rhs)
        else:
            binop(rhs)

    def spmd(self, func, args):
        dprint(2, "Starting remote spmd", self.worker_num)
        try:
            func = func_loads(func)
            fargs = [self.numpy_map[x] if isinstance(x, uuid.UUID) else x for x in args]
            fmfunc = FunctionMetadata(func, [], {})
            dprint(3, "Before local spmd fmfunc call")
            fmfunc(*fargs)
            dprint(3, "After local spmd fmfunc call")
        except Exception:
            print("some exception in remote spmd")
            traceback.print_exc()
            pass
        sys.stdout.flush()

    def run_deferred_ops(
        self, uuid, arrays, delete_gids, pickledvars, exec_dist, fname, code, imports
    ):
        times = [timer()]
        dprint(4, "HERE - deferredops; arrays:", arrays.keys())
        subspace = shardview.clean_range(
            exec_dist[self.worker_num]
        )  # our slice of work range
        # create array shards if needed
        # TODO:  This has become very complicated and ugly.  Consider restructuring, use class instead of tuple.
        # info tuple is (size, distribution, local_border, from_border, to_border, dtype)
        # TODO:  Need to check array construction -- this code may break for views/slices; assumes first view of gid is the canonical, full array
        # [ self.create_array(g, subspace, info[0], None, info[2], info[1], None if info[3] is None else info[3][self.worker_num], None if info[4] is None else info[4][self.worker_num]) for (g,(v,info,bdist,pad)) in arrays.items() if g not in self.numpy_map ]
        for (g, (l, bdist, pad, _)) in arrays.items():
            if g in self.numpy_map:
                continue
            v = l[0][0]
            info = l[0][1]
            self.create_array(
                g,
                subspace,
                info[0],
                None,
                info[2],
                info[5],
                info[1],
                None if info[3] is None else info[3][self.worker_num],
                None if info[4] is None else info[4][self.worker_num],
            )

        times.append(timer())
        # check if function exists, else exec to create it
        ldict = {}
        gdict = globals()
        for imp in imports:
            the_module = __import__(imp)
            gdict[imp] = the_module

        # gdict=sys.modules['__main__'].__dict__
        if fname not in gdict:
            dprint(2, "function does not exist, creating it\n", code)
            exec(code, gdict, ldict)
            gdict[fname] = FunctionMetadata(ldict[fname], [], {})
            # print (gdict.keys())
        func = gdict[fname]
        times.append(timer())

        # Send data to other workers as needed, count how many items to recieve later
        # TODO: optimize by combining messages of overlapping array parts
        arr_parts = (
            {}
        )  # place to keep array parts received and local parts, indexed by variable name
        expected_bits ={} # set of parts that will be sent to this worker from others
        to_send = {}  # save up list of messages to node i, send all as single message
        from_set = {}  # nodes to recieve from
        overlap_time = 0.0
        getview_time = 0.0
        arrview_time = 0.0
        copy_time = 0.0
        for (g, (l, bdist, pad, _)) in arrays.items():
            for (v, info) in l:
                arr_parts[v] = []
                if shardview.is_compat(
                    subspace, info[1][self.worker_num]
                ):  # whole compute range is local
                    sl = self.get_view(g, info[1][self.worker_num])
                    arr_parts[v].append((subspace, sl))
                    continue
                overlap_time -= timer()
                overlap_workers = shardview.get_overlaps(
                    self.worker_num, info[1], exec_dist
                )
                overlap_time += timer()
                # for i in range(num_workers):
                for i in overlap_workers:
                    if i != self.worker_num:
                        # part = shardview.intersect(exec_dist[self.worker_num],info[1][i])  # part of what we need but located at worker i
                        part = shardview.intersect(
                            info[1][i], exec_dist[self.worker_num]
                        )  # part of what we need but located at worker i
                        if not shardview.is_empty(part):
                            if v in expected_bits:
                                expected_bits[v] += [(part, info[1][self.worker_num])]
                            else:
                                expected_bits[v] = [(part, info[1][self.worker_num])]
                            from_set[i] = 1
                    # part = shardview.intersect(exec_dist[i],info[1][self.worker_num])  # part of what we have needed at worker i
                    part = shardview.intersect(
                        info[1][self.worker_num], exec_dist[i]
                    )  # part of what we have needed at worker i
                    if shardview.is_empty(part):
                        continue
                    getview_time -= timer()
                    #sl = self.get_partial_view( g, shardview.to_slice(part), info[1][self.worker_num], remap_view=False,)
                    getview_time += timer()
                    if i == self.worker_num:
                        # arr_parts[v].append( (part, sl) )
                        # arr_parts[v].append( (part, shardview.array_to_view(part, sl)) )
                        arrview_time -= timer()
                        sl = self.get_partial_view( g, shardview.to_slice(part), info[1][self.worker_num], remap_view=False,)
                        arr_parts[v].append( ( shardview.clean_range(part), shardview.array_to_view(part, sl),) )
                        arrview_time += timer()
                        dprint(
                            2,
                            "Worker",
                            self.worker_num,
                            "Keeping local part",
                            part,
                            sl,
                            info[1],
                        )
                    else:
                        # deferred send data to worker i
                        # self.comm_queues[i].put( (uuid, v, part, sl) )
                        if i not in to_send:
                            to_send[i] = []
                        copy_time -= timer()
                        # to_send[i].append( (v, part, sl) )
                        # to_send[i].append( (v, part, sl.copy()))  # pickling a slice is slower than copy+pickle !!
                        merge = False
                        base_part = shardview.as_base(info[1][self.worker_num], part)
                        for j in range(len(to_send[i])):
                            if to_send[i][j][0] == g:
                                # TODO:  should check if worth merging or sending separately
                                to_send[i][j][1] = shardview.union( to_send[i][j][1], base_part )
                                to_send[i][j] += [(v, part, base_part)]
                                merge = True
                                break
                        if not merge:
                            to_send[i].append([g, base_part, (v, part, base_part)])
                        copy_time += timer()
                        dprint(
                            2,
                            "Worker",
                            self.worker_num,
                            "Sending to worker",
                            i,
                            part,
                            #sl,
                            info[1],
                        )
        times.append(timer())

        # actual sends
        for (i, vl) in to_send.items():
            msg = []
            for v in vl: 
                g = v[0]
                base_part = v[1]
                v = v[2:]
                sl = self.get_partial_view( g, shardview.to_slice(base_part), None, global_index=False, remap_view=False )
                msg.append((sl.copy(), base_part, v))
            self.comm_queues[i].put((uuid, self.worker_num, msg))

        expected_parts = len( from_set)  # since messages are coalesced, expect 1 from each node sending to us

        times.append(timer())
        othervars = {v: pickle.loads(val) for (v, val) in pickledvars}
        # Convert 0d arrays to scalars since 0d support may otherwise be spotty in Numba.
        for k, v in othervars.items():
            if isinstance(v, np.ndarray) and v.shape == ():  # 0d array
                # convert to scalar
                othervars[k] = v.item()

        times.append(timer())
        # Receive data from other workers
        # for i in range(expected_parts):
        #    try:
        #        _, v, part, sl = self.comm_queues[self.worker_num].get(gfilter=lambda x: x[0]==uuid, timeout=5)
        #        arr_parts[v].append( (part, sl) )
        #    except Exception:
        #        print("some exception", sys.exc_info()[0])
        #        assert(0)
        msgs = []
        while expected_parts > 0:
            m = self.comm_queues[self.worker_num].multi_get(
                expected_parts,
                gfilter=lambda x: x[0] == uuid,
                timeout=5,
                print_times=(ntiming >= 1 and self.worker_num == timing_debug_worker),
                msginfo=lambda x: "[from " + str(x[1]) + "]",
            )
            msgs += m
            expected_parts -= len(m)
            if expected_parts > 0:
                print("Still waiting for", expected_parts, "items")
        # for _,v,part,sl in msgs:
        #    arr_parts[v].append( (part, sl) )
        for _, _, m in msgs:
            for sl, full_part, l in m:
                for v, part, base_part in l:
                    # arr_parts[v].append( (part, sl) )
                    # arr_parts[v].append( (part, shardview.array_to_view(part, sl)) )
                    arrview_time -= timer()
                    x = shardview.get_start(base_part)
                    x -= shardview.get_start(full_part)
                    sl2 = sl[ shardview.to_slice(base_part) ]
                    arr_parts[v].append(
                        (shardview.clean_range(part), shardview.array_to_view(part, sl2))
                    )
                    arrview_time += timer()

        times.append(timer())
        # Construct set of ranges and array parts to use in each

        vparts = numba.typed.List()
        for _, pl in arr_parts.items():
            for vpart, _ in pl:
                vparts.append(vpart)
        ranges = shardview.get_range_splits_list(vparts)
        times.append(timer())
        rangedvars = [{} for _ in ranges]
        # varlist={}
        for i, rpart in enumerate(ranges):
            varlist = rangedvars[i]
            # varlist.clear()
            for (varname, partlist) in arr_parts.items():
                for (vpart, data) in partlist:
                    if not shardview.overlaps(rpart, vpart):  # skip
                        continue
                    # times.append(timer())
                    # slindex = shardview.to_base_slice(shardview.mapsv(vpart,rpart))
                    tmp = shardview.mapsv(vpart, rpart)
                    # times.append(timer())
                    # slindex = shardview.to_base_slice(tmp)
                    # times.append(timer())
                    # varlist[varname] = data[slindex]
                    varlist[varname] = shardview.get_base_slice(tmp, data)
            # times.append(timer())
            # if not shardview.is_empty(rpart):
            #    func( **varlist, **othervars )
            # times.append(timer())

        #######
        # ranges = [ (subspace, {}) ]
        # for (varname, partlist) in arr_parts.items():
        #    i=0;
        #    while i<len(partlist):
        #        vpart, data = partlist[i]
        #        i+=1
        #        j=0
        #        while j<len(ranges):
        #            rpart,varlist = ranges[j]
        #            if not shardview.overlaps(rpart,vpart):   # skip
        #                j+=1
        #                continue
        #            if shardview.is_compat(rpart,vpart):  # matches range
        #                varlist[varname] = data
        #                break
        #            # does not match;  split the ranges into parts, add to lists
        #            rparts, vparts = shardview.get_range_splits(rpart,vpart)
        #            if len(rparts)>1:
        #                #ranges[j] = ( rparts[0], { v: d[shardview.to_base_slice(shardview.mapslice(rpart,shardview.to_slice(rparts[0])))] for (v,d) in varlist.items() } )
        #                #ranges[j] = ( rparts[0], { v: d[shardview.to_base_slice(shardview.mapsv(rpart,rparts[0]))] for (v,d) in varlist.items() } )
        #                slindex = shardview.to_base_slice(shardview.mapsv(rpart,rparts[0]))
        #                ranges[j] = ( rparts[0], { v: d[slindex] for (v,d) in varlist.items() } )
        #                for r in rparts[1:]:
        #                    #ranges.append( (r, {v: d[shardview.to_base_slice(shardview.mapslice(rpart,shardview.to_slice(r)))] for (v,d) in varlist.items() } ) )
        #                    #ranges.append( (r, {v: d[shardview.to_base_slice(shardview.mapsv(rpart,r))] for (v,d) in varlist.items() } ) )
        #                    slindex = shardview.to_base_slice(shardview.mapsv(rpart,r))
        #                    ranges.append( (r, {v: d[slindex] for (v,d) in varlist.items() } ) )
        #            if len(vparts)>1:
        #                for r in vparts:
        #                    #partlist.append( (r, data[shardview.to_base_slice(shardview.mapslice(vpart,shardview.to_slice(r)))]) )
        #                    partlist.append( (r, data[shardview.to_base_slice(shardview.mapsv(vpart,r))]) )
        #                break
        #            # if we get here, should continue with same j value (repeat with updated ranges)

        times.append(timer())
        if len(ranges) > 1:
            dprint(2, "Ranges:", len(ranges))

        # execute function in each range
        # for (r,arrvars) in ranges:
        for i, r in enumerate(ranges):
            arrvars = rangedvars[i]
            if not shardview.is_empty(r):
                for k, v in arrvars.items():
                    dprint(4, "inputs:", k, v, type(v))
                for k, v in othervars.items():
                    dprint(4, "others:", k, v, type(v))

                func(**arrvars, **othervars)
                for k, v in arrvars.items():
                    dprint(4, "results:", k, v, type(v))
        times.append(timer())

        # delete arrays no longer needed
        [self.destroy_array(g) for g in delete_gids]

        times.append(timer())
        if not USE_ZMQ and USE_MPI:
            ramba_queue.wait_sends()
        tnow = timer()
        times.append(tnow)
        self.deferred_ops_time += tnow - times[0]
        self.deferred_ops_count += 1
        if ntiming >= 1 and self.worker_num == timing_debug_worker:
            times = [
                int((times[i] - times[i - 1]) * 1000000) / 1000
                for i in range(1, len(times))
            ]
            tprint(
                1,
                "Deferred execution",
                (tnow - self.tlast) * 1000,
                times,
                int(overlap_time * 1000000) / 1000,
                int(getview_time * 1000000) / 1000,
                int(arrview_time * 1000000) / 1000,
                int(copy_time * 1000000) / 1000,
            )
            tprint(2, code)
        self.tlast = tnow

    def nop(self):
        return

    def seed(self, x):
        np.random.seed(x + self.worker_num)

    def get_comm_stats(self):
        return [x.get_stats() for x in self.comm_queues]

    def reset_def_ops_stats(self):
        self.deferred_ops_time = 0
        self.deferred_ops_count = 0

    def get_def_ops_stats(self):
        return (self.deferred_ops_count, self.deferred_ops_time)

    def reset_compile_stats(self):
        if numba.version_info.short >= (0, 53):
            self.compile_recorder.buffer = []

    def get_compile_stats(self):
        if numba.version_info.short >= (0, 53):
            return rec_buf_summary(self.compile_recorder.buffer)
        else:
            return 0

    def rpc_serve(self, rvq):
        # z = numa.set_affinity(self.worker_num, numa_zones)
        print_stuff = timing > 2 and self.worker_num == 0
        self.up_rvq = rvq
        if self.is_aggregator:
            if self.my_rvq is None:
                self.my_rvq = ramba_queue.Queue(hint_ip=hint_ip, tag=2)
            msg = ramba_queue.pickle(("RPC", "set_up_rvq", None, [self.my_rvq], {}))
            [self.control_queues[i].put(msg, raw=True) for i in self.children]
        t0 = timer()
        while True:
            if self.is_aggregator:
                msg = self.my_control_queue.get(raw=True, print_times=print_stuff)
                rpctype, method, rvid, args, kwargs = ramba_queue.unpickle(msg)
                if rpctype == "RPC":
                    [self.control_queues[i].put(msg, raw=True) for i in self.children]
            else:
                rpctype, method, rvid, args, kwargs = self.my_control_queue.get(
                    gfilter=lambda x: x[0] == "RPC" or x[0] == "RPC1",
                    print_times=print_stuff,
                )

            if method == "END":
                break
            t1 = timer()
            try:
                op = getattr(self, method)
                retval = op(*args, **kwargs)
            except Exception:
                traceback.print_exc()
                rvq.put(("ERROR", self.worker_num, traceback.format_exc()))
                break
            t2 = timer()
            if rvid is not None:
                if rpctype == "RPC":
                    self.up_rvq.put((rvid + str(self.worker_num), retval))
                else:
                    rvq.put((rvid + str(self.worker_num), retval))
            t3 = timer()
            if self.is_aggregator and rvid is not None and rpctype == "RPC":
                for _ in self.children:
                    msg = self.my_rvq.get(raw=True)
                    self.up_rvq.put(msg, raw=True)
            t4 = timer()
            if print_stuff:
                print(
                    "Command ",
                    method,
                    "get",
                    (t1 - t0) * 1000,
                    "exec",
                    (t2 - t1) * 1000,
                    "ret",
                    (t3 - t2) * 1000,
                    "fwd",
                    (t4 - t3) * 1000,
                )
            t0 = t4

    def mgrid(self, out_gid, out_size, out_distribution):
        divs = shardview.distribution_to_divisions(out_distribution)[self.worker_num]
        dprint(2, "mgrid remote:", self.worker_num, divs)
        if np.all(divs[0] <= divs[1]):
            out_lnd = self.numpy_map[out_gid]
            bcontainer = out_lnd.bcontainer[out_lnd.core_slice]
            mslice = [
                slice(divs[0][i + 1], divs[1][i + 1] + 1)
                for i in range(divs.shape[1] - 1)
            ]
            dprint(2, "mslice", mslice, bcontainer.shape)
            bcontainer[:] = np.mgrid[mslice]

    def load(self, gid, fname, ftype=None, **kwargs):
        lnd = self.numpy_map[gid]
        fldr = fileio.get_load_handler(fname, ftype)
        fldr.read(fname, lnd.bcontainer, shardview.to_slice(lnd.subspace), **kwargs)


if not USE_MPI:
    RemoteState = ray.remote(num_cpus=num_threads)(RemoteState)


# Wrappers to abstract away Ray method calls

ALL_NODES = -1


def _real_remote(nodeid, method, has_retval, args, kwargs):
    if nodeid == ALL_NODES:
        if USE_RAY_CALLS:
            rargs = [ray.put(v) for v in args]
            rkwargs = {k: ray.put(v) for k, v in kwargs.items()}
            return [
                getattr(remote_states[i], method).remote(*rargs, **rkwargs)
                for i in range(num_workers)
            ]
        rvid = str(uuid.uuid4()) if has_retval else None
        if USE_ZMQ or not USE_BCAST:
            msg = ramba_queue.pickle(("RPC", method, rvid, args, kwargs))
            if ntiming >= 1:
                print(
                    "control message size: ",
                    sum(
                        [
                            len(i.raw())
                            if isinstance(i, pickle.PickleBuffer)
                            else len(i)
                            for i in msg
                        ]
                    )
                    if isinstance(msg, list)
                    else len(msg),
                )
        else:
            msg = ("RPC", method, rvid, args, kwargs)
        if USE_MPI and USE_BCAST and not USE_ZMQ:
            ramba_queue.bcast(msg)
        elif USE_ZMQ and USE_BCAST:
            [control_queues[i].put(msg, raw=True) for i in aggregators]
        else:
            [control_queues[i].put(msg, raw=True) for i in range(num_workers)]
        return [rvid + str(i) for i in range(num_workers)] if has_retval else None
    if USE_RAY_CALLS:
        rop = getattr(remote_states[nodeid], method)
        return rop.remote(*args, **kwargs)
    rvid = str(uuid.uuid4()) if has_retval else None
    control_queues[nodeid].put(("RPC1", method, rvid, args, kwargs))
    return rvid + str(nodeid) if has_retval else None


def get_results(refs):
    if USE_RAY_CALLS:
        return ray.get(refs)
    if isinstance(refs, list):
        rv = [get_results(v) for v in refs]
    else:
        rv = retval_queue.get(gfilter=lambda x: x[0] == refs or x[0] == "ERROR")
        if rv[0] == "ERROR":
            raise Exception("Exception in Remote Worker " + str(rv[1]) + "\n" + rv[2])
        rv = rv[1]
    return rv


def remote_exec(nodeid, method, *args, **kwargs):
    _real_remote(nodeid, method, False, args, kwargs)


def remote_async_call(nodeid, method, *args, **kwargs):
    return _real_remote(nodeid, method, True, args, kwargs)


def remote_call(nodeid, method, *args, **kwargs):
    return get_results(_real_remote(nodeid, method, True, args, kwargs))


def remote_exec_all(method, *args, **kwargs):
    # [ _real_remote(i, method, False, args, kwargs) for i in range(num_workers) ]
    _real_remote(ALL_NODES, method, False, args, kwargs)
    # get_results(_real_remote(ALL_NODES, method, True, args, kwargs))


def remote_async_call_all(method, *args, **kwargs):
    # return [ _real_remote(i, method, True, args, kwargs) for i in range(num_workers) ]
    return _real_remote(ALL_NODES, method, True, args, kwargs)


def remote_call_all(method, *args, **kwargs):
    # return get_results([ _real_remote(i, method, True, args, kwargs) for i in range(num_workers) ])
    return get_results(_real_remote(ALL_NODES, method, True, args, kwargs))


if USE_RAY_CALLS:

    def func_dumps(f):
        return f

    def func_loads(f):
        return f


else:
    func_dumps = cloudpickle.dumps
    func_loads = cloudpickle.loads


# *******************************
# ** Initialize Remote Workers **
# *******************************
import atexit

if USE_MPI:
    # MPI setup
    # Queue tags: 0 = general comm, 1 = control, 2 = reply
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == num_workers:  # driver
        # do this stuff only once
        def do_init(done=[]):
            if len(done) > 0:
                print("HERE -- already done")
                return None
            done += [1]
            rv_q = ramba_queue.Queue(tag=2)
            comm_q = comm.allgather(0)[:-1]  # ignore our own invalid one
            con_q = comm.allgather(rv_q)
            aggr = get_aggregators(comm_q)
            return rv_q, comm_q, con_q, aggr

        x = do_init()
        if x is not None:
            retval_queue, comm_queues, control_queues, aggregators = x
            atexit.register(remote_exec_all, "END")

    else:  # workers -- should never leave this section!
        RS = RemoteState(rank, get_common_state())
        con_q = RS.get_control_queue()
        comm_q = RS.get_comm_queue()
        comm_queues = comm.allgather(comm_q)[:-1]  # ignore invalid one for controller
        control_queues = comm.allgather(con_q)
        rv_q = control_queues[num_workers]
        RS.set_comm_queues(comm_queues, control_queues)
        RS.rpc_serve(rv_q)
        sys.exit()

else:  # Ray setup
    # Start remote Actors, but only if this was the process that initialized Ray;
    # This avoids starting the remote workers every time ramba.py is imported
    if ray_first_init:
        dprint(1, "Constructing RemoteState actors")
        from ray.util.placement_group import placement_group

        res = [{"CPU": num_threads}] * num_workers
        pg = placement_group(res, strategy="SPREAD")
        remote_states = [
            RemoteState.options(placement_group=pg).remote(x, get_common_state())
            for x in range(num_workers)
        ]
        control_queues = ray.get([x.get_control_queue.remote() for x in remote_states])
        comm_queues = ray.get([x.get_comm_queue.remote() for x in remote_states])
        aggregators = get_aggregators(comm_queues)
        [x.set_comm_queues.remote(comm_queues, control_queues) for x in remote_states]
        if not USE_RAY_CALLS:
            retval_queue = ramba_queue.Queue()
            [x.rpc_serve.remote(retval_queue) for x in remote_states]


def print_comm_stats():
    if not USE_ZMQ:
        return
    stats = remote_call_all("get_comm_stats")
    totals = {}
    pickle_times = {}
    unpickle_times = {}
    for i, l in enumerate(stats):
        ips = l[i][0]
        for ipd, _, _, sent, unpick, pick in l:
            if (ips, ipd) not in totals:
                totals[(ips, ipd)] = 0
            totals[(ips, ipd)] += sent
            if (ips, ipd) not in pickle_times:
                pickle_times[(ips, ipd)] = 0.0
            pickle_times[(ips, ipd)] += pick
            if ips not in unpickle_times:
                unpickle_times[ips] = 0.0
            unpickle_times[ips] += unpick

    print(totals)
    print(pickle_times)
    print(unpickle_times)


## track live ndarray gids.  map gid to [ref_count, remote_constructed flag, weakset of ndarray references]
# ndarray_gids = {}


def find_index(distribution, index):
    if isinstance(index, int):
        index = (index,)

    dprint(3, "find_index:", distribution, index)
    dshape = distribution.shape
    assert dshape[2] == len(index)

    for i in range(dshape[0]):
        for j in range(dshape[2]):
            if index[j] < distribution[i][0][j] or index[j] > distribution[i][1][j]:
                break
        else:
            return i


HANDLED_FUNCTIONS = {}


def unify_args(lhs, rhs, dtype):
    if hasattr(rhs, "dtype"):
        rhs_dtype = rhs.dtype
    else:
        try:
            rhs_dtype = np.dtype(rhs)
        except Exception:
            rhs_dtype = None
    if dtype is None:
        dtype = np.result_type(lhs, rhs_dtype)
    elif dtype == "float":
        dtype = (
            np.float32
            if rhs_dtype == np.float32 and lhs == np.float32
            else np.float64
        )
    return dtype


def get_ndarrays(item, outlist):
    for i in item:
        if isinstance(i, (list, tuple)):
            get_ndarrays(i, outlist)
        elif isinstance(i, ndarray):
            outlist.append(i)


# DAG stuff
class DAG:
    dag_nodes = weakref.WeakSet()
    in_evaluate = 0

    def __init__(self, name, executor, inplace, ndarray_deps, args, kwargs, executed=False, *, output=None):
        self.name = name
        self.executor = executor
        self.inplace = inplace
        self.args = args
        self.kwargs = kwargs
        self.forward_deps = []
        self.backward_deps = [x.dag for x in ndarray_deps]
        self.executed = executed
        self.output = output

    def __repr__(self):
        return f"DAG({self.name}, {self.executed})"

    @classmethod
    def add(cls, delayed, name, executor, args, kwargs):
        ndarray_deps = []
        get_ndarrays(list(args), ndarray_deps)
        get_ndarrays(list(kwargs.values()), ndarray_deps)
        dag = DAG(name, executor, delayed.inplace, ndarray_deps, args, kwargs)
        if delayed.inplace:
            nres = delayed.inplace
            nres.idag = dag
        else:
            nres = ndarray(delayed.shape, dtype=delayed.dtype, dag=dag)
        dag.output = weakref.ref(nres)
        for dag_node in dag.backward_deps:
            dag_node.forward_deps.append(dag)
        cls.dag_nodes.add(dag)
        dprint(2, "DAG.add", name, len(ndarray_deps), id(nres))
        return nres, dag

    @classmethod
    def one_creator(cls, dag):
        return len(dag.backward_deps) == 1

    @classmethod
    def creator(cls, dag):
        assert len(dag.backward_deps) == 1
        return dag.backward_deps[0]

    @classmethod
    def depth_first_traverse(cls, dag_node, depth_first_nodes, dag_node_processed):
        dprint(2, "dag_node:", id(dag_node.output()), dag_node.name, dag_node.args)
        if dag_node in dag_node_processed:
            return
        dag_node_processed.add(dag_node)
        if dag_node.executed:
            return

        # stack-mean-advindex to groupby
        if dag_node.name == "stack":
            stack_dag = dag_node
            stack_arrays = stack_dag.args[0]
            stack_res = stack_dag.output()
            stack_preds = stack_dag.backward_deps
            # Make sure that the number of arrays matched the axis dimension of stack_res
            assert len(stack_arrays) == stack_res.shape[stack_dag.kwargs["axis"]]
            assert len(stack_arrays) == len(stack_preds)
            # If all the arrays part of stack have not been constructed yet and they are all for nanmean calls.
            if all([x.name == "nanmean" for x in stack_preds]):
                nanmeans = stack_preds
                assert isinstance(nanmeans[0], DAG)
                # Get the first array's join axis
                join_axis = nanmeans[0].kwargs["axis"]
                dprint(3, "join_axis:", join_axis)
                # If all the arrays use the same join axis in nanmean
                if (all([x.kwargs["axis"] == join_axis for x in nanmeans]) and
                    all([DAG.one_creator(x) for x in nanmeans]) and
                    all([DAG.creator(x).name == "getitem_array" for x in nanmeans])
                ):
                    dprint(3, "All axis match and each nanmean array has one creator that is a getitem_array")
                    getitem_ops = [DAG.creator(x) for x in nanmeans]
                    assert isinstance(getitem_ops[0], DAG)
                    # Get the axis that the first getitem operates on
                    getitem_axis = get_advindex_dim(getitem_ops[0].args[1])
                    dprint(3, "getitem_axis:", getitem_axis)
                    # If all the getitems have the same axis
                    if all([get_advindex_dim(x.args[1]) == getitem_axis for x in getitem_ops]):
                        dprint(3, "All getitems operate on the same axis")
                        getitem_arrays = [x.args[0] for x in getitem_ops]
                        assert isinstance(getitem_arrays[0], ndarray)
                        # Get the array on which getitem operates for the first stack array
                        orig_array = getitem_arrays[0]
                        # If all the getitems operate on the same array
                        if all([x is orig_array for x in getitem_arrays]):
                            dprint(3, "All getitems operate on the same array")
                            slot_indices = [x.args[1][getitem_axis] for x in getitem_ops]
                            dprint(3, "slot_indices:", slot_indices)
                            group_array = np.full(orig_array.shape[getitem_axis], -1, dtype=np.int)
                            for i in range(len(slot_indices)):
                                for vi in slot_indices[i]:
                                    group_array[vi] = i
                            with np.printoptions(threshold=np.inf):
                                dprint(3, "group_array:", group_array)

                            def run_and_post(temp_array, orig_array, getitem_axis, group_array, slot_indices):
                                print("run_and_post stack", id(temp_array))
                                gb = orig_array.groupby(getitem_axis, group_array, num_groups=len(slot_indices))
                                res = gb.nanmean()
                                return fromarray(np.moveaxis(res, -1, 0))
                                #temp_array.internal_numpy = res
                                #return temp_array

                            dag_node = DAG("groupby.nanmean", run_and_post, False, [orig_array], (orig_array, getitem_axis, group_array, slot_indices), {})
                            dag_node.output = weakref.ref(stack_res)
                            # FIX ME Need forward_dep here from orig_array DAG node to the new one created here?
        elif dag_node.name == "getitem_array":
            # Check dag_node indices!  FIX ME

            if DAG.one_creator(dag_node) and DAG.creator(dag_node).name == "concatenate":
                concat_node = DAG.creator(dag_node)
                arrayseq = concat_node.args[0]
                assert len(concat_node.backward_deps) == len(arrayseq)
                axis = concat_node.kwargs["axis"]
                #out = concat_node.kwargs["out"]  Why isn't "out" in kwargs?
                concat_res = concat_node.output()
                dprint(3, "process concat:", id(concat_res), concat_res.shape, len(arrayseq), arrayseq[0].shape)
                binops = concat_node.backward_deps

                # If all the arrays to concat are uninstantiated from one array and the result of an array_binop.
                if all([x.name == "array_binop" for x in binops]):
                    dprint(2, "Add concat operations are array_binop")
                    # Get the array_binop operators for each array to concat.
                    # Get the internal operator (e.g., sub) for the first binop.
                    binop = binops[0].args[2]
                    binoptext = binops[0].args[3]
                    dprint(2, "binop", binop, binoptext)
                    # If all the binops have the same operation and have 2 ndarray predecessors.
                    if (all([x.args[2] == binop for x in binops]) and
                        all([len(x.backward_deps) == 2 for x in binops])
                    ):
                        dprint(2, "Add binops use the same operation", binop)
                        lhs_getitem_ops = [x.backward_deps[0] for x in binops]
                        remapped_ops = [x.backward_deps[1] for x in binops]
                        # If all the lhs arrays are uninstantiated and derived from getitem_array and all
                        # the rhs arrays are uninstantiated and derived from remapped_axis.
                        if (all([x.name == "getitem_array" for x in lhs_getitem_ops]) and
                            all([x.name == "remapped_axis" for x in remapped_ops]) and
                            all([DAG.one_creator(x) for x in lhs_getitem_ops]) and
                            all([DAG.one_creator(x) for x in remapped_ops])
                        ):
                            dprint(2, "All lhs are getitem_array and all rhs are remapped_axis")
                            # Get the axis that the first getitem operates on
                            getitem_axis = get_advindex_dim(lhs_getitem_ops[0].args[1])
                            lhs_getitem_sources = [DAG.creator(x) for x in lhs_getitem_ops]
                            reshape_ops = [DAG.creator(x) for x in remapped_ops]
                            # If all the remapped_axis calls use the same newmap.
                            # If all the getitems operate on the same base array and
                            # all the remapped_axis are sourced from reshape calls.
                            # If all the reshape have the same newshape.
                            if (all([remapped_ops[0].args[1] == x.args[1] for x in remapped_ops]) and
                                all([getitem_axis == get_advindex_dim(x.args[1]) for x in lhs_getitem_ops]) and
                                all([x is lhs_getitem_sources[0] for x in lhs_getitem_sources]) and
                                all([x.name == "reshape" for x in reshape_ops]) and
                                all([DAG.one_creator(x) for x in reshape_ops]) and
                                all([reshape_ops[0].args[1] == x.args[1] for x in reshape_ops])
                            ):
                                dprint(2, "All remapped_ops have the same newmap, all remapped derive from reshape")
                                dprint(2, "All reshapes have the same newshape")
                                rhs_getitem_ops = [DAG.creator(x) for x in reshape_ops]
                                orig_array_lhs = lhs_getitem_sources[0].output()

                                if (all([x.name == "getitem_array" for x in rhs_getitem_ops]) and
                                    all([DAG.one_creator(x) for x in rhs_getitem_ops])
                                ):
                                    dprint(2, "All reshape bases are derived from getitem_array")
                                    # A check for index of getitem_ops?
                                    getitem_bases = [DAG.creator(x) for x in rhs_getitem_ops]
                                    if all([getitem_bases[0] is x for x in getitem_bases]):
                                        dprint(2, "All reshape bases are the same")
                                        rhs = getitem_bases[0].output()
                                        dprint(2, "rhs", id(rhs))
                                        slot_indices = [x.args[1][getitem_axis] for x in lhs_getitem_ops]
                                        dprint(3, "slot_indices:", slot_indices)
                                        group_array = np.full(orig_array_lhs.shape[getitem_axis], -1, dtype=np.int)
                                        for i in range(len(slot_indices)):
                                            for vi in slot_indices[i]:
                                                group_array[vi] = i
                                        with np.printoptions(threshold=np.inf):
                                            dprint(3, "group_array:", group_array)

                                        def run_and_post(temp_array, orig_array_lhs, rhs, getitem_axis, group_array, slot_indices):
                                            print("run_and_post concat", id(temp_array))
                                            if hasattr(rhs, "internal_numpy"):
                                                rhsasarray = rhs.internal_numpy
                                            else:
                                                rhsasarray = np.moveaxis(rhs.asarray(), -1, 0)
                                            gb = orig_array_lhs.groupby(getitem_axis, group_array, num_groups=len(slot_indices))
                                            res = eval("gb" + binoptext + "rhsasarray")
                                            return res

                                        dag_node = DAG("groupby.binop", run_and_post, False, [orig_array_lhs, rhs], (orig_array_lhs, rhs, getitem_axis, group_array, slot_indices), {})
                                        dag_node.output = weakref.ref(concat_res)
                                        # FIX ME Need forward_dep here from orig_array and rhs DAG node to the new one created here?

        for backward_dep in dag_node.backward_deps:
            cls.depth_first_traverse(backward_dep, depth_first_nodes, dag_node_processed)

        dprint(2, "Adding depth first node:", dag_node.name, dag_node.args)
        depth_first_nodes.append(dag_node)

    @classmethod
    def instantiate(cls, arr):
        if isinstance(arr, ndarray):
            cls.instantiate_dag_node(arr.dag)

    def execute(self):
        soutput = self.output()
        # This can happen if we register an operator whose result is immediately not stored.
        if soutput is not None:
            exec_array_out = self.executor(soutput, *self.args, **self.kwargs)
            if not self.inplace and not isinstance(exec_array_out, ndarray):
                print("bad type", type(exec_array_out), self.name, self.args)
                breakpoint()
            assert self.inplace or isinstance(exec_array_out, ndarray)
            if not self.inplace and exec_array_out is not soutput:
                soutput.assign(exec_array_out)
        self.name = "Executed"
        self.executed = True
        for dag_node in self.backward_deps:
            dag_node.forward_deps.remove(self)
        self.args = None
        self.kwargs = None
        self.backward_deps = []

    @classmethod
    def instantiate_dag_node(cls, dag_node, do_ops=True):
        if dag_node.executed:
            assert len(dag_node.backward_deps) == 0
            return

        cls.in_evaluate += 1
        depth_first_nodes = []
        dprint(2,"Instantiate:", id(dag_node.output()))
        cls.depth_first_traverse(dag_node, depth_first_nodes, set())
        for op in depth_first_nodes:
            dprint(2, "Running DAG executor for", op.name)
            op.execute()
        # FIX ME?  Does this need to go in the loop above if we alternate between deferred_ops and something else?
        if do_ops:
            deferred_op.do_ops()
        cls.in_evaluate -= 1

    @classmethod
    def execute_all(cls):
        nodeps = list(filter(lambda x: len(x.forward_deps) == 0 and not x.executed, cls.dag_nodes))
        dprint(2, "execute_all:", len(nodeps))
        for dag_node in nodeps:
            cls.instantiate_dag_node(dag_node, do_ops=False)
        deferred_op.do_ops()


class DAGshape:
    def __init__(self, shape, dtype, inplace):
        self.shape = shape
        self.dtype = dtype
        self.inplace = inplace


def DAGapi(func):
    name = func.__qualname__
    def wrapper(*args, **kwargs):
        dprint(1, "DAGapi", name)
        fres = func(*args, **kwargs)
        if isinstance(fres, DAGshape):
            executor = eval(name + "_executor")
            nres, dag = DAG.add(fres, name, executor, args, kwargs) # need a deepcopy of args and kwargs?  maybe pickle if not simple types
            if DAG.in_evaluate > 0:
                dag.execute()
                #deferred_op.do_ops()
            return nres
        else:
            return fres
    return wrapper


# Deferred Ops stuff
class ndarray:
    def __init__(
        self,
        shape,
        *,    # the arguments below can only be passed as keyword
        gid=None,
        distribution=None,
        local_border=0,
        dtype=None,
        flex_dist=True,
        readonly=False,
        dag=None,
        maskarray=None,
        **kwargs
    ):
        t0 = timer()
        self.bdarray = bdarray.assign_bdarray(
            self, shape, gid, distribution, local_border, flex_dist, dtype, **kwargs
        )  # extra options for distribution hints
        # self.gid = self.bdarray.gid
        self.shape = shape
        self.distribution = (
            distribution
            if ((distribution is not None) and (gid is not None))
            else self.bdarray.distribution
        )
        self.local_border = local_border
        self.getitem_cache = {}
        # TODO: move to_border, from_border out of ndarray; put into bdarray, or just compute when needed to construct Local_NDarray on remotes
        if local_border > 0:
            self.from_border, self.to_border = shardview.compute_from_border(
                shape, self.distribution, local_border
            )
        else:
            self.from_border = None
            self.to_border = None
        # self.order = order
        # #self.remote_constructed = False
        # if gid in ndarray_gids:
        #     ndarray_gids[gid][0]+=1   # increment refcount
        #     ndarray_gids[gid][2].add(self)  # add to weakref set
        # else:
        #     # initially: refcount=1, remote_constructed=False, only this array in set
        #     ndarray_gids[gid] = [1, False, weakref.WeakSet([self])]
        # self.broadcasted_dims = broadcasted_dims
        self.readonly = readonly
        self.idag = dag
        self.maskarray = maskarray
        t1 = timer()
        dprint(2, "Created ndarray", self.gid, "shape", shape, "time", (t1 - t0) * 1000)

    @property
    def dag(self):
        if self.idag is None:
            dprint(2, "idag is None in dag")
            self.idag = DAG("Unknown", None, False, [], (), {}, executed=True, output=weakref.ref(self))
        # Add this fake DAG node to DAG.dag_nodes?
        return self.idag

    def instantiate(self):
        DAG.instantiate(self)

    def assign(self, other):
        assert(isinstance(other, ndarray))
        assert(self.shape == other.shape)
        # maybe assign_bdarray?
        self.distribution = other.distribution
        self.local_border = other.local_border
        self.readonly = other.readonly
        self.getitem_cache = other.getitem_cache
        self.from_border = other.from_border
        self.to_border = other.to_border
        self.maskarray = other.maskarray
        self.bdarray = bdarray.assign_bdarray(self, self.shape, gid=other.gid, distribution=self.distribution)
        #self.bdarray = other.bdarray
        if hasattr(other, "internal_numpy"):
            self.internal_numpy = other.internal_numpy
        dprint(2,"assign complete", id(self), id(other))

    # TODO: should consider using a class rather than tuple; alternative -- use weak ref to ndarray
    def get_details(self):
        return tuple(
            [
                self.shape,
                self.distribution,
                self.local_border,
                self.from_border,
                self.to_border,
                self.dtype,
            ]
        )

    @property
    def gid(self):
        return self.bdarray.gid

    @property
    def size(self):
        return np.prod(self.shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def dtype(self):
        return np.dtype(self.bdarray.dtype)

    def transpose(self, *args):
        dprint(1, "transpose", args)

        ndims = len(self.shape)
        if len(args) == 0:
            return self.remapped_axis([ndims - i - 1 for i in range(ndims)])
        else:
            if len(args) > 1:
                axes = args
            elif len(args) == 1 and isinstance(args[0], tuple):
                axes = args[0]
            # Not sufficient...should check for duplicates (i.e., all axes used exactly once)
            assert all(index >= 0 and index < ndims for index in axes)
            return self.remapped_axis(axes)

    @property
    def T(self):
        return self.transpose()

    # @property
    # def T(self):
    #    assert(len(self.shape) == 2)
    #    divs = shardview.distribution_to_divisions(self.distribution)
    #    outdiv = np.flip(divs, axis=2)
    #    rev_base_offsets = [np.flip(x.base_offset) for x in self.distribution]
    #    return ndarray((self.shape[1], self.shape[0]), gid=self.gid, distribution=shardview.divisions_to_distribution(outdiv, base_offset=rev_base_offsets), order=("C" if self.order == "F" else "F"), broadcasted_dims=(None if self.broadcasted_dims is None else self.broadcasted_dims[::-1]))

    """
    def __del__(self):
        print("ndarray::__del__", self, id(self))
        #ndarray_gids[self.gid][0]-=1
        dprint(2, "Deleting ndarray",self.gid, self)
        #if ndarray_gids[self.gid][0] <=0:
        #    if ndarray_gids[self.gid][1]:  # check remote constructed flag
        #        deferred_op.del_remote_array(self.gid)
        #    del ndarray_gids[self.gid]
    """

    def __str__(self):
        return str(self.gid) + " " + str(self.shape) + " " + str(self.local_border)

    @classmethod
    def remapped_axis_executor(cls, temp_array, self, newmap):
        newshape, newdist = shardview.remap_axis(self.shape, self.distribution, newmap)
        return ndarray(newshape, gid=self.gid, distribution=newdist, readonly=self.readonly)

    @DAGapi
    def remapped_axis(self, newmap):
        return DAGshape(shardview.remap_axis_result_shape(self.shape, newmap), self.dtype, False)

    """
    def remapped_axis(self, newmap):
        # make sure array distribution can't change (ie, not flexible or is already constructed)
        if self.bdarray.flex_dist or not self.bdarray.remote_constructed:
            deferred_op.do_ops()
        newshape, newdist = shardview.remap_axis(self.shape, self.distribution, newmap)
        return ndarray(newshape, self.gid, newdist, readonly=self.readonly)
    """

    def asarray(self):
        dprint(1, "asarray", id(self))
        if self.shape == ():
            return self.distribution

        #deferred_op.do_ops()
        DAG.instantiate(self)

        ret = np.empty(self.shape, dtype=self.dtype)
        # dist_shape = self.distribution.shape
        # topleft = tuple([self.distribution[0][0][j] for j in range(dist_shape[2])])
        dprint(2, "asarray:", self.distribution, self.shape)
        dprint(2, "asarray:", shardview.distribution_to_divisions(self.distribution))
        # shards = ray.get([remote_states[i].get_array.remote(self.gid) for i in range(dist_shape[0])])
        # shards = ray.get([remote_states[i].get_partial_array_global.remote(self.gid,
        #                  tuple([slice(self.distribution[i][0][j], self.distribution[i][1][j] + 1) for j in range(dist_shape[2])]) ) for i in range(dist_shape[0])])
        # shards = ray.get([remote_states[i].get_view.remote(self.gid, self.distribution[i]) for i in range(num_workers)])
        shards = get_results(
            [
                remote_async_call(i, "get_view", self.gid, self.distribution[i])
                for i in range(num_workers)
            ]
        )
        #        print("shards:", shards)
        #        for i in range(dist_shape[0]):
        for i in range(num_workers):
            #            dprint(2, "for:", i, dist_shape[2])
            # gindex = tuple([slice(self.distribution[i][0][j], self.distribution[i][1][j] + 1) for j in range(dist_shape[2])])
            # rindex = slice_minus_offset(gindex, topleft)
            # dprint(3, "gindex:", gindex, rindex, shards[i].shape, shards[i].shape)
            dprint(
                3,
                "gslice:",
                shardview.to_slice(self.distribution[i]),
                "bslice:",
                shardview.to_base_slice(self.distribution[i]),
            )
            # ret[rindex] = shards[i]
            ret[shardview.to_slice(self.distribution[i])] = shards[i]
        return ret

    @classmethod
    def array_unaryop_executor(
        cls, temp_array, self, op, optext, reduction=False, imports=[], dtype=None, axis=None
    ):
        dprint(1, "array_unaryop_executor", id(self), op, optext)
        if dtype is None:
            dtype = self.dtype
        elif dtype == "float":
            dtype = np.float32 if self.dtype == np.float32 else np.float64

        if reduction:
            assert not (axis is None or (axis == 0 and self.ndim == 1))
            dsz, dist = shardview.reduce_axis(self.shape, self.distribution, axis)
            k = dsz[axis]
            red_arr = empty(
                dsz, dtype=dtype, distribution=dist, no_defer=True
            )  # should create immediately
            remote_exec_all(
                "array_unaryop",
                self.gid,
                red_arr.gid,
                self.distribution,
                dist,
                op,
                axis,
                dtype,
            )
            sl = tuple(0 if i == axis else slice(None) for i in range(red_arr.ndim))
            if k == 1:  # done, just get the slice with axis removed
                ret = red_arr[sl]
            else:
                # need global reduction
                arr = empty_like(red_arr[sl])
                code = ["", arr, " = " + optext + "( np.array([", red_arr[sl]]
                for j in range(1, k):
                    sl = tuple(
                        j if i == axis else slice(None) for i in range(red_arr.ndim)
                    )
                    code += [", ", red_arr[sl]]
                code.append("]) )")
                deferred_op.add_op(code, arr, imports=imports)
                ret = arr
            return ret
        else:
            new_ndarray = create_array_with_divisions(
                self.shape,
                self.distribution,
                local_border=self.local_border,
                dtype=dtype,
            )
            deferred_op.add_op(
                ["", new_ndarray, " = " + optext + "(", self, ")"],
                new_ndarray,
                imports=imports,
            )
            return new_ndarray

    @DAGapi
    def array_unaryop(
        self, op, optext, reduction=False, imports=[], dtype=None, axis=None
    ):
        if dtype is None:
            dtype = self.dtype
        elif dtype == "float":
            dtype = np.float32 if self.dtype == np.float32 else np.float64

        if reduction:
            if axis is None or (axis == 0 and self.ndim == 1):
                DAG.instantiate(self)
                g1 = remote_call_all(
                    "array_unaryop",
                    self.gid,
                    None,
                    self.distribution,
                    None,
                    op,
                    None,
                    dtype,
                )
                v = np.array(g1)
                uop = getattr(v, op)
                ret = uop(dtype=dtype)
            else:
                ret = DAGshape(tuple([self.shape[:axis]]+[self.shape[axis+1:]]), dtype, False)
            return ret
        else:
            return DAGshape(self.shape, dtype, False)

    """
    def array_unaryop(
        self, op, optext, reduction=False, imports=[], dtype=None, axis=None
    ):
        dprint(1, "array_unaryop", id(self), op, optext)
        if dtype is None:
            dtype = self.dtype
        elif dtype == "float":
            dtype = np.float32 if self.dtype == np.float32 else np.float64
        if reduction:
            # TODO: should see if this can be converted into a deferred op
            deferred_op.do_ops()
            if axis is None or (axis == 0 and self.ndim == 1):
                # v = [remote_async_call(i, "array_unaryop", self.gid, self.distribution[i], None, op, axis, dtype) for i in range(num_workers)]
                # g1 = get_results(v)
                g1 = remote_call_all(
                    "array_unaryop",
                    self.gid,
                    None,
                    self.distribution,
                    None,
                    op,
                    None,
                    dtype,
                )
                v = np.array(g1)
                uop = getattr(v, op)
                ret = uop(dtype=dtype)
            else:
                dsz, dist = shardview.reduce_axis(self.shape, self.distribution, axis)
                k = dsz[axis]
                red_arr = empty(
                    dsz, dtype=dtype, distribution=dist, no_defer=True
                )  # should create immediately
                remote_exec_all(
                    "array_unaryop",
                    self.gid,
                    red_arr.gid,
                    self.distribution,
                    dist,
                    op,
                    axis,
                    dtype,
                )
                sl = tuple(0 if i == axis else slice(None) for i in range(red_arr.ndim))
                if k == 1:  # done, just get the slice with axis removed
                    ret = red_arr[sl]
                else:
                    # need global reduction
                    arr = empty_like(red_arr[sl])
                    code = ["", arr, " = " + optext + "( np.array([", red_arr[sl]]
                    for j in range(1, k):
                        sl = tuple(
                            j if i == axis else slice(None) for i in range(red_arr.ndim)
                        )
                        code += [", ", red_arr[sl]]
                    code.append("]) )")
                    deferred_op.add_op(code, arr, imports=imports)
                    ret = arr
            return ret
        else:
            new_ndarray = create_array_with_divisions(
                self.shape,
                self.distribution,
                local_border=self.local_border,
                dtype=dtype,
            )
            deferred_op.add_op(
                ["", new_ndarray, " = " + optext + "(", self, ")"],
                new_ndarray,
                imports=imports,
            )
            return new_ndarray
    """

    def broadcastable_to(self, shape):
        new_dims = len(shape) - len(self.shape)
        sslice = shape[-len(self.shape) :]
        z1 = zip(sslice, self.shape)
        if any([a > 1 and b > 1 and a != b for a, b in z1]):
            return False
        return True

    def broadcast_to(self, shape):
        dprint(1, "broadcast_to:", self.shape, shape)
        new_dims = len(shape) - len(self.shape)
        dprint(4, "new_dims:", new_dims)
        sslice = shape[-len(self.shape) :]
        dprint(4, "sslice:", sslice)
        z1 = zip(sslice, self.shape)
        dprint(4, "zip check:", z1)
        if any([a > 1 and b > 1 and a != b for a, b in z1]):
            raise ValueError("Non-broadcastable.")

        bd = [
            i < new_dims or (shape[i] != 1 and self.shape[i - new_dims] == 1)
            for i in range(len(shape))
        ]
        dprint(4, "broadcasted_dims:", bd)

        # make sure array distribution can't change (ie, not flexible or is already constructed)
        if self.bdarray.flex_dist or not self.bdarray.remote_constructed:
            deferred_op.do_ops()
        return ndarray(
            shape,
            gid=self.gid,
            distribution=shardview.broadcast(self.distribution, bd, shape),
            local_border=0,
            readonly=True
        )

    @classmethod
    def broadcast(cls, a, b):
        dprint(1, "broadcast")
        new_array_shape = numpy_broadcast_shape(a, b)
        if new_array_shape is None:
            DAG.instantiate(a)
            DAG.instantiate(b)
        # Check for 0d case first.  If so then return the internal value (stored in distribution).
        if isinstance(a, ndarray) and a.shape == ():
            aview = a.distribution
        elif not isinstance(a, ndarray) or new_array_shape == a.shape:
            aview = a
        else:
            aview = a.broadcast_to(new_array_shape)

        # Check for 0d case first.  If so then return the internal value (stored in distribution).
        if isinstance(b, ndarray) and b.shape == ():
            bview = b.distribution
        elif not isinstance(b, ndarray) or new_array_shape == b.shape:
            bview = b
        else:
            bview = b.broadcast_to(new_array_shape)

        return (new_array_shape, aview, bview)

    @classmethod
    def astype_executor(cls, temp_array, self, dtype, copy=True):
        dprint(1, "astype executor:", self.dtype, type(self.dtype), dtype, type(dtype), copy)
        if dtype == self.dtype:
            assert copy
            return copy(self)

        if copy:
            new_ndarray = create_array_with_divisions(
                self.shape, self.distribution, dtype=dtype
            )
            deferred_op.add_op(["", new_ndarray, " = ", self], new_ndarray)
        else:
            raise ValueError("Non-copy version of astype not implemented.")
        return new_ndarray

    @DAGapi
    def astype(self, dtype, copy=True):
        dprint(1, "astype:", self.dtype, type(self.dtype), dtype, type(dtype), copy)
        if dtype == self.dtype and not copy:
            return self
        return DAGshape(self.shape, dtype, False)

    """
    def astype(self, dtype, copy=True):
        dprint(1, "astype:", self.dtype, type(self.dtype), dtype, type(dtype), copy)
        if dtype == self.dtype:
            if copy:
                return copy(self)
            else:
                return self

        if copy:
            new_ndarray = create_array_with_divisions(
                self.shape, self.distribution, dtype=dtype
            )
            deferred_op.add_op(["", new_ndarray, " = ", self], new_ndarray)
        else:
            raise ValueError("Non-copy version of astype not implemented.")
        return new_ndarray
    """

    @classmethod
    def array_binop_executor(
        cls, temp_array, self, rhs, op, optext, inplace=False, reverse=False, imports=[], dtype=None
    ):
        if isinstance(rhs, np.ndarray):
            rhs = fromarray(rhs)
        if inplace:
            sz, selfview, rhsview = ndarray.broadcast(self, rhs)
            assert self.shape == sz

            if not isinstance(selfview, ndarray) and not isinstance(rhsview, ndarray):
                getattr(selfview, op)(rhsview)
                return self

            deferred_op.add_op(["", self, optext, rhsview], self, imports=imports)
            dprint(4, "BINARY_OP:", optext)
            return self
        else:
            lb = max(
                self.local_border, rhs.local_border if isinstance(rhs, ndarray) else 0
            )
            new_array_shape, selfview, rhsview = ndarray.broadcast(self, rhs)

            dtype = unify_args(self.dtype, rhs, dtype)

            if isinstance(new_array_shape, tuple):
                if len(new_array_shape) > 0:
                    new_ndarray = empty(new_array_shape, local_border=lb, dtype=dtype)
                else:
                    res = getattr(selfview, op)(rhsview)
                    if isinstance(res, np.ndarray):
                        return fromarray(res)
                    elif isinstance(res, numbers.Number) and new_array_shape == ():
                        return array(res)
                    else:
                        return res
            else:  # must be scalar output
                DAG.instantiate(rhs)
                res = getattr(selfview, op)(rhsview)
                return res

            if reverse:
                deferred_op.add_op(
                    ["", new_ndarray, " = ", rhsview, optext, selfview],
                    new_ndarray,
                    imports=imports,
                )
            else:
                deferred_op.add_op(
                    ["", new_ndarray, " = ", selfview, optext, rhsview],
                    new_ndarray,
                    imports=imports,
                )
            dprint(4, "BINARY_OP:", optext)
            return new_ndarray


    @DAGapi
    def array_binop(
        self, rhs, op, optext, inplace=False, reverse=False, imports=[], dtype=None
    ):
        dprint(1, "array_binop", id(self), op, optext)
        new_shape = numpy_broadcast_shape(self, rhs)
        new_dtype = unify_args(self.dtype, rhs, dtype)
        if new_shape is None:
            new_shape = ()
            DAG.instantiate(self)
            DAG.instantiate(rhs)
            return ndarray.array_binop_executor(None, self, rhs, op, optext, inplace=inplace, reverse=reverse, imports=imports, dtype=dtype)
        else:
            if inplace:
                return DAGshape(new_shape, new_dtype, self)
            else:
                return DAGshape(new_shape, new_dtype, False)

    """
    def array_binop(
        self, rhs, op, optext, inplace=False, reverse=False, imports=[], dtype=None
    ):
        t0 = timer()
        if isinstance(rhs, np.ndarray):
            rhs = fromarray(rhs)
        if inplace:
            sz, selfview, rhsview = ndarray.broadcast(self, rhs)
            assert self.shape== sz

            if not isinstance(selfview, ndarray) and not isinstance(rhsview, ndarray):
                getattr(selfview, op)(rhsview)
                return self

            deferred_op.add_op(["", self, optext, rhsview], self, imports=imports)
            t1 = timer()
            dprint(4, "BINARY_OP:", optext, "time", (t1 - t0) * 1000)
            return self
        else:
            lb = max(
                self.local_border, rhs.local_border if isinstance(rhs, ndarray) else 0
            )
            new_array_shape, selfview, rhsview = ndarray.broadcast(self, rhs)

            dtype = unify_args(self.dtype, rhs, dtype)
            "
            if hasattr(rhs, "dtype"):
                rhs_dtype = rhs.dtype
            else:
                try:
                    rhs_dtype = np.dtype(rhs)
                except Exception:
                    rhs_dtype = None
            if dtype is None:
                dtype = np.result_type(self.dtype, rhs_dtype)
            elif dtype == "float":
                dtype = (
                    np.float32
                    if rhs_dtype == np.float32 and self.dtype == np.float32
                    else np.float64
                )
            "

            if self.advindex is not None or (isinstance(rhs, ndarray) and rhs.advindex is not None):
                res = ndarray(
                    new_array_shape,
                    dtype=dtype,
                    advindex=("array_binop", (self, rhs, optext, inplace, reverse, imports, dtype))
                )
                return res

            if isinstance(new_array_shape, tuple):
                if len(new_array_shape) > 0:
                    new_ndarray = empty(new_array_shape, local_border=lb, dtype=dtype)
                else:
                    res = getattr(selfview, op)(rhsview)
                    if isinstance(res, np.ndarray):
                        return fromarray(res)
                    elif isinstance(res, numbers.Number) and new_array_shape == ():
                        return array(res)
                    else:
                        return res
            else:  # must be scalar output
                res = getattr(selfview, op)(rhsview)
                return res

            if reverse:
                deferred_op.add_op(
                    ["", new_ndarray, " = ", rhsview, optext, selfview],
                    new_ndarray,
                    imports=imports,
                )
            else:
                deferred_op.add_op(
                    ["", new_ndarray, " = ", selfview, optext, rhsview],
                    new_ndarray,
                    imports=imports,
                )
            t1 = timer()
            dprint(4, "BINARY_OP:", optext, "time", (t1 - t0) * 1000)
            return new_ndarray
    """

    @classmethod
    def setitem_executor(cls, temp_array, self, key, value):
        dprint(1, "ndarray::__setitem__:", key, type(key), value, type(value))
        if self.readonly:
            raise ValueError("assignment destination is read-only")
        is_mask = False
        if isinstance(key, ndarray) and key.dtype==bool and key.broadcastable_to(self.shape):
            is_mask = True
        elif not isinstance(key, tuple):
            key = (key,)

        if is_mask or any([isinstance(i, slice) for i in key]) or len(key) < len(self.shape):
            view = self[key]
            if isinstance(value, (int, bool, float, complex)):
                deferred_op.add_op(["", view, " = ", value, ""], view)
                return
            elif value.broadcastable_to(view.shape):
                if (value.shape!=view.shape):
                    value = value.broadcast_to(view.shape)
                # avoid adding code in case of inplace operations
                if not (
                    view.gid == value.gid
                    and shardview.dist_is_eq(view.distribution, value.distribution)
                ):
                    deferred_op.add_op(["", view, " = ", value, ""], view)
                return
            else:
                # TODO:  Should try to broadcast value to view.shape before giving up;  Done?
                print("Mismatched shapes", view.shape, value.shape)
                assert 0

        if all([isinstance(i, int) for i in key]) and len(key) == len(self.shape):
            print("Setting individual element is not handled yet!")
            assert 0

        """
        if isinstance(key[0], slice):
            if key.start is None and key.stop is None:   # a[:] = b
                if isinstance(value, ndarray):
                    if self.shape == value.shape:
                        deferred_op.add_op(["", self, " = ", value, ""], self)
                        #[remote_states[i].setitem1.remote(self.gid, value.gid) for i in range(num_workers)]
                        return
                elif isinstance(value, (int, float, complex)):
                    deferred_op.add_op(["", self, " = ", value, ""], self)
                    return
            else:   # a[s:e] = b
                view = self[key]
                if view.shape == value.shape:
                    deferred_op.add_op(["", view, " = ", value, ""], view)
                    return
        """
        # Need to handle all possible remaining cases.
        print("Don't know how to set index", key, " of dist array of shape", self.shape)
        assert 0

    @DAGapi
    def setitem(self, key, value):
        return DAGshape(self.shape, self.dtype, self)

    def __setitem__(self, key, value):
        return self.setitem(key, value)

    def __getitem__(self, index):
        if isinstance(index, ndarray):
            return self.getitem_real(index)
        indhash = pickle.dumps(index)
        dprint(3, "__getitem__", id(self), index, type(index), self.shape, indhash in self.getitem_cache)
        if indhash not in self.getitem_cache:
            self.getitem_cache[indhash] = self.getitem_real(index)
        return self.getitem_cache[indhash]

    @classmethod
    def getitem_array_executor(cls, temp_array, self, index):
        if isinstance(index, ndarray) and index.dtype==bool and index.broadcastable_to(self.shape):
            if index.shape != self.shape:
                index = index.broadcast_to(self.shape)
            return ndarray( self.shape, gid=self.gid, distribution=self.distribution, local_border=0, 
                    readonly=self.readonly, dtype=self.dtype, maskarray=index)

        index_has_slice = any([isinstance(i, slice) for i in index])
        index_has_array = any([isinstance(i, np.ndarray) for i in index])

        if index_has_array:
            # This is the advanced indexing case that always creates a copy.
            dim_sizes = dim_sizes_from_index(index, self.shape)
            assert 0 # Need to implement the slow way here in the case it doesn't get optimized away.
        # If any of the indices are slices or the number of indices is less than the number of array dimensions.
        elif index_has_slice or len(index) < len(self.shape):
            # check for out-of-bounds
            for i in range(len(index)):
                if isinstance(index[i], int) and index[i] >= self.shape[i]:
                    raise IndexError(
                        "index "
                        + str(index[i])
                        + " is out of bounds for axis "
                        + str(i)
                        + " with shape "
                        + str(self.shape[i])
                    )

            # make sure array distribution can't change (ie, not flexible or is already constructed)
            if self.bdarray.flex_dist or not self.bdarray.remote_constructed:
                deferred_op.do_ops()
            num_dim = len(self.shape)
            cindex = canonical_index(index, self.shape)
            dim_shapes = tuple([max(0, x.stop - x.start) for x in cindex])
            dprint(2, "getitem slice:", cindex, dim_shapes)

            sdistribution = shardview.slice_distribution(cindex, self.distribution)
            # reduce dimensionality as needed
            axismap = [
                i
                for i in range(len(dim_shapes))
                if i >= len(index) or isinstance(index[i], slice)
            ]
            if len(axismap) < len(dim_shapes):
                dim_shapes, sdistribution = shardview.remap_axis(
                    dim_shapes, sdistribution, axismap
                )
            dprint(2, "getitem slice:", dim_shapes, sdistribution)
            #            deferred_op.add_op(["", self, " = ", value, ""])
            # return ndarray(self.gid, tuple(dim_shapes), np.asarray(sdistribution))
            # Note: slices have local border set to 0 -- otherwise may corrupt data in the array
            return ndarray(
                dim_shapes,
                gid=self.gid,
                distribution=sdistribution,
                local_border=0,
                readonly=self.readonly,
                dtype=self.dtype
            )

        print("Don't know how to get index", index, type(index), " of dist array of shape", self.shape)
        assert 0  # Handle other types

    @DAGapi
    def getitem_array(self, index):
        if isinstance(index, ndarray) and index.dtype==bool and index.broadcastable_to(self.shape):
            return DAGshape(self.shape, self.dtype, False)

        index_has_slice = any([isinstance(i, slice) for i in index])
        index_has_array = any([isinstance(i, np.ndarray) for i in index])

        if index_has_array:
            # This is the advanced indexing case that always creates a copy.
            dim_sizes = dim_sizes_from_index(index, self.shape)
            return DAGshape(dim_sizes, self.dtype, False)
        # If any of the indices are slices or the number of indices is less than the number of array dimensions.
        elif index_has_slice or len(index) < len(self.shape):
            # check for out-of-bounds
            for i in range(len(index)):
                if isinstance(index[i], int) and index[i] >= self.shape[i]:
                    raise IndexError(
                        "index "
                        + str(index[i])
                        + " is out of bounds for axis "
                        + str(i)
                        + " with shape "
                        + str(self.shape[i])
                    )

            cindex = canonical_index(index, self.shape)
            dim_shapes = tuple([max(0, x.stop - x.start) for x in cindex])
            axismap = [
                i
                for i in range(len(dim_shapes))
                if i >= len(index) or isinstance(index[i], slice)
            ]
            if len(axismap) < len(dim_shapes):
                dim_shapes = shardview.remap_axis_result_shape(dim_shapes, axismap)
            dprint(2, "__getitem__array slice:", cindex, dim_shapes)
            return DAGshape(dim_shapes, self.dtype, False)

        print("Don't know how to get index", index, type(index), " of dist array of shape", self.shape)
        assert 0  # Handle other types

    def getitem_real(self, index):
        dprint(2, "ndarray::__getitem__real:", index, type(index), self.shape, len(self.shape))
        # index is a mask ndarray -- boolean type with same shape as array (or broadcastable to that shape)
        if isinstance(index, ndarray) and index.dtype==bool and index.broadcastable_to(self.shape):
            return self.getitem_array(index)

        if not isinstance(index, tuple):
            index = (index,)

        if index[-1] is Ellipsis:
            index = index[:-1] + tuple([slice(None,None) for _ in range(self.ndim - (len(index)-1))])

        if any([x is None for x in index]):
            newdims = [i for i in range(len(index)) if index[i] is None]
            expanded_array = expand_dims(self, newdims)
            return expanded_array

        # If all the indices are integers and the number of indices equals the number of array dimensions.
        if all([isinstance(i, int) for i in index]) and len(index) == len(self.shape):
            self.instantiate()

            owner = shardview.find_index(
                self.distribution, index
            )  # find_index(self.distribution, index)
            dprint(2, "owner:", owner)
            # ret = ray.get(remote_states[owner].getitem_global.remote(self.gid, index, self.distribution[owner]))
            ret = remote_call(
                owner, "getitem_global", self.gid, index, self.distribution[owner]
            )
            return ret

        return self.getitem_array(index)

        """
        index_has_slice = any([isinstance(i, slice) for i in index])
        index_has_array = any([isinstance(i, np.ndarray) for i in index])

        if index_has_array:
            # This is the advanced indexing case that always creates a copy.
            dim_sizes = dim_sizes_from_index(index, self.shape)
            dprint(2, "created advindex ndarray")
            res = ndarray(
                dim_sizes,
                dtype=self.dtype,
                advindex=("__getitem__", (self, index))
            )
            return res
        # If any of the indices are slices or the number of indices is less than the number of array dimensions.
        elif index_has_slice or len(index) < len(self.shape):
            # check for out-of-bounds
            for i in range(len(index)):
                if isinstance(index[i], int) and index[i] >= self.shape[i]:
                    raise IndexError(
                        "index "
                        + str(index[i])
                        + " is out of bounds for axis "
                        + str(i)
                        + " with shape "
                        + str(self.shape[i])
                    )

            # make sure array distribution can't change (ie, not flexible or is already constructed)
            if self.bdarray.flex_dist or not self.bdarray.remote_constructed:
                deferred_op.do_ops()
            num_dim = len(self.shape)
            cindex = canonical_index(index, self.shape)
            dim_shapes = tuple([max(0, x.stop - x.start) for x in cindex])
            dprint(2, "getitem slice:", cindex, dim_shapes)

            # sdistribution = [
            #                 [[max(self.distribution[i][0][j], cindex[j].start)  for j in range(num_dim)],
            #                  [max(max(self.distribution[i][0][j]-1, cindex[j].start), min(self.distribution[i][1][j], cindex[j].stop-1)) for j in range(num_dim)]]
            #                  for i in range(len(self.distribution))]
            sdistribution = shardview.slice_distribution(cindex, self.distribution)
            # reduce dimensionality as needed
            axismap = [
                i
                for i in range(len(dim_shapes))
                if i >= len(index) or isinstance(index[i], slice)
            ]
            if len(axismap) < len(dim_shapes):
                dim_shapes, sdistribution = shardview.remap_axis(
                    dim_shapes, sdistribution, axismap
                )
            dprint(2, "getitem slice:", dim_shapes, sdistribution)
            #            deferred_op.add_op(["", self, " = ", value, ""])
            # return ndarray(self.gid, tuple(dim_shapes), np.asarray(sdistribution))
            # Note: slices have local border set to 0 -- otherwise may corrupt data in the array
            return ndarray(
                dim_shapes,
                gid=self.gid,
                distribution=sdistribution,
                local_border=0,
                readonly=self.readonly,
                dtype=self.dtype
            )

        print("Don't know how to get index", index, type(index), " of dist array of shape", self.shape)
        assert 0  # Handle other types
        """

    def get_remote_ranges(self, required_division):
        return get_remote_ranges(self.distribution, required_division)

    def __matmul__(self, rhs):
        return matmul(self, rhs)

    def reshape(self, newshape):
        return reshape(self, newshape)

    def reshape_copy(self, newshape):
        return reshape_copy(self, newshape)

    def mean(self, axis=None, dtype=None):
        dprint(1, "mean", id(self))
        n = np.prod(self.shape) if axis is None else self.shape[axis]
        s = 1 / n
        rv = s * self.sum(axis=axis, dtype=dtype)
        if dtype is not None:
            if isinstance(rv, ndarray):
                rv = rv.astype(dtype)
            else:
                rv = np.mean([rv], dtype=dtype)
        return rv

    @classmethod
    def nanmean_executor(cls, temp_array, self, axis=None, dtype=None):
        dprint(1, "nanmean_executor", id(self))
        assert axis is not None
        assert 0 # Actually do the computation if it isn't optimized away.

    @DAGapi
    def nanmean(self, axis=None, dtype=None):
        dprint(1, "nanmean", id(self))
        if axis is None:
            DAG.instantiate(self)  # here or in sreduce?
            #return 0
            sres = sreduce(lambda x: (x,1) if x != np.nan else (0,0), lambda x,y: (x[0]+y[0], x[1]+y[1]), (0,0), self)
            print("nanmean sres:", sres)
            return sres[0] / sres[1]
        else:
            res_size = tuple(self.shape[:axis] + self.shape[axis+1:])
            return DAGshape(res_size, dtype, False)

    """
    def nanmean(self, axis=None, dtype=None):
        dprint(1, "nanmean", id(self))
        if axis is None:
            #return 0
            sres = sreduce(lambda x: (x,1) if x != np.nan else (0,0), lambda x,y: (x[0]+y[0], x[1]+y[1]), (0,0), self)
            print("nanmean sres:", sres)
            return sres[0] / sres[1]
        else:
            res_size = tuple(self.shape[:axis] + self.shape[axis+1:])
            res = ndarray(
                res_size,
                dtype=self.dtype,
                advindex=("nanmean", (self, axis))
            )
            return res
    """

    # def __len__(self):
    #    return self.shape[0]

    def __array_function__(self, func, types, args, kwargs):
        dprint(
            1,
            "__array_function__",
            func,
            types,
            len(args),
            args,
            kwargs,
            func in HANDLED_FUNCTIONS,
        )
        for arg in args:
            dprint(4, "arg:", arg, type(arg))
        new_args = []
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        hf = HANDLED_FUNCTIONS[func]
        if hf[1]:
            new_args.append(self)
        new_args.extend(args)
        """
        for arg in args:
            if isinstance(arg, np.ndarray):
                new_args.append(fromarray(arg))
            elif isinstance(arg, ndarray):
                new_args.append(arg)
            else:
                return NotImplemented
        """
        return hf[0](*new_args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        dprint(1, "__array_ufunc__", ufunc, type(ufunc), method, len(inputs), kwargs)
        real_args = []
        if method == "__call__":
            for arg in inputs:
                if isinstance(arg, np.ndarray):
                    real_args.append(fromarray(arg))
                elif isinstance(arg, ndarray):
                    real_args.append(arg)
                elif isinstance(arg, numbers.Number):
                    real_args.append(arg)
                else:
                    print(type(arg))
                    return NotImplemented
            isreversed = not isinstance(inputs[0], ndarray)
            if ufunc.__name__ in ufunc_map:
                mapname = ufunc_map[ufunc.__name__]
            else:
                mapname = ufunc.__name__
            if isreversed:
                mapname = "r" + mapname
                real_args = real_args[::-1]

            attrres = getattr(real_args[0], "__" + mapname + "__", None)
            if attrres is None:
                attrres = getattr(real_args[0], mapname, None)
            if attrres is None:
                attrres = getattr(real_args[0], ufunc.__name__, None)
            assert attrres is not None
            dprint(2, "attrres:", attrres, type(attrres), real_args)
            return attrres(*real_args[1:], **kwargs)
        elif method == "reduce":
            return sreduce(lambda x: x, ufunc, *inputs)
        else:
            return NotImplemented

    def groupby(self, dim, value_to_group, num_groups=None):
        if num_groups is None:
            num_groups = value_to_group.max()
        return RambaGroupby(self, dim, value_to_group, num_groups=num_groups)


# We only have to put functions here where the ufunc name is different from the
# Python operation name.
ufunc_map = {
    "multiply": "mul",
    "subtract": "sub",
    "divide": "div",
    "true_divide": "truediv",
    "floor_divide": "floordiv",
}


def dot(a, b, out=None):
    dprint(1, "dot")
    ashape = a.shape
    bshape = b.shape
    if len(ashape) <= 2 and len(bshape) <= 2:
        return matmul(a, b, out=out)
    else:
        print("dot for matrices higher than 2 dimensions not currently supported.")
        assert 0



def matmul(a, b, reduction=False, out=None):
    #DAG.in_evaluate += 1
    res = matmul_internal(a, b, reduction=reduction, out=out)
    #DAG.in_evaluate -= 1
    return res


@DAGapi
def matmul(a, b, reduction=False, out=None):
    ashape = a.shape
    bshape = b.shape
    dtype = np.result_type(a.dtype, b.dtype)
    if len(ashape) == 1 and len(bshape) == 1:
        DAG.instantiate(a)
        DAG.instantiate(b)
        assert ashape[0] == bshape[0]
        # shortcut
        return (a * b).sum()

    if len(ashape) == 1:
        a = reshape(a, (1, ashape[0]))
        ashape = (1, ashape[0])

    if len(ashape) > 2 or len(bshape) > 2:
        print("matmul for matrices higher than 2 dimensions not currently supported.")
        assert 0

    assert ashape[1] == bshape[0]

    if len(bshape) == 1:
        out_shape = (ashape[0],)
    else:
        out_shape = (ashape[0], bshape[1])

    if out is not None:
        assert out.shape == out_shape
        return DAGshape(out_shape, dtype, out)
    else:
        return DAGshape(out_shape, dtype, False)


def matmul_executor(temp_array, a, b, reduction=False, out=None):
    return matmul_internal(a, b, reduction=reduction, out=out)

def matmul_internal(a, b, reduction=False, out=None):
    #DAG.instantiate(a)
    #DAG.instantiate(b)
    #DAG.execute_all()
    dprint(1, "matmul", a.shape, b.shape)
    pre_matmul_start_time = timer()
    ashape = a.shape
    bshape = b.shape

    # Handle the 1D x 1D case.
    if len(ashape) == 1 and len(bshape) == 1:
        assert ashape[0] == bshape[0]
        # shortcut
        return (a * b).sum()

    aextend = False
    bextend = False
    if len(ashape) == 1:
        aextend = True
        a = reshape(a, (1, ashape[0]))
        ashape = a.shape

    if len(bshape) == 1:
        bextend = True

    if len(ashape) > 2 or len(bshape) > 2:
        print("matmul for matrices higher than 2 dimensions not currently supported.")
        assert 0

    assert ashape[1] == bshape[0]

    if bextend:
        out_shape = (ashape[0],)
    else:
        out_shape = (ashape[0], bshape[1])

    if out is not None:
        assert out.shape == out_shape
        out_ndarray = out
    else:
        out_ndarray = empty(out_shape, dtype=np.result_type(a.dtype, b.dtype), no_defer=True)
    out_ndarray.instantiate()
    #DAG.execute_all()

    pre_matmul_end_time = timer()
    tprint(
        2,
        "pre_matmul_total_time:",
        pre_matmul_end_time - pre_matmul_start_time,
        ashape,
        bshape,
    )

    a_send_recv = uuid.uuid4()
    b_send_recv = uuid.uuid4()

    # If the output is not distributed then do the compute localized and then reduce.
    if not reduction and do_not_distribute(out_shape):
        dprint(2, "matmul output is not distributed and is not recursive")
        if ntiming >= 1:
            sync_start_time = timer()
            sync()
            sync_end_time = timer()
            tprint(
                2,
                "matmul_sync_total_time:",
                sync_end_time - sync_start_time,
                ashape,
                bshape,
            )
        #else:
            #DAG.execute_all()
            #if isinstance(a, ndarray):
            #    DAG.instantiate(a)
            #if isinstance(b, ndarray):
            #    DAG.instantiate(b)
            #deferred_op.do_ops()

        matmul_start_time = timer()

        adivs = shardview.distribution_to_divisions(a.distribution)
        bdivs = shardview.distribution_to_divisions(b.distribution)
        dprint(4, "matmul adivs:", adivs, "\n", adivs[:, :, 0], "\n", adivs[:, :, 1])
        dprint(3, "matmul bdivs:", bdivs, "\n", bdivs[:, :, 0])
        if do_not_distribute(bshape):
            dprint(2, "matmul b matrix is not distributed")
            blocal = b.asarray()
            dprint(2, "blocal", blocal.shape, blocal)
            worker_info = []
            workers = []
            matmul_workers = []

            reduction_slicing_start_time = timer()
            reduction_slicing_end_time = timer()
            launch_start_time = timer()
            matmul_workers = remote_async_call_all(
                "matmul",
                out_shape,
                out_ndarray.gid,
                out_ndarray.distribution,
                a.gid,
                a.shape,
                a.distribution,
                blocal,
                0,
                0,
                bextend,
                a_send_recv,
                b_send_recv,
            )
            launch_end_time = timer()
            launch_total = launch_end_time - launch_start_time

            worker_timings = get_results(matmul_workers)
            post_get_end_time = timer()
            if not fast_reduction:
                reduce_start_time = timer()
                redres = functools.reduce(operator.add, [x[11] for x in worker_timings])
                reduce_end_time = timer()
                out_ndarray[:] = fromarray(redres)
                fromarray_end_time = timer()

                if ntiming >= 1:
                    sync()
                sync_end_time = timer()
                tprint(
                    2,
                    "driver_reduction_time:",
                    sync_end_time - reduce_start_time,
                    reduce_end_time - reduce_start_time,
                    fromarray_end_time - reduce_end_time,
                    sync_end_time - fromarray_end_time,
                )

            matmul_end_time = timer()
            tprint(
                2,
                "matmul_total_time:",
                matmul_end_time - matmul_start_time,
                ashape,
                bshape,
                reduction_slicing_end_time - reduction_slicing_start_time,
                launch_total,
                post_get_end_time - reduction_slicing_end_time,
                matmul_end_time - post_get_end_time,
            )
            for worker_data in worker_timings:
                (
                    worker_num,
                    worker_total,
                    compute_comm,
                    comm_time,
                    len_arange,
                    len_brange,
                    exec_time,
                    a_send_stats,
                    a_recv_stats,
                    b_send_stats,
                    b_recv_stats,
                    _,
                ) = worker_data
                tprint(
                    3,
                    "reduction matmul_worker:",
                    worker_num,
                    worker_total,
                    compute_comm,
                    comm_time,
                    exec_time,
                    len_arange,
                    len_brange,
                    a_send_stats,
                    a_recv_stats,
                    b_send_stats,
                    b_recv_stats,
                )

            add_time("matmul_b_c_not_dist", matmul_end_time - pre_matmul_start_time)
            add_sub_time(
                "matmul_b_c_not_dist",
                "pre",
                pre_matmul_end_time - pre_matmul_start_time,
            )
            add_sub_time("matmul_b_c_not_dist", "launch", launch_total)
            add_sub_time(
                "matmul_b_c_not_dist",
                "compute_comm",
                max([x[2] for x in worker_timings]),
            )
            add_sub_time(
                "matmul_b_c_not_dist", "comm", max([x[3] for x in worker_timings])
            )
            add_sub_time(
                "matmul_b_c_not_dist", "exec", max([x[6] for x in worker_timings])
            )

            if aextend:
                return reshape(out_ndarray, (out_shape[1],))
            else:
                return out_ndarray
        elif (
            np.array_equal(adivs[:, :, 1], bdivs[:, :, 0])
            and np.min(adivs[:, 0, 0]) == np.max(adivs[:, 0, 0])
            and np.min(adivs[:, 1, 0]) == np.max(adivs[:, 1, 0])
        ):
            dprint(
                2,
                "matmul b matrix is distributed and has same inner distribution as the a matrix outer distribution",
            )
            adivs_shape = adivs.shape
            assert adivs_shape[0] == num_workers

            worker_info = []
            workers = []
            matmul_workers = []

            reduction_slicing_start_time = timer()
            reduction_slicing_end_time = timer()
            launch_start_time = timer()
            matmul_workers = remote_async_call_all(
                "matmul",
                out_shape,
                out_ndarray.gid,
                out_ndarray.distribution,
                a.gid,
                a.shape,
                a.distribution,
                b.gid,
                b.shape,
                b.distribution,
                bextend,
                a_send_recv,
                b_send_recv,
            )
            launch_end_time = timer()
            launch_total = launch_end_time - launch_start_time
            worker_timings = get_results(matmul_workers)
            post_get_end_time = timer()
            if not fast_reduction:
                reduce_start_time = timer()
                redres = functools.reduce(operator.add, [x[11] for x in worker_timings])
                reduce_end_time = timer()
                out_ndarray[:] = fromarray(redres)
                fromarray_end_time = timer()

                if ntiming >= 1:
                    sync()
                sync_end_time = timer()
                tprint(
                    2,
                    "driver_reduction_time:",
                    sync_end_time - reduce_start_time,
                    reduce_end_time - reduce_start_time,
                    fromarray_end_time - reduce_end_time,
                    sync_end_time - fromarray_end_time,
                )

            matmul_end_time = timer()
            tprint(
                2,
                "matmul_total_time:",
                matmul_end_time - matmul_start_time,
                ashape,
                bshape,
                "slicing_time",
                0,
                "launch_time",
                launch_total,
                "get_results_time",
                post_get_end_time - launch_end_time,
            )
            for worker_data in worker_timings:
                (
                    worker_num,
                    worker_total,
                    compute_comm,
                    comm_time,
                    len_arange,
                    len_brange,
                    exec_time,
                    a_send_stats,
                    a_recv_stats,
                    b_send_stats,
                    b_recv_stats,
                    _,
                ) = worker_data
                tprint(
                    3,
                    "reduction matmul_worker:",
                    worker_num,
                    worker_total,
                    compute_comm,
                    comm_time,
                    exec_time,
                    len_arange,
                    len_brange,
                    a_send_stats,
                    a_recv_stats,
                    b_send_stats,
                    b_recv_stats,
                )

            add_time(
                "matmul_c_not_dist_a_b_dist_match",
                matmul_end_time - pre_matmul_start_time,
            )
            add_sub_time(
                "matmul_c_not_dist_a_b_dist_match",
                "pre",
                pre_matmul_end_time - pre_matmul_start_time,
            )
            add_sub_time("matmul_c_not_dist_a_b_dist_match", "launch", launch_total)
            add_sub_time(
                "matmul_c_not_dist_a_b_dist_match",
                "reduction",
                max([x[3] for x in worker_timings]),
            )
            add_sub_time(
                "matmul_c_not_dist_a_b_dist_match",
                "exec",
                max([x[6] for x in worker_timings]),
            )

            if aextend:
                return reshape(out_ndarray, (out_shape[1],))
            else:
                return out_ndarray
        else:
            # print("not simple case:", out_shape, adivs, adivs[:,:,1], bdivs, bdivs[:,:,0], np.array_equal(adivs[:,:,1], bdivs[:,:,0]))
            dprint(
                2,
                "matmul b matrix is distributed but a is not distributed across 2nd dimension nor b across inner dimension",
            )
            adivs_shape = adivs.shape
            divisions = np.empty((adivs_shape[0], 2, len(out_shape)), dtype=np.int64)
            starts = np.zeros(len(out_shape), dtype=np.int64)
            ends = np.array(list(out_shape), dtype=np.int64)
            # the ends are inclusive, not one past the last index
            ends -= 1
            assert adivs_shape[0] == num_workers

            worker_info = []
            workers = []
            partials = []

            reduction_slicing_start_time = timer()
            for i in range(adivs_shape[0]):
                shardview.make_uni_dist(divisions, i, starts, ends)
                partial_matmul_res = zeros(
                    out_shape,
                    distribution=shardview.divisions_to_distribution(divisions),
                )
                aslice_struct = (
                    slice(adivs[i, 0, 0], adivs[i, 1, 0] + 1),
                    slice(adivs[i, 0, 1], adivs[i, 1, 1] + 1),
                )
                aslice = a[aslice_struct]
                if bextend:
                    bslice_struct = (slice(adivs[i, 0, 1], adivs[i, 1, 1] + 1),)
                    bslice = b[bslice_struct]
                    cslice_struct = (slice(adivs[i, 0, 0], adivs[i, 1, 0] + 1),)
                    cslice = partial_matmul_res[cslice_struct]
                else:
                    bslice_struct = (
                        slice(adivs[i, 0, 1], adivs[i, 1, 1] + 1),
                        slice(0, out_shape[1]),
                    )
                    bslice = b[bslice_struct]
                    cslice_struct = (
                        slice(adivs[i, 0, 0], adivs[i, 1, 0] + 1),
                        slice(0, out_shape[1]),
                    )
                    cslice = partial_matmul_res[cslice_struct]
                dprint(
                    2,
                    "matmul part:",
                    i,
                    partial_matmul_res,
                    aslice_struct,
                    bslice_struct,
                    cslice_struct,
                )

                partials.append(partial_matmul_res)
                worker_info.append((aslice, bslice, cslice, partial_matmul_res))
            reduction_slicing_end_time = timer()

            # Just so that the partial results zeros are there.
            #DAG.execute_all()
            #deferred_op.do_ops()

            matmul_workers = []

            launch_start_time = timer()
            for wi in worker_info:
                aslice, bslice, cslice, partial_matmul_res = wi
                matmul_workers.extend(
                    matmul_internal(aslice, bslice, reduction=True, out=cslice)
                )

            launch_end_time = timer()

            worker_timings = get_results(matmul_workers)
            post_get_end_time = timer()
            if ndebug > 2:
                pasarray = [x.asarray() for x in partials]
                for p in pasarray:
                    print("partial result:", p)
            out_ndarray[:] = functools.reduce(operator.add, partials)

            if ntiming >= 1:
                sync()

            matmul_end_time = timer()
            tprint(
                2,
                "matmul_total_time:",
                matmul_end_time - matmul_start_time,
                ashape,
                bshape,
                reduction_slicing_end_time - reduction_slicing_start_time,
                launch_end_time - launch_start_time,
                post_get_end_time - launch_end_time,
                matmul_end_time - post_get_end_time,
            )
            for worker_data in worker_timings:
                (
                    worker_num,
                    worker_total,
                    compute_comm,
                    comm_time,
                    len_arange,
                    len_brange,
                    exec_time,
                    a_send_stats,
                    a_recv_stats,
                    b_send_stats,
                    b_recv_stats,
                    _,
                ) = worker_data
                tprint(
                    3,
                    "reduction matmul_worker:",
                    worker_num,
                    worker_total,
                    compute_comm,
                    comm_time,
                    exec_time,
                    len_arange,
                    len_brange,
                    a_send_stats,
                    a_recv_stats,
                    b_send_stats,
                    b_recv_stats,
                )

            add_time(
                "matmul_c_not_dist_a_b_dist_non_match",
                matmul_end_time - pre_matmul_start_time,
            )

            if aextend:
                return reshape(out_ndarray, (out_shape[1],))
            else:
                return out_ndarray

    if not reduction:
        if ntiming >= 1:
            sync_start_time = timer()
            sync()
            sync_end_time = timer()
            tprint(
                2,
                "matmul_sync_total_time:",
                sync_end_time - sync_start_time,
                ashape,
                bshape,
            )
        #else:
            #DAG.execute_all()
            #deferred_op.do_ops()

    matmul_start_time = timer()
    dprint(
        4,
        "matmul a:",
        ashape,
        a.distribution,
        shardview.distribution_to_divisions(a.distribution),
    )
    dprint(4, "matmul b:", bshape, shardview.distribution_to_divisions(b.distribution))
    out_distribution = shardview.distribution_to_divisions(out_ndarray.distribution)
    dprint(4, "matmul out:", out_shape, out_distribution)

    # a_send_recv = uuid.uuid4()
    # b_send_recv = uuid.uuid4()
    end_compute_comm_set = timer()

    # matmul_workers = remote_call_all("matmul", out_ndarray.gid,
    #                                a.gid, a.shape, a.distribution,
    #                                b.gid, b.shape, b.distribution, bextend,
    #                                a_send_recv, b_send_recv)

    launch_start_time = timer()
    matmul_workers = remote_async_call_all(
        "matmul",
        out_ndarray.gid,
        out_ndarray.shape,
        out_ndarray.distribution,
        a.gid,
        a.shape,
        a.distribution,
        b.gid,
        b.shape,
        b.distribution,
        bextend,
        a_send_recv,
        b_send_recv,
    )
    launch_end_time = timer()
    launch_total = launch_end_time - launch_start_time

    if reduction:
        return matmul_workers
    else:
        # worker_timings = ray.get(matmul_workers)
        worker_timings = get_results(matmul_workers)

        matmul_end_time = timer()
        tprint(
            2, "matmul_total_time:", matmul_end_time - matmul_start_time, ashape, bshape
        )
        for worker_data in worker_timings:
            (
                worker_num,
                worker_total,
                compute_comm,
                comm_time,
                len_arange,
                len_brange,
                exec_time,
                a_send_stats,
                a_recv_stats,
                b_send_stats,
                b_recv_stats,
                _,
            ) = worker_data
            tprint(
                3,
                "matmul_worker:",
                worker_num,
                worker_total,
                compute_comm,
                comm_time,
                exec_time,
                len_arange,
                len_brange,
                a_send_stats,
                a_recv_stats,
                b_send_stats,
                b_recv_stats,
            )

        add_time("matmul_general", matmul_end_time - pre_matmul_start_time)
        add_sub_time(
            "matmul_general", "pre", pre_matmul_end_time - pre_matmul_start_time
        )
        add_sub_time("matmul_general", "launch", launch_total)
        add_sub_time(
            "matmul_general", "compute_comm", max([x[2] for x in worker_timings])
        )
        add_sub_time("matmul_general", "comm", max([x[3] for x in worker_timings]))
        add_sub_time("matmul_general", "exec", max([x[6] for x in worker_timings]))

        if aextend:
            return reshape(out_ndarray, (out_shape[1],))
        else:
            return out_ndarray


def matmul_summary():
    print(get_timing_str(details=True))


if ntiming > 0:
    print("registering matmul_summary")
    atexit.register(matmul_summary)


def canonical_dim(dim, dim_size, end=False):
    if dim is None:
        return dim_size if end else 0
    if dim < 0:
        return max(0, dim + dim_size)
    elif dim < dim_size:
        return dim
    else:
        return dim_size


def dim_sizes_from_index(index, size):
    newindex = []
    if not isinstance(index, tuple):
        index = (index,)
    assert len(index) <= len(size)
    for i in range(len(size)):
        if i >= len(index):
            newindex.append(size[i])
            continue
        ti = index[i]
        if isinstance(ti, int):
            newindex.append(1)
        elif isinstance(ti, slice):
            newindex.append(canonical_dim(ti.stop, size[i], end=True) - canonical_dim(ti.start, size[i]))
        elif isinstance(ti, np.ndarray):
            newindex.append(len(ti))
        else:
            assert 0
    return tuple(newindex)


def canonical_index(index, shape):
    newindex = []
    if not isinstance(index, tuple):
        index = (index,)
    assert len(index) <= len(shape)
    for i in range(len(shape)):
        if i >= len(index):
            newindex.append(slice(0, shape[i]))
            continue
        ti = index[i]
        if isinstance(ti, int):
            ni = canonical_dim(ti, shape[i])
            newindex.append(slice(ni, ni + 1))
        elif isinstance(ti, slice):
            newindex.append(
                slice(
                    canonical_dim(ti.start, shape[i]),
                    canonical_dim(ti.stop, shape[i], end=True),
                )
            )
        else:
            assert 0
    return tuple(newindex)


def numpy_broadcast_shape(a, b):
    # This function can work on tuples containing shapes but then the scalar output section may not work.
    if isinstance(a, tuple):
        rev_ashape = a[::-1]
    else:
        rev_ashape = a.shape[::-1]

    if isinstance(b, tuple):
        rev_bshape = b[::-1]
    else:
        if isinstance(b, (ndarray, np.ndarray)):
            rev_bshape = b.shape[::-1]
        elif isinstance(b, numbers.Number):
            rev_bshape = ()
        else:
            rev_bshape = (1,)

    # scalar output
    if (isinstance(a, numbers.Number) or rev_ashape == ()) and (
        isinstance(b, numbers.Number) or rev_bshape == ()
    ):
        return None

    dprint(3, "numpy_broadcast_shape:", rev_ashape, rev_bshape)

    # make sure that ashape is not shorter than bshape
    if len(rev_bshape) > len(rev_ashape):
        rev_ashape, rev_bshape = rev_bshape, rev_ashape
    new_shape = []

    for i in range(len(rev_bshape)):
        if rev_ashape[i] == 1:
            new_shape.append(rev_bshape[i])
        elif rev_bshape[i] == 1:
            new_shape.append(rev_ashape[i])
        elif rev_ashape[i] != rev_bshape[i]:
            raise ValueError(
                "operands could not be broadcast together with shapes "
                + str(a.shape)
                + " and "
                + str(b.shape)
            )
        else:
            new_shape.append(rev_ashape[i])

    diff = len(rev_bshape) - len(rev_ashape)
    if diff < 0:
        new_shape.extend(rev_ashape[diff:])

    return tuple(new_shape[::-1])


def make_method(
    name,
    optext,
    inplace=False,
    unary=False,
    reduction=False,
    reverse=False,
    imports=[],
    dtype=None,
):
    if unary:

        def _method(self, **kwargs):
            if "dtype" not in kwargs:
                kwargs["dtype"] = dtype
            retval = self.array_unaryop(
                name, optext, reduction, imports=imports, **kwargs
            )
            return retval

    else:

        def _method(self, rhs):
            t0 = timer()
            #            print("make_method", name, type(self), type(rhs))
            #            if isinstance(self, ndarray) and isinstance(rhs, ndarray):
            #                assert(self.shape == rhs.shape)
            new_ndarray = self.array_binop(
                rhs, name, optext, inplace, reverse, imports=imports, dtype=dtype
            )
            t1 = timer()
            dprint(4, "BIN METHOD: time", (t1 - t0) * 1000)
            return new_ndarray

    _method.__name__ = name
    return _method


class op_info:
    def __init__(self, code, imports=[], dtype=None):
        self.code = code
        self.imports = imports
        self.dtype = dtype


array_binop_funcs = {
    "__add__": op_info(" + "),
    "__mul__": op_info(" * "),
    "__sub__": op_info(" - "),
    "__floordiv__": op_info(" // "),
    "__truediv__": op_info(" / ", dtype="float"),
    "__mod__": op_info(" % "),
    "__pow__": op_info(" ** "),
    "__gt__": op_info(" > ", dtype=np.bool_),
    "__lt__": op_info(" < ", dtype=np.bool_),
    "__ge__": op_info(" >= ", dtype=np.bool_),
    "__le__": op_info(" <= ", dtype=np.bool_),
    "__eq__": op_info(" == ", dtype=np.bool_),
    "__ne__": op_info(" != ", dtype=np.bool_),
}
for (abf, code) in array_binop_funcs.items():
    new_func = make_method(abf, code.code, imports=code.imports, dtype=code.dtype)
    setattr(ndarray, abf, new_func)

array_binop_rfuncs = {
    "__radd__": op_info(" + "),
    "__rmul__": op_info(" * "),
    "__rsub__": op_info(" - "),
    "__rtruediv__": op_info(" / ", dtype="float"),
    "__rfloordiv__": op_info(" // "),
    "__rmod__": op_info(" % "),
    "__rpow__": op_info(" ** "),
}
for (abf, code) in array_binop_rfuncs.items():
    new_func = make_method(abf, code.code, dtype=code.dtype, reverse=True)
    setattr(ndarray, abf, new_func)

array_inplace_binop_funcs = {
    "__iadd__": " += ",
    "__isub__": " -= ",
    "__imul__": " *= ",
    "__itruediv__": " /= ",
    "__ifloordiv__": " //= ",
    "__imod__": " %= ",
    "__ipow__": " **= ",
}
for (abf, code) in array_inplace_binop_funcs.items():
    new_func = make_method(abf, code, inplace=True)
    setattr(ndarray, abf, new_func)

array_unaryop_funcs = {
    "__abs__": op_info(" numpy.abs", imports=["numpy"]),
    "abs": op_info(" numpy.abs", imports=["numpy"]),
    "square": op_info(" numpy.square", imports=["numpy"]),
    "sqrt": op_info(" numpy.sqrt", imports=["numpy"], dtype="float"),
    "sin": op_info(" numpy.sin", imports=["numpy"], dtype="float"),
    "cos": op_info(" numpy.cos", imports=["numpy"], dtype="float"),
    "tan": op_info(" numpy.tan", imports=["numpy"], dtype="float"),
    "arcsin": op_info(" numpy.arcsin", imports=["numpy"], dtype="float"),
    "arccos": op_info(" numpy.arccos", imports=["numpy"], dtype="float"),
    "arctan": op_info(" numpy.arctan", imports=["numpy"], dtype="float"),
    "__neg__": op_info(" -"),
    "exp": op_info(" math.exp", imports=["math"], dtype="float"),
    "log": op_info(" math.log", imports=["math"], dtype="float"),
    "isnan": op_info(" numpy.isnan", imports=["numpy"], dtype=np.bool_),
}

for (abf, code) in array_unaryop_funcs.items():
    new_func = make_method(
        abf, code.code, unary=True, imports=code.imports, dtype=code.dtype
    )
    setattr(ndarray, abf, new_func)

array_simple_reductions = ["sum", "prod", "min", "max"]
for abf in array_simple_reductions:
    new_func = make_method(
        abf, "np." + abf, imports=["numpy"], unary=True, reduction=True
    )
    setattr(ndarray, abf, new_func)


class deferred_op:
    ramba_deferred_ops = None
    count = 0
    last_call_time = timer()
    last_add_time = timer()

    def __init__(self, shape, distribution, fdist):
        self.shape = shape
        self.distribution = distribution
        self.flex_dist = fdist
        self.delete_gids = []
        self.read_arrs = []   # (gid,dist) list of read arrays;  dist is None if flex_dist
        self.write_arrs = []  # (gid,dist) list of write arrays; dist is None if flex_dist
        self.use_gids = (
            {}
        )  # map gid to tuple ([(tmp var name, array details)*], bdarray dist, pad)
        self.preconstructed_gids = (
            {}
        )  # subset of use_gids that already are remote_constructed
        self.use_other = {}  # map tmp var name to pickled data
        self.codelines = []
        self.imports = []
        self.varcount = 0
        self.uuid = "ramba_def_ops_%05d" % (deferred_op.count)
        deferred_op.count += 1

    def get_var_name(self):
        nm = "ramba_tmp_var_%05d" % (self.varcount)
        self.varcount += 1
        return nm

    def add_gid(self, gid, arr_info, bd_info, pad, flex_dist):
        if gid not in self.use_gids:
            # self.use_gids[gid] = tuple([self.get_var_name(), arr_info, bd_info, pad])
            self.use_gids[gid] = ([], bd_info, pad, flex_dist)
        for (v, ai) in self.use_gids[gid][0]:
            if shardview.dist_is_eq(arr_info[1], ai[1]):
                return v
        v = self.get_var_name()
        dprint(4, "deferred_ops.ad_gid:", v, gid, arr_info[1])
        self.use_gids[gid][0].append((v, arr_info))
        return v

    def add_var(self, v):
        nm = self.get_var_name()
        self.use_other[nm] = pickle.dumps(v)
        return nm

    # Execute what we have now
    def execute(self):
        times = [timer()]
        # Do stuff, then clear out list
        # print("Doing deferred ops", self.codelines)
        # find if any used ndarrays are still in scope, or were already remote_created
        #     create these remotely if needed
        #     these will need to be indexed (i.e., not a per element scalar temp)
        live_gids = {
            k: v
            for (k, v) in self.use_gids.items()
            if bdarray.valid_gid(k) or k in self.preconstructed_gids
        }
        dprint(3, "use_gids:", self.use_gids.keys(), "\nlive_gids", live_gids.keys(), "\npreconstructed gids", self.preconstructed_gids)
        # Change distributions for any flexible arrays -- we should not have slices here
        for (_, (_, d, _, flex)) in live_gids.items():
            if flex:
                #dcopy = libcopy.deepcopy(self.distribution)
                dcopy = shardview.clean_dist(self.distribution)
                # d.clear()
                for i, v in enumerate(
                    dcopy
                ):  # deep copy distribution, but keep reference same as other arrays may be pointing to the same one
                    d[i] = v
        times.append(timer())
        # substitute var_name with var_name[index] for ndarrays
        for (k, v) in live_gids.items():
            for (vn, ai) in v[0]:
                for i in range(len(self.codelines)):
                    self.codelines[i] = self.codelines[i].replace(vn, vn + "[index]")
        times.append(timer())
        # compile into function, using unpickled variables (non-ndarrays)
        precode = []
        args = []
        for i, (k, v) in enumerate(live_gids.items()):
            # precode.append("  "+v+" = arrays["+str(i)+"]")
            for (vn, ai) in v[0]:
                args.append(vn)
        for i, (v, b) in enumerate(self.use_other.items()):
            # precode.append("  "+v+" = vars["+str(i)+"]")
            args.append(v)
        # precode.append("  import numpy as np")
        # precode.append("\n".join(self.imports))
        if len(self.codelines)>0:
            precode.append(
                "  for index in numba.pndindex("
                + list(live_gids.items())[0][1][0][0][0]
                + ".shape):"
            )
        code = "\n".join(precode) + "\n" + "\n".join(self.codelines)
        fname = "ramba_deferred_ops_func_" + str(len(args)) + str(abs(hash(code)))
        code = "def " + fname + "(" + ",".join(args) + "):\n" + code + "\n  pass"
        # code = "@numba.njit\ndef "+fname+"("+",".join(args)+"):\n"+code
        dprint(2, "Updated code:\n" + code)
        times.append(timer())
        # execute on remote nodes
        # [ remote_states[i].run_deferred_ops.remote(
        #              self.uuid, live_gids, self.delete_gids, self.use_other.items(), self.distribution, fname, code, self.imports)
        #          for i in range(num_workers) ]
        remote_exec_all(
            "run_deferred_ops",
            self.uuid,
            live_gids,
            self.delete_gids,
            list(self.use_other.items()),
            self.distribution,
            fname,
            code,
            self.imports,
        )
        times.append(timer())
        # all live arrays used should be constructed by now
        for k in live_gids.keys():
            if k not in self.delete_gids:
                bdarray.get_by_gid(
                    k
                ).remote_constructed = True  # set remote_constructed
                bdarray.get_by_gid(k).flex_dist = False  # not flexible anymore
        times.append(timer())
        if ntiming >= 1:
            times = [
                int((times[i] - times[i - 1]) * 1000000) / 1000
                for i in range(1, len(times))
            ]
            print("Deferred Ops execution", times)
            tprint(2, "deferred_ops::execute code", code)

    # TODO: There may be a race condition here.  Need to check and fix.
    # deferred (if needed) deletions of the remote arrays
    @classmethod
    def del_remote_array(cls, gid):
        if cls.ramba_deferred_ops is None:
            # if ray.is_initialized(): [remote_states[i].destroy_array.remote(gid) for i in range(num_workers)]
            if USE_MPI or ray.is_initialized():
                remote_exec_all("destroy_array", gid)
        else:
            cls.ramba_deferred_ops.delete_gids.append(gid)

    # static method to ensure deferred ops are done;  should be called before any immediate remote ops
    @classmethod
    def do_ops(cls):
        if cls.ramba_deferred_ops is not None:
            t0 = timer()
            cls.ramba_deferred_ops.execute()
            cls.ramba_deferred_ops = None
            t1 = timer()
            dprint(
                1,
                "DO_OPS: ",
                (t0 - cls.last_call_time) * 1000,
                " executs: ",
                (t1 - t0) * 1000,
            )
            cls.last_call_time = t0

    # add operations to the deferred stack; if not compatible, execute what we have first
    @classmethod
    def add_op(
        cls, oplist, write_array, imports=[]
    ):  # oplist is a list of alternating code string, variable reference, code ...
        t0 = timer()
        dprint(1, "ADD_OP: time from last", (t0 - cls.last_add_time) * 1000)
        # get code and variables from oplist
        codes = oplist[::2]
        operands = oplist[1::2]
        # find ndarray -- assume all are compatible (same shape, divisions);  TODO: should check
        arr = (
            write_array
            if write_array is not None
            else next(filter(lambda x: isinstance(x, ndarray), operands), None)
        )
        assert arr is not None, "Deferred op with no ndarray parameter"
        shape = arr.shape
        distribution = arr.distribution
        # Check to see if existing deferred ops are compatible in size, else do ops
        if (cls.ramba_deferred_ops is not None) and (
            cls.ramba_deferred_ops.shape != shape
            or (
                not shardview.compatible_distributions(
                    cls.ramba_deferred_ops.distribution, distribution
                )
                and (arr.bdarray.remote_constructed or not arr.bdarray.flex_dist)
                and not cls.ramba_deferred_ops.flex_dist
            )
        ):
            dprint(
                2,
                "deferred ops mismatch:",
                cls.ramba_deferred_ops.shape,
                shape,
                cls.ramba_deferred_ops.distribution,
                distribution,
            )
            cls.ramba_deferred_ops.do_ops()

        # Alias check 1;  op reads/writes shifted version of an array that was written previously
        if ( cls.ramba_deferred_ops is not None and any([
                isinstance(o, ndarray) and o.gid==wgid and wdist is not None and not shardview.dist_is_eq(wdist, o.distribution)
                for o in operands for (wgid,wdist) in cls.ramba_deferred_ops.write_arrs]) ):
            dprint(2, "Read/write after write with mismatched distributions detected. Will not fuse.")
            # force prior stuff to execute
            cls.ramba_deferred_ops.do_ops()

        # Alias check 2:  op writes and reads from shifted versions of same array
        #    NOTE:  works only for assignment / inplace operators, not general functions
        if (write_array is not None) and (
                any ([ isinstance(o, ndarray) and o.gid==write_array.gid and not shardview.dist_is_eq(o.distribution, write_array.distribution) for o in operands ])
                or (cls.ramba_deferred_ops is not None and any([ rgid==write_array.gid and (rdist is not None) and not shardview.dist_is_eq(rdist, write_array.distribution) for (rgid,rdist) in cls.ramba_deferred_ops.read_arrs])) ):
            assert codes[0]==''
            dprint(2, "Write after read with mismatched distributions detected. Will not fuse, adding temporary array.")
            # make temp, assign computation to temp, do ops, then assign temp to write array
            tmp_array = empty_like(write_array)
            orig_op = codes[1]
            oplist[1] = tmp_array
            oplist[2] = ' = '
            cls.add_op( oplist, tmp_array, imports )
            cls.ramba_deferred_ops.do_ops()
            oplist = [ '', write_array, orig_op, tmp_array ]
            operands = [ write_array, tmp_array ]
            codes = [ '', orig_op ]

        if cls.ramba_deferred_ops is None:
            dprint(2, "Create new deferred op set")
            cls.ramba_deferred_ops = cls(shape, distribution, arr.bdarray.flex_dist)
        if not arr.bdarray.flex_dist:
            cls.ramba_deferred_ops.distribution = distribution
            cls.ramba_deferred_ops.flex_dist = False
            dprint(2, "fixing distribution")

        # Handle masked array write
        if write_array.maskarray is not None:
            oplist = [ "if ", write_array.maskarray, ": "+oplist[0] ] + oplist[1:]
            operands = [ write_array.maskarray ] + operands

        # add vars
        if (write_array is not None):
            cls.ramba_deferred_ops.write_arrs.append((write_array.gid, None if write_array.bdarray.flex_dist else write_array.distribution))
        for i, x in enumerate(operands):
            if isinstance(x, ndarray):
                cls.ramba_deferred_ops.read_arrs.append((x.gid, None if x.bdarray.flex_dist else x.distribution))
                if x.shape == ():  # 0d check
                    oplist[1 + 2 * i] = cls.ramba_deferred_ops.add_var(x.distribution)
                else:
                    bd = bdarray.get_by_gid(x.gid)
                    oplist[1 + 2 * i] = cls.ramba_deferred_ops.add_gid(
                        x.gid,
                        x.get_details(),
                        bd.distribution,
                        bd.pad,
                        bd.flex_dist and not bd.remote_constructed,
                    )
                    # check if already remote_constructed
                    if bd.remote_constructed:
                        cls.ramba_deferred_ops.preconstructed_gids[x.gid] = oplist[
                            1 + 2 * i
                        ]
            else:
                oplist[1 + 2 * i] = cls.ramba_deferred_ops.add_var(x)
        # add codeline to list
        codeline = "    " + "".join(oplist)
        if oplist[0]!="#":   # avoid adding empty comment -- should really check if starts with #
            cls.ramba_deferred_ops.codelines.append(codeline)
        cls.ramba_deferred_ops.imports.extend(imports)
        t1 = timer()
        cls.last_add_time = t0
        dprint(2, "Added line:", codeline, "Time:", (t1 - t0) * 1000)


def create_array_with_divisions(shape, divisions, local_border=0, dtype=None):
    new_ndarray = ndarray(
        shape,
        distribution=shardview.clean_dist(divisions),
        local_border=local_border,
        dtype=dtype,
    )
    dprint(3, "divisions:", divisions)
    return new_ndarray


def create_array_executor(
    temp_ndarray,
    shape,
    filler,
    local_border=0,
    dtype=None,
    distribution=None,
    tuple_arg=True,
    filler_prepickled=False,
    no_defer=False,
    **kwargs
):
    new_ndarray = ndarray(
        shape,
        local_border=local_border,
        dtype=dtype,
        distribution=distribution,
        **kwargs
    )
    if shape == ():
        if isinstance(filler, str):
            new_ndarray.distribution = np.array(eval(filler), dtype=new_ndarray.dtype)
        elif isinstance(filler, numbers.Number):
            new_ndarray.distribution = np.array(filler, dtype=new_ndarray.dtype)
        elif isinstance(filler, np.ndarray) and filler.shape == ():
            new_ndarray.distribution = filler.copy()
        return new_ndarray

    if isinstance(filler, str): # ignore no_defer
        deferred_op.add_op(["", new_ndarray, " = " + filler], new_ndarray)
    elif filler is None and no_defer==False:
        deferred_op.add_op(["#", new_ndarray], new_ndarray) # deferred op no op, just to make sure empty array is constructed
    else:
        filler = filler if filler_prepickled else func_dumps(filler)
        [
            remote_exec(
                i,
                "create_array",
                new_ndarray.gid,
                new_ndarray.distribution[i],
                shape,
                filler,
                local_border,
                dtype,
                new_ndarray.distribution,
                new_ndarray.from_border[i]
                if new_ndarray.from_border is not None
                else None,
                new_ndarray.to_border[i] if new_ndarray.to_border is not None else None,
                tuple_arg,
            )
            for i in range(num_workers)
        ]
        new_ndarray.bdarray.remote_constructed = True  # set remote_constructed = True
        new_ndarray.bdarray.flex_dist = False  # distribution is fixed
    return new_ndarray


@DAGapi
def create_array(
    shape,
    filler,
    local_border=0,
    dtype=None,
    distribution=None,
    tuple_arg=True,
    filler_prepickled=False,
    no_defer=False,
    **kwargs
):
    if no_defer:
        return create_array_executor(None,
                                     shape,
                                     filler,
                                     local_border=local_border,
                                     dtype=dtype,
                                     distribution=distribution,
                                     tuple_arg=tuple_arg,
                                     filler_prepickled=filler_prepickled,
                                     no_defer=no_defer,
                                     **kwargs)
    return DAGshape(shape, dtype, False)

"""
def create_array(
    shape,
    filler,
    local_border=0,
    dtype=None,
    distribution=None,
    tuple_arg=True,
    filler_prepickled=False,
    no_defer=False,
    **kwargs
):
    new_ndarray = ndarray(
        shape,
        local_border=local_border,
        dtype=dtype,
        distribution=distribution,
        **kwargs
    )
    if shape == ():
        if isinstance(filler, str):
            new_ndarray.distribution = np.array(eval(filler), dtype=new_ndarray.dtype)
        elif isinstance(filler, numbers.Number):
            new_ndarray.distribution = np.array(filler, dtype=new_ndarray.dtype)
        elif isinstance(filler, np.ndarray) and filler.shape == ():
            new_ndarray.distribution = filler.copy()
        return new_ndarray

    if isinstance(filler, str): # ignore no_defer
        deferred_op.add_op(["", new_ndarray, " = " + filler], new_ndarray)
    elif filler is None and no_defer==False:
        deferred_op.add_op(["#", new_ndarray], new_ndarray) # deferred op no op, just to make sure empty array is constructed
    else:
        filler = filler if filler_prepickled else func_dumps(filler)
        [
            remote_exec(
                i,
                "create_array",
                new_ndarray.gid,
                new_ndarray.distribution[i],
                shape,
                filler,
                local_border,
                dtype,
                new_ndarray.distribution,
                new_ndarray.from_border[i]
                if new_ndarray.from_border is not None
                else None,
                new_ndarray.to_border[i] if new_ndarray.to_border is not None else None,
                tuple_arg,
            )
            for i in range(num_workers)
        ]
        new_ndarray.bdarray.remote_constructed = True  # set remote_constructed = True
        new_ndarray.bdarray.flex_dist = False  # distribution is fixed
    return new_ndarray
"""


def init_array(
    shape,
    filler,
    local_border=0,
    dtype=None,
    distribution=None,
    tuple_arg=True,
    **kwargs
):
    if isinstance(shape, int):
        shape = (shape,)
    return create_array(
        shape,
        filler,
        local_border=local_border,
        dtype=dtype,
        distribution=distribution,
        tuple_arg=tuple_arg,
        **kwargs
    )


def fromfunction(lfunc, shape, dtype=None, **kwargs):
    return init_array(shape, lfunc, dtype=dtype, tuple_arg=False, **kwargs)


# TODO: creating an empty array and then using in a non-deferred-op skeleton may not work
def empty(shape, local_border=0, dtype=None, distribution=None, **kwargs):
    return init_array(
        shape,
        None,
        local_border=local_border,
        dtype=dtype,
        distribution=distribution,
        **kwargs
    )


def empty_like(other_ndarray,**kwargs):
    return empty(
        other_ndarray.shape,
        local_border=other_ndarray.local_border,
        dtype=other_ndarray.dtype,
        **kwargs
    )


def zeros(shape, local_border=0, dtype=None, distribution=None, **kwargs):
    return init_array(
        shape,
        "0",
        local_border=local_border,
        dtype=dtype,
        distribution=distribution,
        **kwargs
    )


def zeros_like(other_ndarray):
    return zeros(
        other_ndarray.shape,
        local_border=other_ndarray.local_border,
        dtype=other_ndarray.dtype,
    )


def ones(shape, local_border=0, dtype=None, distribution=None, **kwargs):
    return init_array(shape, "1", local_border=local_border, distribution=distribution, dtype=dtype, **kwargs)


def ones_like(other_ndarray):
    return ones(
        other_ndarray.shape,
        local_border=other_ndarray.local_border,
        dtype=other_ndarray.dtype,
    )


def full(shape, v, local_border=0, **kwargs):
    return init_array(shape, str(v), local_border=local_border, **kwargs)


def eye(N, M=None, dtype=float32, local_border=0, **kwargs):
    if M is None:
        M = N
    return init_array(
        (N, M),
        lambda x: 1 if x[0] == x[1] else 0,
        local_border=local_border,
        dtype=dtype,
        **kwargs
    )


def copy(arr, local_border=0):
    dprint(1, "copy")
    new_ndarray = create_array_with_divisions(
        arr.shape, arr.distribution, local_border=local_border
    )
    deferred_op.add_op(["", new_ndarray, " = ", arr, ""], new_ndarray)
    """
    [remote_states[i].copy.remote(new_ndarray.gid,
                                  arr.gid,
                                  local_border,
                                  new_ndarray.from_border[i] if new_ndarray.from_border is not None else None,
                                  new_ndarray.to_border[i] if new_ndarray.to_border is not None else None)
        for i in range(num_workers)]
    """
    return new_ndarray


# TODO: should make this deferred;  need a njit function to provide global index
def arange(size, local_border=0):
    return init_array(size, lambda x: x[0], local_border=local_border, dtype=int64)


def asarray(x):
    dprint(1, "asarray global")
    assert isinstance(x, ndarray)
    return x.asarray()


def fromarray_executor(temp_array, x, local_border=0, dtype=None, **kwargs):
    dprint(1, "fromarray global")
    if isinstance(x, numbers.Number):
        return create_array((), filler=x, dtype=dtype, **kwargs)
    if isinstance(x, np.ndarray) and x.shape == ():  # 0d array
        return create_array((), filler=x, dtype=dtype, **kwargs)
    if isinstance(x, list):
        x = np.array(x)

    shape = temp_array.shape
    dtype = temp_array.dtype
    new_ndarray = create_array(
        shape, None, local_border=local_border, dtype=dtype, **kwargs
    )
    distribution = new_ndarray.distribution
    [
        remote_exec(
            i,
            "push_array",
            new_ndarray.gid,
            distribution[i],
            shape,
            x[shardview.to_slice(distribution[i])],
            local_border,
            distribution,
            new_ndarray.from_border[i] if new_ndarray.from_border is not None else None,
            new_ndarray.to_border[i] if new_ndarray.to_border is not None else None,
            x.dtype,
        )
        for i in range(num_workers)
    ]
    new_ndarray.bdarray.remote_constructed = True  # set remote_constructed = True
    return new_ndarray


@DAGapi
def fromarray(x, local_border=0, dtype=None, **kwargs):
    dprint(1, "fromarray global")
    if isinstance(x, numbers.Number):
        return DAGshape((), dtype, False)
    if isinstance(x, np.ndarray) and x.shape == ():  # 0d array
        return DAGshape((), dtype, False)

    if isinstance(x, list):
        x = np.array(x)

    shape = x.shape

    if dtype is None:
        dtype = x.dtype
    return DAGshape(shape, dtype, False)


"""
def fromarray(x, local_border=0, dtype=None, **kwargs):
    dprint(1, "fromarray global")
    if isinstance(x, numbers.Number):
        return create_array((), filler=x, dtype=dtype, **kwargs)
    if isinstance(x, np.ndarray) and x.shape == ():  # 0d array
        return create_array((), filler=x, dtype=dtype, **kwargs)
    if isinstance(x, list):
        x = np.array(x)

    shape = x.shape
    if dtype is None:
        dtype = x.dtype
    new_ndarray = create_array(
        shape, None, local_border=local_border, dtype=dtype, **kwargs
    )
    DAG.instantiate(new_ndarray)
    distribution = new_ndarray.distribution
    [
        remote_exec(
            i,
            "push_array",
            new_ndarray.gid,
            distribution[i],
            shape,
            x[shardview.to_slice(distribution[i])],
            local_border,
            distribution,
            new_ndarray.from_border[i] if new_ndarray.from_border is not None else None,
            new_ndarray.to_border[i] if new_ndarray.to_border is not None else None,
            x.dtype,
        )
        for i in range(num_workers)
    ]
    new_ndarray.bdarray.remote_constructed = True  # set remote_constructed = True
    return new_ndarray
"""


def array(*args):
    return fromarray(*args)


def load(fname, dtype=None, local=False, ftype=None, **kwargs):
    fldr = fileio.get_load_handler(fname, ftype)
    if local or not fldr.is_dist:
        print("efficient local file load has not yet been implemented")
        tmp = fldr.readall(fname, **kwargs)
        if dtype is None:
            dtype = tmp.dtype
        arr = fromarray(tmp, dtype=dtype)
    else:
        shp, dt = fldr.getinfo(fname, **kwargs)
        if dtype is None:
            dtype = dt
        arr = empty(shp, dtype=dtype)
        arr.instantiate()
        remote_exec_all("load", arr.gid, fname, **kwargs)
    return arr


def transpose(a, *args):
    dprint(1, "transpose global")
    return a.transpose(*args)


def _compute_remote_ranges(out_distribution, out_mapping):
    divisions = shardview.distribution_to_divisions(out_distribution)
    dprint(
        2,
        "_compute_remote_ranges:",
        out_distribution,
        "out_mapping:",
        out_mapping,
        "\ndivisions:",
        divisions,
    )
    from_ret = [[] for i in range(num_workers)]
    to_ret = [[] for i in range(num_workers)]
    for i in range(len(out_distribution)):
        out_on_i = out_distribution[i]
        out_on_i_end = shardview.get_start(out_on_i) + shardview.get_size(out_on_i) - 1
        out_combined = np.concatenate(
            [
                np.expand_dims(shardview.get_start(out_on_i), axis=0),
                np.expand_dims(out_on_i_end, axis=0),
            ],
            axis=0,
        )
        dprint(3, "out_on_i", i, out_on_i, out_on_i_end, out_combined)
        for (
            map_out_start,
            map_out_end,
            map_other_start,
            map_other_end,
            other_array,
        ) in out_mapping:
            map_out_combined = np.concatenate(
                [
                    np.expand_dims(map_out_start, axis=0),
                    np.expand_dims(map_out_end, axis=0) - 1,
                ],
                axis=0,
            )
            bires = shardview.block_intersection(out_combined, map_out_combined)
            dprint(
                3,
                "map_out_combined\n",
                out_combined,
                "\n",
                map_out_combined,
                "\n",
                bires,
            )
            if bires is not None:
                diff = bires - map_out_combined[0]
                dprint(3, "diff:", diff)
                grrres = other_array.get_remote_ranges(diff)
                for node_num, bi, sview in grrres:
                    dprint(3, "grres:", i, node_num, bi, sview)
                    from_ret[i].append(
                        (other_array.gid, node_num, bi, map_out_combined)
                    )
                    to_ret[node_num].append((other_array.gid, i, bi, map_out_combined))
    return from_ret, to_ret


def implements(numpy_function, uses_self):
    numpy_function = "np." + numpy_function

    def decorator(func):
        HANDLED_FUNCTIONS[eval(numpy_function)] = (func, uses_self)
        return func

    return decorator


def concatenate_executor(temp_array, arrayseq, axis=0, out=None, **kwargs):
    out_shape = list(arrayseq[0].shape)
    found_ndarray = isinstance(arrayseq[0], ndarray)
    first_dtype = arrayseq[0].dtype

    origin_regions = [
        (
            [0] * len(out_shape),
            out_shape.copy(),
            [0] * len(out_shape),
            out_shape.copy(),
            arrayseq[0],
        )
    ]
    for array in arrayseq[1:]:
        found_ndarray = found_ndarray or isinstance(array, ndarray)
        assert len(out_shape) == len(array.shape)
        start = origin_regions[-1][0].copy()
        start[axis] = out_shape[axis]
        for i in range(len(out_shape)):
            if axis == i:
                out_shape[i] += array.shape[i]
            else:
                assert out_shape[i] == array.shape[i]
        assert first_dtype == array.dtype
        origin_regions.append(
            (start, out_shape, [0] * len(out_shape), array.shape, array)
        )
    out_shape = tuple(out_shape)
    assert found_ndarray

    if out is None:
        out = empty(out_shape, dtype=first_dtype, **kwargs, no_defer=True)
        dprint(2, "Create concatenate output:", out.gid, out_shape, out.dtype)
    else:
        assert isinstance(out, ndarray)
        assert out.shape == out_shape

    #deferred_op.do_ops()

    from_ranges, to_ranges = _compute_remote_ranges(out.distribution, origin_regions)
    dprint(2, "from_ranges\n", from_ranges)
    dprint(2, "to_ranges\n", to_ranges)
    #deferred_op.do_ops()

    send_recv_uuid = uuid.uuid4()
    # [remote_states[i].push_pull_copy.remote(out.gid,
    #                                        from_ranges[i],
    #                                        to_ranges[i])
    #    for i in range(num_workers)]
    [
        remote_exec(
            i, "push_pull_copy", out.gid, from_ranges[i], to_ranges[i], send_recv_uuid
        )
        for i in range(num_workers)
    ]
    return out


@implements("concatenate", False)
@DAGapi
def concatenate(arrayseq, axis=0, out=None, **kwargs):
    dprint(1, "concatenate global", len(arrayseq), type(arrayseq))
    out_shape = list(arrayseq[0].shape)
    found_ndarray = isinstance(arrayseq[0], ndarray)
    first_dtype = arrayseq[0].dtype

    origin_regions = [
        (
            [0] * len(out_shape),
            out_shape.copy(),
            [0] * len(out_shape),
            out_shape.copy(),
            arrayseq[0],
        )
    ]
    for array in arrayseq[1:]:
        found_ndarray = found_ndarray or isinstance(array, ndarray)
        assert len(out_shape) == len(array.shape)
        start = origin_regions[-1][0].copy()
        start[axis] = out_shape[axis]
        for i in range(len(out_shape)):
            if axis == i:
                out_shape[i] += array.shape[i]
            else:
                assert out_shape[i] == array.shape[i]
        assert first_dtype == array.dtype
        origin_regions.append(
            (start, out_shape, [0] * len(out_shape), array.shape, array)
        )
    out_shape = tuple(out_shape)
    dprint(2, "concatenate out_shape:", out_shape, first_dtype)
    assert found_ndarray
    return DAGshape(out_shape, first_dtype, False)


"""
@implements("concatenate", False)
def concatenate(arrayseq, axis=0, out=None, **kwargs):
    dprint(1, "concatenate global", len(arrayseq))
    out_shape = list(arrayseq[0].shape)
    found_ndarray = isinstance(arrayseq[0], ndarray)
    first_dtype = arrayseq[0].dtype

    origin_regions = [
        (
            [0] * len(out_shape),
            out_shape.copy(),
            [0] * len(out_shape),
            out_shape.copy(),
            arrayseq[0],
        )
    ]
    for array in arrayseq[1:]:
        found_ndarray = found_ndarray or isinstance(array, ndarray)
        assert len(out_shape) == len(array.shape)
        start = origin_regions[-1][0].copy()
        start[axis] = out_shape[axis]
        for i in range(len(out_shape)):
            if axis == i:
                out_shape[i] += array.shape[i]
            else:
                assert out_shape[i] == array.shape[i]
        assert first_dtype == array.dtype
        origin_regions.append(
            (start, out_shape, [0] * len(out_shape), array.shape, array)
        )
    out_shape = tuple(out_shape)
    dprint(2, "concatenate out_shape:", out_shape, first_dtype)
    assert found_ndarray

    if any([isinstance(x, ndarray) and x.advindex is not None for x in arrayseq]):
        res = ndarray(
            out_shape,
            dtype=first_dtype,
            advindex=("concatenate", (arrayseq, axis, out))
        )
        res.process_advindex()
        return res


    if out is None:
        out = empty(out_shape, dtype=first_dtype, **kwargs, no_defer=True)
        dprint(2, "Create concatenate output:", out.gid, out_shape, out.dtype)
    else:
        assert isinstance(out, ndarray)
        assert out.shape == out_shape
    deferred_op.do_ops()

    from_ranges, to_ranges = _compute_remote_ranges(out.distribution, origin_regions)
    dprint(2, "from_ranges\n", from_ranges)
    dprint(2, "to_ranges\n", to_ranges)
    deferred_op.do_ops()

    send_recv_uuid = uuid.uuid4()
    # [remote_states[i].push_pull_copy.remote(out.gid,
    #                                        from_ranges[i],
    #                                        to_ranges[i])
    #    for i in range(num_workers)]
    [
        remote_exec(
            i, "push_pull_copy", out.gid, from_ranges[i], to_ranges[i], send_recv_uuid
        )
        for i in range(num_workers)
    ]
    return out
"""


prop_to_array = [
    "ndim",
]


for mfunc in prop_to_array:
    mcode = "def " + mfunc + "(the_array):\n"
    mcode += "    if isinstance(the_array, ndarray):\n"
    mcode += "        return the_array." + mfunc + "\n"
    mcode += "    else:\n"
    mcode += "        return np." + mfunc + "(the_array)\n"
    exec(mcode)
    implements(mfunc, False)(eval(mfunc))


mod_to_array = [
    "sum",
    "prod",
    "exp",
    "log",
    "isnan",
    "abs",
    "square",
    "sqrt",
    "mean",
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "nanmean",
]
for mfunc in mod_to_array:
    mcode = "def " + mfunc + "(the_array, *args, **kwargs):\n"
    mcode += "    if isinstance(the_array, ndarray):\n"
    mcode += "        return the_array." + mfunc + "(*args, **kwargs)\n"
    mcode += "    else:\n"
    mcode += "        return np." + mfunc + "(the_array, *args, **kwargs)\n"
    exec(mcode)
    implements(mfunc, False)(eval(mfunc))

# def isnan(the_array, *args, **kwargs):
#    if isinstance(the_array, ndarray):
#        return the_array.isnan(*args, **kwargs)
#    else:
#        return np.isnan(the_array)
#
# def abs(the_array, *args, **kwargs):
#    if isinstance(the_array, ndarray):
#        return the_array.abs(*args, **kwargs)
#    else:
#        return np.abs(the_array)
#
# def mean(the_array, *args, **kwargs):
#    if isinstance(the_array, ndarray):
#        return the_array.sum(*args, **kwargs)/np.prod(the_array.shape)
#    else:
#        return np.mean(the_array, *args, **kwargs)


def power(a, b):
    if isinstance(a, ndarray) or isinstance(b, ndarray):
        return a ** b
    else:
        return np.power(a, b)


def where_executor(temp_array, cond, a, b):
    if isinstance(cond, np.ndarray):
        a = fromarray(cond)
    if isinstance(a, np.ndarray):
        a = fromarray(a)
    if isinstance(b, np.ndarray):
        b = fromarray(b)
    assert (
        isinstance(cond, ndarray) and isinstance(a, ndarray) and isinstance(b, ndarray)
    )
    dprint(2, "where:", cond.shape, a.shape, b.shape, cond, a, b)

    lb = max(a.local_border, b.local_border)
    ab_new_array_shape, ab_aview, ab_bview = ndarray.broadcast(a, b)
    conda_new_array_shape, conda_condview, conda_aview = ndarray.broadcast(
        cond, ab_aview
    )
    condb_new_array_shape, condb_condview, condb_bview = ndarray.broadcast(
        cond, ab_bview
    )
    assert conda_new_array_shape == condb_new_array_shape

    dprint(2, "newshape:", ab_new_array_shape)
    new_ndarray = empty(conda_new_array_shape, dtype=a.dtype, local_border=lb)
    deferred_op.add_op(
        [
            "",
            new_ndarray,
            " = ",
            conda_aview,
            " if ",
            conda_condview,
            " else ",
            condb_bview,
        ],
        new_ndarray,
        imports=[],
    )
    return new_ndarray


@implements("where", False)
@DAGapi
def where(cond, a, b):
    dprint(2, "where:", cond.shape, a.shape, b.shape, cond, a, b)
    ab_shape = numpy_broadcast_shape(a, b)
    condab_shape = numpy_broadcast_shape(cond, ab_shape)
    return DAGshape(condab_shape, a.dtype, False)


"""
@implements("where", False)
def where(cond, a, b):
    if isinstance(a, np.ndarray):
        a = fromarray(a)
    if isinstance(b, np.ndarray):
        b = fromarray(b)
    assert (
        isinstance(cond, ndarray) and isinstance(a, ndarray) and isinstance(b, ndarray)
    )
    dprint(2, "where:", cond.shape, a.shape, b.shape, cond, a, b)

    lb = max(a.local_border, b.local_border)
    ab_new_array_shape, ab_aview, ab_bview = ndarray.broadcast(a, b)
    conda_new_array_shape, conda_condview, conda_aview = ndarray.broadcast(
        cond, ab_aview
    )
    condb_new_array_shape, condb_condview, condb_bview = ndarray.broadcast(
        cond, ab_bview
    )
    assert conda_new_array_shape == condb_new_array_shape

    dprint(2, "newshape:", ab_new_array_shape)
    new_ndarray = empty(conda_new_array_shape, dtype=a.dtype, local_border=lb)
    # new_ndarray = create_array_with_divisions(conda_new_array_shape, conda_aview.distribution, local_border=lb)
    deferred_op.add_op(
        [
            "",
            new_ndarray,
            " = ",
            conda_aview,
            " if ",
            conda_condview,
            " else ",
            condb_bview,
        ],
        new_ndarray,
        imports=[],
    )
    return new_ndarray
"""

def stack_executor(temp_array, arrays, axis=0, out=None):
    assert 0 # Actually do the operation here if not optimized away.

@implements("stack", False)
@DAGapi
def stack(arrays, axis=0, out=None):
    dprint(1, "stack", len(arrays), type(arrays))
    first_array = arrays[0]
    assert(all([x.shape == first_array.shape for x in arrays]))
    res_size = (len(arrays),) + first_array.shape
    return DAGshape(res_size, first_array.dtype, False)

"""
@implements("stack", False)
def stack(arrays, axis=0, out=None):
    first_array = arrays[0]
    assert(all([x.shape == first_array.shape for x in arrays]))
    res_size = (len(arrays),) + first_array.shape
    res = ndarray(
        res_size,
        dtype=first_array.dtype,
        advindex=("stack", (arrays, axis, out))
    )
    return res
"""


@implements("result_type", False)
def result_type(*args):
    new_args = []
    for arg in args:
        if isinstance(arg, ndarray):
            new_args.append(arg.dtype)
        else:
            new_args.append(arg)
    return np.result_type(*new_args)


def sync():
    sync_start_time = timer()
    DAG.execute_all()
    #deferred_op.do_ops()
    remote_call_all("nop")
    sync_end_time = timer()
    tprint(1, "sync_func_time", sync_end_time - sync_start_time)


class ReshapeError(Exception):
    pass


def expand_dims(a, axis):
    dprint(1, "expand_dims global")
    ashape = a.shape
    if not isinstance(axis, (list, tuple)):
        axis = (axis,)
    new_shape = [0 for _ in range(len(ashape) + len(axis))]
    for newd in axis:
        if newd >= len(new_shape):
            assert 0 # should throw AxisError
        new_shape[newd] = 1
    next_index = 0
    for i in range(len(new_shape)):
        if i in axis:
            continue
        new_shape[i] = ashape[next_index]
        next_index += 1

    return reshape(a, tuple(new_shape))


def squeeze(a, axis=None):
    dprint(1, "squeeze global")
    ashape = a.shape
    if axis is None:
        # axis is tuple of indices that have length 1
        axis = tuple(filter(lambda x: ashape[x] == 1, range(len(ashape))))
    if not isinstance(axis, tuple):
        axis = (axis,)
    # Make sure all the indices they try to remove have length 1
    if not all([ashape[x] == 1 for x in axis]):
        assert 0  # TO-DO: should raise some error

    new_shape = [ashape[i] for i in range(len(ashape)) if i not in axis]
    return reshape(a, tuple(new_shape))


def reshape_executor(temp_array, arr, newshape):
    # first check if this is just for adding / removing dinmensions of size 1 -- this will require no data movement
    realshape = tuple([i for i in arr.shape if i != 1])
    realnewshape = tuple([i for i in newshape if i != 1])
    dprint(2, arr.shape, realshape, newshape, realnewshape)
    if realshape == realnewshape:
        dprint(1, "reshape can be done")
        # make sure array distribution can't change (ie, not flexible or is already constructed)
        if arr.bdarray.flex_dist or not arr.bdarray.remote_constructed:
            deferred_op.do_ops()
        realaxes = [
            i
            for i, v in enumerate(arr.shape)
            if v != 1
        ]
        junkaxes = [
            i
            for i, v in enumerate(arr.shape)
            if v == 1
        ]
        dist = arr.distribution
        oldsize = arr.shape
        dprint(2, realaxes, junkaxes, dist)
        if len(arr.shape) < len(newshape):
            dprint(1, "need to add axes")
            newdims = len(newshape) - len(arr.shape)
            bcastdim = [True if i < newdims else False for i in range(len(newshape))]
            bcastsize = [
                1 if i < newdims else arr.shape[i - newdims]
                for i in range(len(newshape))
            ]
            dist = shardview.broadcast(dist, bcastdim, bcastsize)
            realaxes = [i + newdims for i in realaxes]
            junkaxes = [i for i in range(newdims)] + [i + newdims for i in junkaxes]
            oldsize = tuple(bcastsize)
        dprint(2, realaxes, junkaxes, dist)
        newmap = [junkaxes.pop(0) if v == 1 else realaxes.pop(0) for v in newshape]
        sz, dist = shardview.remap_axis(oldsize, dist, newmap)
        dprint(2, sz, dist)
        return ndarray(
            sz, gid=arr.gid, distribution=dist, local_border=0, readonly=arr.readonly
        )
    # general reshape
    global reshape_forwarding
    if reshape_forwarding:
        return reshape_copy(arr, newshape)
    else:
        raise ReshapeError(
            "ramba.reshape not supported as distributed array reshape cannot be done inplace.  Use reshape_copy instead to create a non-inplace reshape or set RAMBA_RESHAPE_COPY environment variable to convert all reshape calls to reshape_copy."
        )


@DAGapi
def reshape(arr, newshape):
    # first check if this is just for adding / removing dinmensions of size 1 -- this will require no data movement
    realshape = tuple([i for i in arr.shape if i != 1])
    realnewshape = tuple([i for i in newshape if i != 1])
    dprint(2, arr.shape, realshape, newshape, realnewshape)
    if realshape == realnewshape:
        dprint(1, "reshape can be done")
        realaxes = [
            i
            for i, v in enumerate(arr.shape)
            if v != 1
        ]
        junkaxes = [
            i
            for i, v in enumerate(arr.shape)
            if v == 1
        ]
        oldsize = arr.shape
        if len(arr.shape) < len(newshape):
            dprint(1, "need to add axes")
            newdims = len(newshape) - len(arr.shape)
            bcastdim = [True if i < newdims else False for i in range(len(newshape))]
            bcastsize = [
                1 if i < newdims else arr.shape[i - newdims]
                for i in range(len(newshape))
            ]
            realaxes = [i + newdims for i in realaxes]
            junkaxes = [i for i in range(newdims)] + [i + newdims for i in junkaxes]
            oldsize = tuple(bcastsize)
        newmap = [junkaxes.pop(0) if v == 1 else realaxes.pop(0) for v in newshape]
        return DAGshape(shardview.remap_axis_result_shape(oldsize, newmap), arr.dtype, False)

    # general reshape
    global reshape_forwarding
    if reshape_forwarding:
        return reshape_copy(arr, newshape)
    else:
        raise ReshapeError(
            "ramba.reshape not supported as distributed array reshape cannot be done inplace.  Use reshape_copy instead to create a non-inplace reshape or set RAMBA_RESHAPE_COPY environment variable to convert all reshape calls to reshape_copy."
        )


"""
def reshape(arr, newshape):
    # first check if this is just for adding / removing dinmensions of size 1 -- this will require no data movement
    realshape = tuple([i for i in arr.shape if i != 1])
    realnewshape = tuple([i for i in newshape if i != 1])
    dprint(2, arr.shape, realshape, newshape, realnewshape)
    if realshape == realnewshape:
        dprint(1, "reshape can be done")
        # make sure array distribution can't change (ie, not flexible or is already constructed)
        if arr.bdarray.flex_dist or not arr.bdarray.remote_constructed:
            deferred_op.do_ops()
        realaxes = [
            i
            for i, v in enumerate(arr.shape)
            if v != 1
        ]
        junkaxes = [
            i
            for i, v in enumerate(arr.shape)
            if v == 1
        ]
        dist = arr.distribution
        oldsize = arr.shape
        dprint(2, realaxes, junkaxes, dist)
        if len(arr.shape) < len(newshape):
            dprint(1, "need to add axes")
            newdims = len(newshape) - len(arr.shape)
            bcastdim = [True if i < newdims else False for i in range(len(newshape))]
            bcastsize = [
                1 if i < newdims else arr.shape[i - newdims]
                for i in range(len(newshape))
            ]
            dist = shardview.broadcast(dist, bcastdim, bcastsize)
            realaxes = [i + newdims for i in realaxes]
            junkaxes = [i for i in range(newdims)] + [i + newdims for i in junkaxes]
            oldsize = tuple(bcastsize)
        dprint(2, realaxes, junkaxes, dist)
        newmap = [junkaxes.pop(0) if v == 1 else realaxes.pop(0) for v in newshape]
        sz, dist = shardview.remap_axis(oldsize, dist, newmap)
        dprint(2, sz, dist)
        return ndarray(
            sz, gid=arr.gid, distribution=dist, local_border=0, readonly=arr.readonly
        )
    # general reshape
    global reshape_forwarding
    if reshape_forwarding:
        return reshape_copy(arr, newshape)
    else:
        raise ReshapeError(
            "ramba.reshape not supported as distributed array reshape cannot be done inplace.  Use reshape_copy instead to create a non-inplace reshape or set RAMBA_RESHAPE_COPY environment variable to convert all reshape calls to reshape_copy."
        )
"""

def reshape_copy_executor(temp_array, arr, newshape):
    assert np.prod(arr.shape) == np.prod(newshape)
    new_arr = empty(newshape, dtype=arr.dtype, no_defer=True)

    dprint(
        2,
        "reshape_copy",
        arr.shape,
        temp_array.bdarray.gid,
        arr.bdarray.gid,
        new_arr.bdarray.gid,
        newshape,
        shardview.distribution_to_divisions(arr.distribution),
        shardview.distribution_to_divisions(new_arr.distribution),
    )
    reshape_workers = remote_call_all(
        "reshape",
        new_arr.gid,
        new_arr.shape,
        new_arr.distribution,
        arr.gid,
        arr.shape,
        arr.distribution,
        uuid.uuid4(),
    )
    return new_arr


@DAGapi
def reshape_copy(arr, newshape):
    assert np.prod(arr.shape) == np.prod(newshape)
    return DAGshape(newshape, arr.dtype, False)


"""
def reshape_copy(arr, newshape):
    assert np.prod(arr.shape) == np.prod(newshape)
    new_arr = empty(newshape, dtype=arr.dtype)

    deferred_op.do_ops()
    dprint(
        2,
        "reshape_copy",
        arr.shape,
        newshape,
        shardview.distribution_to_divisions(arr.distribution),
        shardview.distribution_to_divisions(new_arr.distribution),
    )
    reshape_workers = remote_call_all(
        "reshape",
        new_arr.gid,
        new_arr.shape,
        new_arr.distribution,
        arr.gid,
        arr.shape,
        arr.distribution,
        uuid.uuid4(),
    )
    return new_arr
"""

def mgrid_gen_getitem_executor(temp_array, index):
    dim_sizes = temp_array.shape
    res_dist = shardview.default_distribution(dim_sizes, dims_do_not_distribute=[0])
    res = empty(dim_sizes, dtype=np.int64, distribution=res_dist, no_defer=True)
    reshape_workers = remote_call_all("mgrid", res.gid, res.shape, res.distribution)
    return res


@DAGapi
def mgrid_gen_getitem(index):
    num_dim = len(index)
    dim_sizes = [num_dim]
    for i in range(num_dim):
        if isinstance(index[i], int):
            dim_sizes.append(index[i])
        elif isinstance(index[i], slice):
            assert index[i].step is None
            dim_sizes.append(index[i].stop - index[i].start)
    dim_sizes = tuple(dim_sizes)
    return DAGshape(dim_sizes, np.int64, False)


class MgridGen:
    def __getitem__(self, index):
        return mgrid_gen_getitem(index)

    """
    def __getitem__(self, index):
        num_dim = len(index)
        dim_sizes = [num_dim]
        for i in range(num_dim):
            if isinstance(index[i], int):
                dim_sizes.append(index[i])
            elif isinstance(index[i], slice):
                assert index[i].step is None
                dim_sizes.append(index[i].stop - index[i].start)
        dim_sizes = tuple(dim_sizes)
        dprint(2, "MgridGen:", index)
        res_dist = shardview.default_distribution(dim_sizes, dims_do_not_distribute=[0])
        res = empty(dim_sizes, dtype=np.int64, distribution=res_dist, no_defer=True)
        reshape_workers = remote_call_all("mgrid", res.gid, res.shape, res.distribution)
        return res
    """


mgrid = MgridGen()


def linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None):
    assert num > 0
    length = stop - start
    if endpoint:
        step = length / (num - 1)
    else:
        step = length / num

    def impl(bcontainer, dim_lens, starts):
        local_start = start + (starts[0] * step)
        local_stop = start + ((starts[0] + dim_lens[0]) * step)
        bcontainer[:] = np.linspace(
            local_start, local_stop, num=dim_lens[0], endpoint=False, dtype=dtype
        )

    res = init_array(
        num,
        Filler(impl, mode=Filler.WHOLE_ARRAY_INPLACE, do_compile=False),
        dtype=dtype,
    )

    if retstep:
        return (res, step)
    else:
        return res


def triu_executor(temp_array, m, k=0):
    assert len(m.shape) == 2
    new_ndarray = create_array_with_divisions(
        m.shape, m.distribution, local_border=m.local_border, dtype=m.dtype
    )
    remote_exec_all(
        "triu",
        new_ndarray.gid,
        m.gid,
        k,
        new_ndarray.local_border,
        new_ndarray.from_border[i] if new_ndarray.from_border is not None else None,
        new_ndarray.to_border[i] if new_ndarray.to_border is not None else None,
    )
    return new_ndarray


@DAGapi
def triu(m, k=0):
    assert len(m.shape) == 2
    return DAGshape(m.shape, m.dtype, False)


"""
def triu(m, k=0):
    assert len(m.shape) == 2
    deferred_op.do_ops()
    new_ndarray = create_array_with_divisions(
        m.shape, m.distribution, local_border=m.local_border
    )
    remote_exec_all(
        "triu",
        new_ndarray.gid,
        m.gid,
        k,
        new_ndarray.local_border,
        new_ndarray.from_border[i] if new_ndarray.from_border is not None else None,
        new_ndarray.to_border[i] if new_ndarray.to_border is not None else None,
    )
    return new_ndarray
"""


##### Skeletons ######


def smap_internal_executor(temp_array, func, attr, *args, dtype=None, parallel=True):
    # TODO: should see if this can be converted into a deferred op
    partitioned = list(filter(lambda x: isinstance(x, ndarray), args))
    assert len(partitioned) > 0
    shape = partitioned[0].shape
    dist = partitioned[0].distribution
    for arg in partitioned:
        assert shardview.dist_is_eq(arg.distribution, dist)

    if dtype is None:
        dtype = partitioned[0].dtype
    new_ndarray = create_array_with_divisions(
        shape, dist, partitioned[0].local_border, dtype=dtype
    )
    args_to_remote = [x.gid if isinstance(x, ndarray) else x for x in args]
    # [getattr(remote_states[i], attr).remote(new_ndarray.gid, partitioned[0].gid, args_to_remote, func) for i in range(num_workers)]
    remote_exec_all(
        attr, new_ndarray.gid, partitioned[0].gid, args_to_remote, func_dumps(func), dtype, parallel
    )
    new_ndarray.bdarray.remote_constructed = True  # set remote_constructed = True
    dprint(2, "smap_internal done")
    return new_ndarray


@DAGapi
def smap_internal(func, attr, *args, dtype=None, parallel=True):
    # TODO: should see if this can be converted into a deferred op
    partitioned = list(filter(lambda x: isinstance(x, ndarray), args))
    assert len(partitioned) > 0
    shape = partitioned[0].shape
    if dtype is None:
        dtype = partitioned[0].dtype
    return DAGshape(shape, dtype, False)


"""
def smap_internal(func, attr, *args, dtype=None, parallel=True):
    # TODO: should see if this can be converted into a deferred op
    deferred_op.do_ops()
    partitioned = list(filter(lambda x: isinstance(x, ndarray), args))
    assert len(partitioned) > 0
    shape = partitioned[0].shape
    dist = partitioned[0].distribution
    for arg in partitioned:
        DAG.instantiate(arg)
        assert shardview.dist_is_eq(arg.distribution, dist)

    if dtype is None:
        dtype = partitioned[0].dtype
    new_ndarray = create_array_with_divisions(
        shape, dist, partitioned[0].local_border, dtype=dtype
    )
    args_to_remote = [x.gid if isinstance(x, ndarray) else x for x in args]
    # [getattr(remote_states[i], attr).remote(new_ndarray.gid, partitioned[0].gid, args_to_remote, func) for i in range(num_workers)]
    remote_exec_all(
        attr, new_ndarray.gid, partitioned[0].gid, args_to_remote, func_dumps(func), dtype, parallel
    )
    new_ndarray.bdarray.remote_constructed = True  # set remote_constructed = True
    dprint(2, "smap_internal done")
    return new_ndarray
"""


def smap(func, *args, dtype=None, parallel=True):
    return smap_internal(func, "smap", *args, dtype=dtype, parallel=parallel)


def smap_index(func, *args, dtype=None, parallel=True):
    return smap_internal(func, "smap_index", *args, dtype=dtype, parallel=parallel)


class SreduceReducer:
    def __init__(self, worker_func, driver_func):
        self.worker_func = worker_func
        self.driver_func = driver_func


def sreduce_internal(func, reducer, identity, attr, *args, parallel=True):
    deferred_op.do_ops()
    start_time = timer()
    partitioned = list(filter(lambda x: isinstance(x, ndarray), args))
    assert len(partitioned) > 0
    shape = partitioned[0].shape
    for arg in partitioned:
        DAG.instantiate(arg)
        assert arg.shape == shape
    if not isinstance(reducer, SreduceReducer):
        reducer = SreduceReducer(reducer, reducer)

    a_send_recv = uuid.uuid4()
    args_to_remote = [x.gid if isinstance(x, ndarray) else x for x in args]
    worker_results = remote_call_all(
        attr,
        partitioned[0].gid,
        args_to_remote,
        func_dumps(func),
        func_dumps(reducer.worker_func),
        func_dumps(reducer.driver_func),
        identity,
        a_send_recv,
        parallel
    )
    after_remote_time = timer()
    final_result = worker_results[0]
    #dprint(2, "sreduce first result:", final_result[0].shape, final_result[0].shape * final_result[0].itemsize, final_result[0].shape)
    """
    for result in worker_results[1:]:
        final_result = reducer.driver_func(final_result, result, *args)
    """
    tprint(2, "sreduce remote time:", after_remote_time - start_time)
    tprint(2, "sreduce local reduction time:", timer() - after_remote_time)
    return final_result


def sreduce(func, reducer, identity, *args, parallel=True):
    return sreduce_internal(func, reducer, identity, "sreduce", *args, parallel=parallel)


def sreduce_index(func, reducer, identity, *args, parallel=True):
    return sreduce_internal(func, reducer, identity, "sreduce_index", *args, parallel=parallel)


def sstencil(func, *args, **kwargs):
    sstencil_start = timer()
    DAG.execute_all()
    if func.neighborhood is None:
        fake_args = [
            np.empty(tuple([1] * len(x.shape))) if isinstance(x, ndarray) else x
            for x in args
        ]
        slocal = func.compile_local()
        try:
            slocal(*fake_args)
        except Exception:
            pass
        func.neighborhood = slocal.neighborhood
    neighborhood = func.neighborhood
    dprint(2, "sstencil driver:", neighborhood)
    nmax = 0
    for n in neighborhood:
        nmax = max(nmax, abs(n[0]))
        nmax = max(nmax, abs(n[1]))

    partitioned = list(filter(lambda x: isinstance(x, ndarray), args))
    assert len(partitioned) > 0
    shape = partitioned[0].shape
    border = partitioned[0].local_border
    for arg in partitioned:
        assert arg.shape == shape
        assert arg.local_border == border
    assert border >= nmax

    stencil_op_uuid = uuid.uuid4()
    if "out" in kwargs:
        new_ndarray = kwargs["out"]
        create_flag = False
        # TODO:  Check shapes
    else:
        new_ndarray = create_array_with_divisions(
            shape, partitioned[0].distribution, partitioned[0].local_border
        )
        create_flag = True
    args_to_remote = [x.gid if isinstance(x, ndarray) else x for x in args]
    # [remote_states[i].sstencil.remote(stencil_op_uuid, new_ndarray.gid, neighborhood, partitioned[0].gid, args_to_remote, func, create_flag) for i in range(num_workers)]
    sstencil_before_remote = timer()
    stencil_workers = remote_async_call_all(
        "sstencil",
        stencil_op_uuid,
        new_ndarray.gid,
        neighborhood,
        partitioned[0].gid,
        args_to_remote,
        func_dumps(func),
        create_flag,
    )
    sstencil_after_remote = timer()
    worker_timings = get_results(stencil_workers)
    sstencil_end = timer()

    add_time("sstencil", sstencil_end - sstencil_start)
    add_sub_time("sstencil", "before_remote", sstencil_before_remote - sstencil_start)
    add_sub_time("sstencil", "remote_call", sstencil_after_remote - sstencil_before_remote)
    add_sub_time("sstencil", "get_results", sstencil_end - sstencil_after_remote)
    add_sub_time("sstencil", "max_remote_prep", max([x[0] for x in worker_timings]))
    add_sub_time("sstencil", "max_remote_compile", max([x[1] for x in worker_timings]))
    add_sub_time("sstencil", "max_remote_exec", max([x[2] for x in worker_timings]))
    add_sub_time("sstencil", "max_remote_total", max([x[3] for x in worker_timings]))

    new_ndarray.bdarray.remote_constructed = True  # set remote_constructed = True
    return new_ndarray


def scumulative(local_func, final_func, array):
    deferred_op.do_ops()
    assert isinstance(array, ndarray)
    shape = array.shape
    assert len(shape) == 1

    new_ndarray = create_array_with_divisions(
        shape, array.distribution, array.local_border
    )
    # Can we do this without ray.get?
    # ray.get([remote_states[i].scumulative_local.remote(new_ndarray.gid, array.gid, local_func) for i in range(num_workers)])
    remote_exec_all(
        "scumulative_local", new_ndarray.gid, array.gid, func_dumps(local_func)
    )
    end_index = (
        shardview.get_start(array.distribution[0])[0]
        + shardview.get_size(array.distribution[0])[0]
        - 1
    )
    slice_to_get = (slice(end_index, end_index + 1),)
    dprint(3, "slice_to_get:", slice_to_get)
    # boundary_val = ray.get(remote_states[0].get_partial_array.remote(new_ndarray.gid, slice_to_get))
    boundary_val = remote_call(0, "get_partial_array", new_ndarray.gid, slice_to_get)
    dprint(3, "boundary_val:", boundary_val, type(boundary_val))
    for i in range(1, num_workers):
        # TODO: combine these into one?
        # ray.get(remote_states[i].scumulative_final.remote(new_ndarray.gid, boundary_val, final_func))
        remote_exec(
            i,
            "scumulative_final",
            new_ndarray.gid,
            boundary_val,
            func_dumps(final_func),
        )
        local_element = shardview.get_size(array.distribution[i])[0] - 1
        slice_to_get = (slice(local_element, local_element + 1),)
        # boundary_val = ray.get(remote_states[i].get_partial_array.remote(new_ndarray.gid, slice_to_get))
        boundary_val = remote_call(
            i, "get_partial_array", new_ndarray.gid, slice_to_get
        )
    new_ndarray.bdarray.remote_constructed = True  # set remote_constructed = True
    return new_ndarray


def spmd(func, *args):
    deferred_op.do_ops()
    args_to_remote = [x.gid if isinstance(x, ndarray) else x for x in args]
    # ray.get([remote_states[i].spmd.remote(func, args_to_remote) for i in range(num_workers)])
    dprint(2, "Before exec_all spmd")
    remote_exec_all("spmd", func_dumps(func), args_to_remote)
    dprint(2, "After exec_all spmd")


#-------------------------------------------------------------------------------

class mean_identity:
    def __init__(self, drop_groupdim, dtype):
        self.drop_groupdim = drop_groupdim
        self.dtype = dtype

    def __call__(self, *args, **kwargs):
        return (np.zeros(self.drop_groupdim, dtype=self.dtype), np.zeros(self.drop_groupdim, dtype=int))

class mean_sum:
    def __init__(self, drop_groupdim, dtype):
        self.drop_groupdim = drop_groupdim
        self.dtype = dtype

    def __call__(self, *args, **kwargs):
        return np.zeros(self.drop_groupdim, dtype=self.dtype)

class mean_count:
    def __init__(self, drop_groupdim):
        self.drop_groupdim = drop_groupdim

    def __call__(self, *args, **kwargs):
        return np.zeros(self.drop_groupdim, dtype=int)

#-------------------------------------------------------------------------------

class sum_identity:
    def __init__(self, drop_groupdim, dtype):
        self.drop_groupdim = drop_groupdim
        self.dtype = dtype

    def __call__(self, *args, **kwargs):
        return np.zeros(self.drop_groupdim, dtype=self.dtype)

#-------------------------------------------------------------------------------

class count_identity:
    def __init__(self, drop_groupdim):
        self.drop_groupdim = drop_groupdim

    def __call__(self, *args, **kwargs):
        return np.zeros(self.drop_groupdim, dtype=int)

#-------------------------------------------------------------------------------

class prod_identity:
    def __init__(self, drop_groupdim, dtype):
        self.drop_groupdim = drop_groupdim
        self.dtype = dtype

    def __call__(self, *args, **kwargs):
        return np.ones(self.drop_groupdim, dtype=self.dtype)

#-------------------------------------------------------------------------------

class min_identity:
    def __init__(self, drop_groupdim, dtype):
        self.drop_groupdim = drop_groupdim
        self.dtype = dtype

    def __call__(self, *args, **kwargs):
        if np.issubdtype(self.dtype, np.integer):
            max_val = np.iinfo(self.dtype).max
        elif np.issubdtype(self.dtype, np.floating):
            max_val = np.inf
        else:
            assert 0
        return np.full(self.drop_groupdim, max_val, dtype=self.dtype)

#-------------------------------------------------------------------------------

class max_identity:
    def __init__(self, drop_groupdim, dtype):
        self.drop_groupdim = drop_groupdim
        self.dtype = dtype

    def __call__(self, *args, **kwargs):
        if np.issubdtype(self.dtype, np.integer):
            min_val = np.iinfo(self.dtype).min
        elif np.issubdtype(self.dtype, np.floating):
            min_val = -np.inf
        else:
            assert 0
        return np.full(self.drop_groupdim, min_val, dtype=self.dtype)

#-------------------------------------------------------------------------------

# aggregations done: mean, sum, count, prod, min, max, var, std
# aggregations to do: argmin, argmax, first, last, all, any

class RambaGroupby:
    def __init__(self, array_to_group, dim, group_array, num_groups=None):
        self.array_to_group = array_to_group
        self.dim = dim
        self.group_array = group_array
        self.num_groups = num_groups
        # performance warning that ndarray partitioned only on groupby dimension.
        if isinstance(self.group_array, ndarray):
            self.np_group = asarray(self.group_array)
        else:
            self.np_group = self.group_array
        assert(isinstance(self.np_group, np.ndarray))

    def mean(self, dim=None, ret_separate=False):
        assert(self.num_groups)  # we don't handle the case where they didn't specify the number of groups yet
        tprint(2, "Starting groupby mean.")
        mean_start = timer()
        # original dimensions minus groupby dimension with num-groups added to front
        orig_dims = self.array_to_group.shape
        drop_groupdim = list(orig_dims)
        drop_groupdim[self.dim] = self.num_groups
        drop_groupdim = tuple(drop_groupdim)

        mean_func_txt  =  "def mean_func(idx, value, group_array, sumout, countout):\n"
        beforedim = ",".join([f"idx[{nd}]" for nd in range(self.dim)])
        afterdim = ",".join([f"idx[{nd}]" for nd in range(self.dim+1, self.array_to_group.ndim)])
        mean_func_txt += f"    groupid = ({beforedim}, group_array[idx[{self.dim}]],{afterdim})\n"
        mean_func_txt +=  "    sumout[groupid] += value\n"
        mean_func_txt +=  "    countout[groupid] += 1\n"
        mean_func_txt +=  "    return (sumout, countout)\n"
        ldict = {}
        gdict = globals()
        exec(mean_func_txt, gdict, ldict)

        def mean_reducer_driver(result, fres):
            return (result[0] + fres[0], result[1] + fres[1])

        def mean_reducer_worker(result, fres):
            return fres

        res = sreduce_index(ldict["mean_func"], SreduceReducer(mean_reducer_worker, mean_reducer_driver), mean_identity(drop_groupdim, self.array_to_group.dtype), self.array_to_group, self.np_group, mean_sum(drop_groupdim, self.array_to_group.dtype), mean_count(drop_groupdim), parallel=False)
        tprint(2, "groupby mean time:", timer() - mean_start)
        #with np.printoptions(threshold=np.inf):
        #    print("sum last:", res[0])
        #    print("count last:", res[1])
        if ret_separate:
            return res[0], res[1]
        else:
            return res[0] / res[1]

    def nanmean(self, dim=None, ret_separate=False):
        assert(self.num_groups)  # we don't handle the case where they didn't specify the number of groups yet
        tprint(2, "Starting groupby nammean.")
        mean_start = timer()
        # original dimensions minus groupby dimension with num-groups added to front
        orig_dims = self.array_to_group.shape
        drop_groupdim = list(orig_dims)
        drop_groupdim[self.dim] = self.num_groups
        drop_groupdim = tuple(drop_groupdim)

        mean_func_txt  =  "def mean_func(idx, value, group_array, sumout, countout):\n"
        beforedim = ",".join([f"idx[{nd}]" for nd in range(self.dim)])
        afterdim = ",".join([f"idx[{nd}]" for nd in range(self.dim+1, self.array_to_group.ndim)])
        mean_func_txt +=  "    if value != np.nan:\n"
        mean_func_txt += f"        groupid = ({beforedim}, group_array[idx[{self.dim}]],{afterdim})\n"
        mean_func_txt +=  "        sumout[groupid] += value\n"
        mean_func_txt +=  "        countout[groupid] += 1\n"
        mean_func_txt +=  "    return (sumout, countout)\n"
        ldict = {}
        gdict = globals()
        exec(mean_func_txt, gdict, ldict)

        def mean_reducer_driver(result, fres):
            return (result[0] + fres[0], result[1] + fres[1])

        def mean_reducer_worker(result, fres):
            return fres

        #res = sreduce_index(ldict["mean_func"], SreduceReducer(lambda x, y: y, lambda x, y: (x[0] + y[0], x[1] + y[1])), mean_identity(drop_groupdim, self.array_to_group.dtype), self.array_to_group, self.np_group, mean_sum(drop_groupdim, self.array_to_group.dtype), mean_count(drop_groupdim))
        res = sreduce_index(ldict["mean_func"], SreduceReducer(mean_reducer_worker, mean_reducer_driver), mean_identity(drop_groupdim, self.array_to_group.dtype), self.array_to_group, self.np_group, mean_sum(drop_groupdim, self.array_to_group.dtype), mean_count(drop_groupdim), parallel=False)
        tprint(2, "groupby mean time:", timer() - mean_start)
        #with np.printoptions(threshold=np.inf):
        #    print("sum last:", res[0])
        #    print("count last:", res[1])
        if ret_separate:
            return res[0], res[1]
        else:
            return res[0] / res[1]

    def sum(self, dim=None):
        assert(self.num_groups)  # we don't handle the case where they didn't specify the number of groups yet
        tprint(2, "Starting groupby sum.")
        sum_start = timer()
        # original dimensions minus groupby dimension with num-groups added to front
        orig_dims = self.array_to_group.shape
        drop_groupdim = list(orig_dims)
        drop_groupdim[self.dim] = self.num_groups
        drop_groupdim = tuple(drop_groupdim)

        sum_func_txt  =  "def sum_func(idx, value, group_array, sumout):\n"
        beforedim = ",".join([f"idx[{nd}]" for nd in range(self.dim)])
        afterdim = ",".join([f"idx[{nd}]" for nd in range(self.dim+1, self.array_to_group.ndim)])
        sum_func_txt += f"    groupid = ({beforedim}, group_array[idx[{self.dim}]],{afterdim})\n"
        sum_func_txt +=  "    sumout[groupid] += value\n"
        sum_func_txt +=  "    return sumout\n"
        ldict = {}
        gdict = globals()
        exec(sum_func_txt, gdict, ldict)

        def sum_reducer_driver(result, fres):
            return result + fres

        def sum_reducer_worker(result, fres):
            return fres

        res = sreduce_index(ldict["sum_func"], SreduceReducer(sum_reducer_worker, sum_reducer_driver), sum_identity(drop_groupdim, self.array_to_group.dtype), self.array_to_group, self.np_group, sum_identity(drop_groupdim, self.array_to_group.dtype), parallel=False)
        tprint(2, "groupby sum time:", timer() - sum_start)
        #with np.printoptions(threshold=np.inf):
        #    print("sum last:", res)
        return res

    def count(self, dim=None):
        assert(self.num_groups)  # we don't handle the case where they didn't specify the number of groups yet
        tprint(2, "Starting groupby count.")
        count_start = timer()
        # original dimensions minus groupby dimension with num-groups added to front
        orig_dims = self.array_to_group.shape
        drop_groupdim = list(orig_dims)
        drop_groupdim[self.dim] = self.num_groups
        drop_groupdim = tuple(drop_groupdim)

        count_func_txt  =  "def count_func(idx, value, group_array, countout):\n"
        beforedim = ",".join([f"idx[{nd}]" for nd in range(self.dim)])
        afterdim = ",".join([f"idx[{nd}]" for nd in range(self.dim+1, self.array_to_group.ndim)])
        count_func_txt += f"    groupid = ({beforedim}, group_array[idx[{self.dim}]],{afterdim})\n"
        count_func_txt +=  "    countout[groupid] += 1\n"
        count_func_txt +=  "    return countout\n"
        ldict = {}
        gdict = globals()
        exec(count_func_txt, gdict, ldict)

        def count_reducer_driver(result, fres):
            return result + fres

        def count_reducer_worker(result, fres):
            return fres

        res = sreduce_index(ldict["count_func"], SreduceReducer(count_reducer_worker, count_reducer_driver), count_identity(drop_groupdim), self.array_to_group, self.np_group, count_identity(drop_groupdim), parallel=False)
        tprint(2, "groupby count time:", timer() - count_start)
        #with np.printoptions(threshold=np.inf):
        #    print("count last:", res)
        return res

    def prod(self, dim=None):
        assert(self.num_groups)  # we don't handle the case where they didn't specify the number of groups yet
        tprint(2, "Starting groupby prod.")
        prod_start = timer()
        # original dimensions minus groupby dimension with num-groups added to front
        orig_dims = self.array_to_group.shape
        drop_groupdim = list(orig_dims)
        drop_groupdim[self.dim] = self.num_groups
        drop_groupdim = tuple(drop_groupdim)

        prod_func_txt  =  "def prod_func(idx, value, group_array, prodout):\n"
        beforedim = ",".join([f"idx[{nd}]" for nd in range(self.dim)])
        afterdim = ",".join([f"idx[{nd}]" for nd in range(self.dim+1, self.array_to_group.ndim)])
        prod_func_txt += f"    groupid = ({beforedim}, group_array[idx[{self.dim}]],{afterdim})\n"
        prod_func_txt +=  "    prodout[groupid] *= value\n"
        prod_func_txt +=  "    return prodout\n"
        ldict = {}
        gdict = globals()
        exec(prod_func_txt, gdict, ldict)

        def prod_reducer_driver(result, fres):
            return result * fres

        def prod_reducer_worker(result, fres):
            return fres

        res = sreduce_index(ldict["prod_func"], SreduceReducer(prod_reducer_worker, prod_reducer_driver), prod_identity(drop_groupdim, self.array_to_group.dtype), self.array_to_group, self.np_group, prod_identity(drop_groupdim, self.array_to_group.dtype), parallel=False)
        tprint(2, "groupby prod time:", timer() - prod_start)
        #with np.printoptions(threshold=np.inf):
        #    print("prod last:", res)
        return res

    def min(self, dim=None):
        assert(self.num_groups)  # we don't handle the case where they didn't specify the number of groups yet
        tprint(2, "Starting groupby min.")
        min_start = timer()
        # original dimensions minus groupby dimension with num-groups added to front
        orig_dims = self.array_to_group.shape
        drop_groupdim = list(orig_dims)
        drop_groupdim[self.dim] = self.num_groups
        drop_groupdim = tuple(drop_groupdim)

        min_func_txt  =  "def min_func(idx, value, group_array, minout):\n"
        beforedim = ",".join([f"idx[{nd}]" for nd in range(self.dim)])
        afterdim = ",".join([f"idx[{nd}]" for nd in range(self.dim+1, self.array_to_group.ndim)])
        min_func_txt += f"    groupid = ({beforedim}, group_array[idx[{self.dim}]],{afterdim})\n"
        min_func_txt +=  "    minout[groupid] = min(value, minout[groupid])\n"
        min_func_txt +=  "    return minout\n"
        ldict = {}
        gdict = globals()
        exec(min_func_txt, gdict, ldict)

        def min_reducer_driver(result, fres):
            return np.minimum(result, fres)

        def min_reducer_worker(result, fres):
            return fres

        res = sreduce_index(ldict["min_func"], SreduceReducer(min_reducer_worker, min_reducer_driver), min_identity(drop_groupdim, self.array_to_group.dtype), self.array_to_group, self.np_group, min_identity(drop_groupdim, self.array_to_group.dtype), parallel=False)
        tprint(2, "groupby min time:", timer() - min_start)
        #with np.printoptions(threshold=np.inf):
        #    print("min last:", res)
        return res

    def max(self, dim=None):
        assert(self.num_groups)  # we don't handle the case where they didn't specify the number of groups yet
        tprint(2, "Starting groupby max.")
        max_start = timer()
        # original dimensions maxus groupby dimension with num-groups added to front
        orig_dims = self.array_to_group.shape
        drop_groupdim = list(orig_dims)
        drop_groupdim[self.dim] = self.num_groups
        drop_groupdim = tuple(drop_groupdim)

        max_func_txt  =  "def max_func(idx, value, group_array, maxout):\n"
        beforedim = ",".join([f"idx[{nd}]" for nd in range(self.dim)])
        afterdim = ",".join([f"idx[{nd}]" for nd in range(self.dim+1, self.array_to_group.ndim)])
        max_func_txt += f"    groupid = ({beforedim}, group_array[idx[{self.dim}]],{afterdim})\n"
        max_func_txt +=  "    maxout[groupid] = max(value, maxout[groupid])\n"
        max_func_txt +=  "    return maxout\n"
        ldict = {}
        gdict = globals()
        exec(max_func_txt, gdict, ldict)

        def max_reducer_driver(result, fres):
            return np.maximum(result, fres)

        def max_reducer_worker(result, fres):
            return fres

        res = sreduce_index(ldict["max_func"], SreduceReducer(max_reducer_worker, max_reducer_driver), max_identity(drop_groupdim, self.array_to_group.dtype), self.array_to_group, self.np_group, max_identity(drop_groupdim, self.array_to_group.dtype), parallel=False)
        tprint(2, "groupby max time:", timer() - max_start)
        #with np.printoptions(threshold=np.inf):
        #    print("max last:", res)
        return res

    def var(self, dim=None):
        assert(self.num_groups)  # we don't handle the case where they didn't specify the number of groups yet
        tprint(2, "Starting groupby var.")
        var_start = timer()
        # original dimensions minus groupby dimension with num-groups added to front
        orig_dims = self.array_to_group.shape
        drop_groupdim = list(orig_dims)
        drop_groupdim[self.dim] = self.num_groups
        drop_groupdim = tuple(drop_groupdim)

        mean_groupby_sum, mean_groupby_count = self.mean(dim=dim, ret_separate=True)
        mean_groupby = mean_groupby_sum / mean_groupby_count

        sqr_mean_diff_func_txt  =  "def sqr_mean_diff_func(idx, value, group_array, sumout, mean_groupby):\n"
        beforedim = ",".join([f"idx[{nd}]" for nd in range(self.dim)])
        afterdim = ",".join([f"idx[{nd}]" for nd in range(self.dim+1, self.array_to_group.ndim)])
        sqr_mean_diff_func_txt += f"    groupid = ({beforedim}, group_array[idx[{self.dim}]],{afterdim})\n"
        sqr_mean_diff_func_txt +=  "    sumout[groupid] += (value - mean_groupby[groupid])**2\n"
#        sqr_mean_diff_func_txt +=  "    print(idx, value, mean_groupby[groupid], sumout[groupid])\n"
        sqr_mean_diff_func_txt +=  "    return sumout\n"
        ldict = {}
        gdict = globals()
        exec(sqr_mean_diff_func_txt, gdict, ldict)

        def sum_reducer_driver(result, fres):
            return result + fres

        def sum_reducer_worker(result, fres):
            return fres

        res = sreduce_index(ldict["sqr_mean_diff_func"], SreduceReducer(sum_reducer_worker, sum_reducer_driver), sum_identity(drop_groupdim, mean_groupby.dtype), self.array_to_group, self.np_group, sum_identity(drop_groupdim, mean_groupby.dtype), mean_groupby, parallel=False)
        res_var = res / mean_groupby_count
        tprint(2, "groupby var time:", timer() - var_start)
        #with np.printoptions(threshold=np.inf):
        #    print("sum last:", res[0])
        #    print("count last:", res[1])
        return res_var

    def std(self, dim=None):
        return np.sqrt(self.var(dim=dim))


def groupby_attr(item, itxt, imports, dtype):
    func_txt =  f"def gba{item}(self, rhs):\n"
    if ntiming:
        func_txt +=  "    start_time = timer()\n"
    func_txt += f"    gtext =  \"def group{item}(idx, value, rhs, groupid, dim):\\n\"\n"
    func_txt +=  "    beforedim = \",\".join([f\"idx[{nd}]\" for nd in range(self.dim)])\n"
    func_txt +=  "    afterdim = \",\".join([f\"idx[{nd}]\" for nd in range(self.dim+1, self.array_to_group.ndim)])\n"
    func_txt +=  "    gtext += f\"    drop_groupdim = ({beforedim}, groupid[idx[dim]],{afterdim})\\n\"\n"
#    func_txt +=  "    gtext +=  \"    drop_groupdim = (groupid[idx[dim]],\" + \",\".join([f\"idx[{nd}]\" for nd in range(self.array_to_group.ndim) if nd != self.dim]) + \")\\n\"\n"
#    func_txt +=  "    gtext += \"    print('group:', idx, value, drop_groupdim, rhs[drop_groupdim], dim)\\n\"\n"
    func_txt += f"    gtext += \"    return value{itxt}rhs[drop_groupdim]\\n\"\n"
    func_txt +=  "    ldict = {}\n"
    func_txt +=  "    gdict = globals()\n"
#    func_txt +=  "    print(\"running gtext\", gtext)\n"
    func_txt +=  "    exec(gtext, gdict, ldict)\n"
    func_txt +=  "    new_dtype = unify_args(self.array_to_group.dtype, rhs, None)\n"
    func_txt += f"    res = smap_index(ldict[\"group{item}\"], self.array_to_group, rhs, self.np_group, self.dim, dtype=new_dtype)\n"
    if ntiming:
        func_txt +=  "    print(\"gba time:\", timer() - start_time)\n"
    func_txt +=  "    return res\n"
    ldict = {}
    gdict = globals()
    exec(func_txt, gdict, ldict)
    return ldict[f"gba{item}"]


# Add array binop style support to groupby's.
for (abf, code) in array_binop_funcs.items():
    new_func = groupby_attr(abf, code.code, imports=code.imports, dtype=code.dtype)
    setattr(RambaGroupby, abf, new_func)
