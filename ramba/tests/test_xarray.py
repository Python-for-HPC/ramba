import ramba
import numpy as np
import random
import numba
import pytest
import math

xarray = pytest.importorskip("xarray")


def rb_comparer(np_res, ramba_res, comp):
    if isinstance(np_res, tuple):
        assert isinstance(ramba_res, tuple)
        assert len(np_res) == len(ramba_res)
        for i in range(len(np_res)):
            rb_comparer(np_res[i], ramba_res[i], comp)
    elif isinstance(np_res, np.ndarray):
        assert isinstance(ramba_res, ramba.ndarray)
        ramba_local = ramba_res.asarray()
        assert comp(np_res, ramba_local)
    elif isinstance(np_res, float):
        assert isinstance(ramba_res, float)
        return comp (np_res, ramba_res)
    else:
        assert np_res == ramba_res


def run_both(func, *args, comp):
    ramba_res = func(ramba, *args)
    ramba.sync()
    np_res = func(np, *args)
    rb_comparer(np_res, ramba_res, comp)


class TestXarray:
    def test1(self):
        def impl(app):
            ra1 = app.fromfunction(lambda x, y: x + y, (10, 20))
            xa1 = xarray.DataArray(ra1)
            xa2 = xa1 + 10.0
            xa22 = xa2 * 7.1
            xa3 = np.sin(xa22)
            xa4 = xa3.transpose()
            xa5 = xa4.sum()
            return xa5.data.item()

        run_both(impl, comp=math.isclose)
