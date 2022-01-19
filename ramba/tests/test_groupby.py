import ramba
import numpy as np
import random
import numba
import pytest

xarray = pytest.importorskip("xarray")
pd = pytest.importorskip("pandas")

class TestGroupby:
    def test_mean_groupby1(self):
        def ramba_impl(x):
            rlin = ramba.fromarray(x)
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("1/1/2000", "31/12/2004", freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=366)
            gbmean = gb.mean()
            final = gb - gbmean
            return final.asarray()

        def xarray_numpy_impl(rlin):
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("1/1/2000", "31/12/2004", freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbmean = gb.mean("time")
            final = gb - gbmean
            return final.data

        size = (2, 1827)
        rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        assert np.allclose(xnres, rres)

    def test_mean_groupby2(self):
        def ramba_impl(x):
            rlin = ramba.fromarray(x)
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("1/1/2000", "31/12/2004", freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([(pd.Timestamp(x).month % 12) // 3 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=4)
            gbmean = gb.mean()
            final = gb - gbmean
            return final.asarray()

        def xarray_numpy_impl(rlin):
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("1/1/2000", "31/12/2004", freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.season")
            gbmean = gb.mean("time")
            final = gb - gbmean
            return final.data

        size = (2, 1827)
        rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        assert np.allclose(xnres, rres)
