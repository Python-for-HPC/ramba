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
                #coords={"time":pd.date_range("2000-1-1", "2000-12-31", freq="D")},
                #coords={"time":pd.date_range("2000-1-1", "2001-12-31", freq="D")},
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=366)
            gbmean = gb.mean()
            #print("gbmean:", gbmean.dtype, gbmean)
            final = gb - gbmean
            #print("final:", final.dtype)
            return final.asarray()

        def xarray_numpy_impl(rlin):
            da = xarray.DataArray(
                rlin,
                #coords={"time":pd.date_range("2000-1-1", "2000-12-31", freq="D")},
                #coords={"time":pd.date_range("2000-1-1", "2001-12-31", freq="D")},
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbmean = gb.mean("time")
            #print("xarray gbmean:", gbmean)
            final = gb - gbmean
            return final.data

        #size = (1, 366)
        #size = (1, 731)
        size = (1, 1827)
        rlin = np.arange(size[0] * size[1]).reshape(size)
        #rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        #with np.printoptions(threshold=np.inf):
        #    #print("xnres:", xnres.dtype)
        #    #print("rres:", rres.dtype)
        #    print("xnres:", xnres.dtype, xnres)
        #    print("rres:", rres.dtype, rres)
        assert np.allclose(xnres, rres)

    def test_mean_groupby2(self):
        def ramba_impl(x):
            rlin = ramba.fromarray(x)
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
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
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.season")
            gbmean = gb.mean("time")
            final = gb - gbmean
            return final.data

        size = (2, 1827)
        #rlin = np.random.random(size)
        rlin = np.arange(size[0] * size[1]).reshape(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        assert np.allclose(xnres, rres)

    def test_mean_groupby3(self):
        def ramba_impl(x):
            rlin = ramba.fromarray(x)
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
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
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbmean = gb.mean("time")
            final = gb - gbmean
            return final.data

        size = (2, 1827)
        np.random.seed(12345)
        rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        #with np.printoptions(threshold=np.inf):
        #    #print("xnres:", xnres.dtype)
        #    #print("rres:", rres.dtype)
        #    print("xnres:", xnres.dtype, xnres)
        #    print("rres:", rres.dtype, rres)
        assert np.allclose(xnres, rres)

    def test_sum_groupby1(self):
        def ramba_impl(x):
            rlin = ramba.fromarray(x)
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=366)
            gbsum = gb.sum()
            final = gb - gbsum
            return final.asarray()

        def xarray_numpy_impl(rlin):
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbsum = gb.sum("time")
            final = gb - gbsum
            return final.data

        size = (2, 1827)
        rlin = np.arange(size[0] * size[1]).reshape(size)
        #rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        #with np.printoptions(threshold=np.inf):
        #    #print("xnres:", xnres.dtype)
        #    #print("rres:", rres.dtype)
        #    print("xnres:", xnres.dtype, xnres)
        #    print("rres:", rres.dtype, rres)
        assert np.allclose(xnres, rres)

    def test_count_groupby1(self):
        def ramba_impl(x):
            rlin = ramba.fromarray(x)
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=366)
            gbcount = gb.count()
            final = gb - gbcount
            return final.asarray()

        def xarray_numpy_impl(rlin):
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbcount = gb.count("time")
            final = gb - gbcount
            return final.data

        size = (2, 1827)
        rlin = np.arange(size[0] * size[1]).reshape(size)
        #rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        #with np.printoptions(threshold=np.inf):
        #    #print("xnres:", xnres.dtype)
        #    #print("rres:", rres.dtype)
        #    print("xnres:", xnres.dtype, xnres)
        #    print("rres:", rres.dtype, rres)
        assert np.allclose(xnres, rres)

    def test_prod_groupby1(self):
        def ramba_impl(x):
            rlin = ramba.fromarray(x)
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=366)
            gbprod = gb.prod()
            final = gb - gbprod
            return final.asarray()

        def xarray_numpy_impl(rlin):
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbprod = gb.prod("time")
            final = gb - gbprod
            return final.data

        size = (2, 1827)
        rlin = np.arange(size[0] * size[1]).reshape(size)
        #rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        #with np.printoptions(threshold=np.inf):
        #    #print("xnres:", xnres.dtype)
        #    #print("rres:", rres.dtype)
        #    print("xnres:", xnres.dtype, xnres)
        #    print("rres:", rres.dtype, rres)
        assert np.allclose(xnres, rres)

    def test_min_groupby1(self):
        def ramba_impl(x):
            rlin = ramba.fromarray(x)
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=366)
            gbprod = gb.min()
            final = gb - gbprod
            return final.asarray()

        def xarray_numpy_impl(rlin):
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbprod = gb.min("time")
            final = gb - gbprod
            return final.data

        size = (1, 1827)
        rlin = np.arange(size[0] * size[1]).reshape(size)
        #rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        #with np.printoptions(threshold=np.inf):
        #    #print("xnres:", xnres.dtype)
        #    #print("rres:", rres.dtype)
        #    print("xnres:", xnres.dtype, xnres)
        #    print("rres:", rres.dtype, rres)
        assert np.allclose(xnres, rres)

    def test_max_groupby1(self):
        def ramba_impl(x):
            rlin = ramba.fromarray(x)
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=366)
            gbprod = gb.max()
            final = gb - gbprod
            return final.asarray()

        def xarray_numpy_impl(rlin):
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbprod = gb.max("time")
            final = gb - gbprod
            return final.data

        size = (2, 1827)
        rlin = np.arange(size[0] * size[1]).reshape(size)
        #rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        #with np.printoptions(threshold=np.inf):
        #    #print("xnres:", xnres.dtype)
        #    #print("rres:", rres.dtype)
        #    print("xnres:", xnres.dtype, xnres)
        #    print("rres:", rres.dtype, rres)
        assert np.allclose(xnres, rres)

    def test_var_groupby1(self):
        def ramba_impl(x):
            rlin = ramba.fromarray(x)
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=366)
            gbvar = gb.var()
            return gbvar

        def xarray_numpy_impl(rlin):
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbvar = gb.var("time")
            return gbvar.data

        size = (1, 1827)
        rlin = np.arange(size[0] * size[1]).reshape(size)
        #rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        #with np.printoptions(threshold=np.inf):
        #    #print("xnres:", xnres.dtype)
        #    #print("rres:", rres.dtype)
        #    print("xnres:", xnres.dtype, xnres.shape, xnres)
        #    print("rres:", rres.dtype, rres.shape, rres)
        assert np.allclose(xnres, rres)

    def test_std_groupby1(self):
        def ramba_impl(x):
            rlin = ramba.fromarray(x)
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=366)
            gbstd = gb.std()
            return gbstd

        def xarray_numpy_impl(rlin):
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range("2000-1-1", "2004-12-31", freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbstd = gb.std("time")
            return gbstd.data

        size = (1, 1827)
        rlin = np.arange(size[0] * size[1]).reshape(size)
        #rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        #with np.printoptions(threshold=np.inf):
        #    #print("xnres:", xnres.dtype)
        #    #print("rres:", rres.dtype)
        #    print("xnres:", xnres.dtype, xnres.shape, xnres)
        #    print("rres:", rres.dtype, rres.shape, rres)
        assert np.allclose(xnres, rres)


class TestGroupbyVariations:
    def test_mean_groupby_slice1(self):
        offset = 25
        def ramba_impl(x, slice_size):
            rlin = ramba.fromarray(x)
            rlin = rlin[:, offset:offset+slice_size]
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range(start="1/1/2001", periods=slice_size, freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=365)
            #breakpoint()
            gbmean = gb.mean()
            #print("ramba gbmean:", type(gbmean), gbmean.dtype, gbmean.shape, gbmean)
            final = gb - gbmean
            #print("final ramba:", final.dtype)
            return final.asarray()

        def xarray_numpy_impl(rlin, slice_size):
            rlin = rlin[:, offset:offset+slice_size]
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range(start="1/1/2001", periods=slice_size, freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbmean = gb.mean("time")
            #print("xarray gbmean:", type(gbmean), gbmean.dtype, gbmean.shape, gbmean)
            #return gbmean
            final = gb - gbmean
            #print("final xarray:", final.dtype)
            return final.data

        #size = (1, 366)
        #size = (1, 731)
        total_size = (1, 400)
        slice_size = 365
        rlin = np.arange(total_size[0] * total_size[1]).reshape(total_size)
        #rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin, slice_size)
        rres = ramba_impl(rlin, slice_size)
        #with np.printoptions(threshold=np.inf):
        #    #print("xnres:", xnres.dtype)
        #    #print("rres:", rres.dtype)
        #    print("xnres:", xnres.dtype, xnres)
        #    print("rres:", rres.dtype, rres)
        assert np.allclose(xnres, rres)

    def test_mean_groupby_transpose1(self):
        def ramba_impl(x):
            rlin = ramba.fromarray(x)
            rlin = rlin.T
            rshape = rlin.shape
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range(start="1/1/2001", periods=rshape[1], freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=365)
            #breakpoint()
            gbmean = gb.mean()
            #print("ramba gbmean:", type(gbmean), gbmean.dtype, gbmean.shape, gbmean)
            #return gbmean
            final = gb - gbmean
            #print("final ramba:", final.dtype)
            return final.asarray()

        def xarray_numpy_impl(rlin):
            rlin = rlin.T
            rshape = rlin.shape
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range(start="1/1/2001", periods=rshape[1], freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbmean = gb.mean("time")
            #print("xarray gbmean:", type(gbmean), gbmean.dtype, gbmean.shape, gbmean)
            #return gbmean
            final = gb - gbmean
            #print("final xarray:", final.dtype)
            return final.data

        total_size = (365, 1)
        rlin = np.arange(total_size[0] * total_size[1]).reshape(total_size)
        #rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin)
        rres = ramba_impl(rlin)
        #with np.printoptions(threshold=np.inf):
        #    #print("xnres:", xnres.dtype)
        #    #print("rres:", rres.dtype)
        #    print("xnres:", xnres.dtype, xnres)
        #    print("rres:", rres.dtype, rres)
        assert np.allclose(xnres, rres)

    def test_mean_groupby_slice_transpose1(self):
        offset = 25
        def ramba_impl(x, slice_size):
            rlin = ramba.fromarray(x)
            rlin = rlin.T
            rlin = rlin[:, offset:offset+slice_size]
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range(start="1/1/2001", periods=slice_size, freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=365)
            #breakpoint()
            gbmean = gb.mean()
            #print("ramba gbmean:", type(gbmean), gbmean.dtype, gbmean.shape, gbmean)
            #return gbmean
            final = gb - gbmean
            #print("final ramba:", final.dtype)
            return final.asarray()

        def xarray_numpy_impl(rlin, slice_size):
            rlin = rlin.T
            rlin = rlin[:, offset:offset+slice_size]
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range(start="1/1/2001", periods=slice_size, freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbmean = gb.mean("time")
            #print("xarray gbmean:", type(gbmean), gbmean.dtype, gbmean.shape, gbmean)
            #return gbmean
            final = gb - gbmean
            #print("final xarray:", final.dtype)
            return final.data

        #size = (1, 366)
        #size = (1, 731)
        total_size = (400, 1)
        slice_size = 365
        rlin = np.arange(total_size[0] * total_size[1]).reshape(total_size)
        #rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin, slice_size)
        rres = ramba_impl(rlin, slice_size)
        #with np.printoptions(threshold=np.inf):
        #    #print("xnres:", xnres.dtype)
        #    #print("rres:", rres.dtype)
        #    print("xnres:", xnres.dtype, xnres)
        #    print("rres:", rres.dtype, rres)
        assert np.allclose(xnres, rres)

    def test_mean_groupby_slice_transpose2(self):
        offset = 25
        def ramba_impl(x, slice_size):
            rlin = ramba.fromarray(x)
            rlin = rlin[offset:offset+slice_size, :]
            rlin = rlin.T
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range(start="1/1/2001", periods=slice_size, freq="D")},
                dims=("x", "time"),
            )
            coords = da.coords["time"].values
            coord_days = ramba.array([pd.Timestamp(x).dayofyear - 1 for x in coords])
            gb = da.data.groupby(1, coord_days, num_groups=365)
            #breakpoint()
            gbmean = gb.mean()
            #print("ramba gbmean:", type(gbmean), gbmean.dtype, gbmean.shape, gbmean)
            #return gbmean
            final = gb - gbmean
            #print("final ramba:", final.dtype)
            return final.asarray()

        def xarray_numpy_impl(rlin, slice_size):
            rlin = rlin[offset:offset+slice_size, :]
            rlin = rlin.T
            da = xarray.DataArray(
                rlin,
                coords={"time":pd.date_range(start="1/1/2001", periods=slice_size, freq="D")},
                dims=("x", "time"),
            )
            gb = da.groupby("time.dayofyear")
            gbmean = gb.mean("time")
            #print("xarray gbmean:", type(gbmean), gbmean.dtype, gbmean.shape, gbmean)
            #return gbmean
            final = gb - gbmean
            #print("final xarray:", final.dtype)
            return final.data

        #size = (1, 366)
        #size = (1, 731)
        total_size = (400, 1)
        slice_size = 365
        rlin = np.arange(total_size[0] * total_size[1]).reshape(total_size)
        #rlin = np.random.random(size)
        xnres = xarray_numpy_impl(rlin, slice_size)
        rres = ramba_impl(rlin, slice_size)
        #with np.printoptions(threshold=np.inf):
            ##print("xnres:", xnres.dtype)
            ##print("rres:", rres.dtype)
            #print("xnres:", xnres.dtype, xnres)
            #print("rres:", rres.dtype, rres)
        assert np.allclose(xnres, rres)

