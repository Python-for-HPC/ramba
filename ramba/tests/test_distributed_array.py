import ramba
import numpy as np
import random
import numba
import time


@ramba.stencil
def stencil1(a):
    return a[-2, -2] + a[0, 0] + a[2, 2]


class TestStencil:
    def test_skeleton(self):                # Stencil skeleton
        a = ramba.ones((20, 20), local_border=3)
        b = ramba.sstencil(stencil1, a)
        c = ramba.copy(b)

    def test_weighted_subarrays(self):                # Stencil using weighted subarrays
        def impl(app):
            A = app.ones(100,dtype=float)
            B = app.zeros(100,dtype=float)
            for _ in range(10):
                B[2:-2] += 0.2*A[:-4] - 0.5*A[1:-3] + 0.4*A[2:-2] - 0.5*A[3:-1] + 0.2*A[4:]
                A *= 1.1
            return int(abs(B).sum() * 1e8)  # Test sum to 1e-8 precision

        run_both(impl)

    def test_read_after_write(self):                # Stencil using weighted subarrays
                                    # Here, stencil and update of source are same size, so depends
                                    # critically on read after write detection and not fusing loops
        def impl(app):
            A = app.ones(100,dtype=float)
            B = app.zeros(100,dtype=float)
            for _ in range(10):
                B[2:-2] += 0.2*A[:-4] - 0.5*A[1:-3] + 0.4*A[2:-2] - 0.5*A[3:-1] + 0.2*A[4:]
                A[2:-2] *= 1.1
            return int(abs(B).sum() * 1e8)  # Test sum to 1e-8 precision

        run_both(impl)

    def test_reduction_fusion(self):                # Stencil fused with reduction
        A = ramba.ones((50,5))
        v = (0.2*A[:-2] + 0.5*A[1:-1] + 0.3*A[2:]).sum(axis=0).sum()
        h = (0.2*A[:-2] + 0.5*A[1:-1] + 0.3*A[2:]).sum(axis=1).sum()
        s = (0.2*A[:-2] + 0.5*A[1:-1] + 0.3*A[2:]).sum(asarray=True)[0]
        z = (0.2*A[:-2] + 0.5*A[1:-1] + 0.3*A[2:]).sum()
        B = np.ones((50,5))
        n = (0.2*B[:-2] + 0.5*B[1:-1] + 0.3*B[2:]).sum()
        print(v,h,s,z,n)
        assert v==h and s==z and h==s and s==n

class TestApps:
    def test_matmul1(self):      # manual matmul using broadcast_to, transpose, and reduction
        def impl(app):
            A = app.fromfunction(lambda x, y: x + y, (20,30))
            B = app.fromfunction(lambda x, y: x + y, (30,40))
            return (app.broadcast_to(A.T, (40,30,20)).T * app.broadcast_to(B, (20,30,40))).sum(axis=1)

        run_both(impl)

    def test_matmul2(self):      # manual matmul using expand_dims, implicit broadcast, and reduction
        def impl(app):
            A = app.fromfunction(lambda x, y: x + y, (20,30))
            B = app.fromfunction(lambda x, y: x + y, (30,40))
            return (app.expand_dims(A,2)*B).sum(axis=1)

        run_both(impl)

    def test_matmul_big1(self):  # big test -- should work on GitHub (7GB) assuming fusion works
        A = ramba.fromfunction(lambda x, y: x + y, (1000,1100))
        B = ramba.fromfunction(lambda x, y: x + y, (1100,1200))
        C = (ramba.broadcast_to(A.T, (1200,1100,1000)).T * ramba.broadcast_to(B, (1000,1100,1200))).sum(axis=1)
        c_7_3 = ((np.arange(1100)+7)*(np.arange(1100)+3)).sum()
        assert C[7,3] == c_7_3

    def test_matmul_big2(self):  # big test -- should work on GitHub (7GB) assuming fusion works
        A = ramba.fromfunction(lambda x, y: x + y, (1000,1100))
        B = ramba.fromfunction(lambda x, y: x + y, (1100,1200))
        C = (ramba.expand_dims(A,2)*B).sum(axis=1)
        c_12_4 = ((np.arange(1100)+12)*(np.arange(1100)+4)).sum()
        assert C[12,4] == c_12_4

    def test_pi_integration(self):  # integtrate 1/(1+x^2) from 0 to 1 --> arctan(1) --> pi/4
        def impl(app):
            nsteps = 1000
            step = 1.0/nsteps
            X = app.linspace(0.5*step, 1.0-0.5*step, num=nsteps)
            Y = 1.0 / (1.0+X*X)
            pi = 4.0 * step * app.sum(Y)
            return int(pi*1e8)    # Test to 1e-8 precision

        run_both(impl)

    def test_pi_integration_fused(self):  # Should fit on GitHub VM (7GB) if fused and no arrays materialized
        def calc_pi(nsteps):
            step = 1.0/nsteps
            X = ramba.linspace(0.5*step, 1.0-0.5*step, num=nsteps)
            Y = 4.0 * step / (1.0+X*X)
            return ramba.sum(Y, asarray=True)   # keep in array form to defer caclulation of sum until after function returns

        pi_arr = calc_pi(2000*1000*1000)
        print (pi_arr[0])



class TestFusion:
    def test_fuse(self):
        a = ramba.zeros(1000,dtype=float)
        ramba.sync()
        a += 1      # warmup
        ramba.sync()
        t0 = time.time()
        a += 1      # run once
        ramba.sync()
        t1 = time.time()
        for _ in range(10): # warmup for run 10
            a+=1
        ramba.sync()
        t2 = time.time()
        for _ in range(10): # run 10
            a+=1
        ramba.sync()
        t3 = time.time()
        overhead1 = t1-t0
        overhead10 = t3-t2

        a = ramba.zeros(500*1000*1000,dtype=float)  # Should fit in GitHub runner VM (7GB RAM)
        ramba.sync()
        t0 = time.time()
        a += 1      # run once
        ramba.sync()
        t1 = time.time()
        for _ in range(10): # run 10
            a+=1
        ramba.sync()
        t2 = time.time()
        runtime1 = t1-t0
        runtime10 = t2-t1
        exec1 = runtime1-overhead1
        exec10 = runtime10-overhead10
        print (runtime1, runtime10, overhead1, overhead10, exec1, exec10, exec10/exec1)
        assert(exec10<2*exec1)

    def test_nofuse(self):
        a = ramba.zeros(1000,dtype=float)
        ramba.sync()
        a += 1      # warmup
        ramba.sync()
        t0 = time.time()
        a += 1      # run once
        ramba.sync()
        t1 = time.time()
        for i in range(10): # warmup for run 10
            a[i:]+=1
        ramba.sync()
        t2 = time.time()
        for i in range(10): # run 10
            a[i:]+=1
        ramba.sync()
        t3 = time.time()
        overhead1 = t1-t0
        overhead10 = t3-t2

        a = ramba.zeros(500*1000*1000,dtype=float)  # Should fit in GitHub runner VM (7GB RAM)
        ramba.sync()
        t0 = time.time()
        a += 1      # run once
        ramba.sync()
        t1 = time.time()
        for i in range(10): # run 10
            a[i:]+=1
        ramba.sync()
        t2 = time.time()
        runtime1 = t1-t0
        runtime10 = t2-t1
        exec1 = runtime1-overhead1
        exec10 = runtime10-overhead10
        print (runtime1, runtime10, overhead1, overhead10, exec1, exec10, exec10/exec1)
        assert(exec10>5*exec1)

    def test_fuse2(self):
        a = ramba.ones(500*1000*1000,dtype=float)  # Should fit in GitHub runner VM (7GB RAM)
        a += (7*a-3)+(4*a+5*a)      # Should continue to fit if fused, no temporaries materialized
        assert a[0]==14

class TestBroadcast:
    def test1(self):
        N = 10

        a = ramba.arange(N)
        anp = np.arange(N)
        a_l = a.asarray()
        assert np.array_equal(anp, a_l)

        X = ramba.fromfunction(lambda x, y: x + y, (N, 1))
        Xnp = np.fromfunction(lambda x, y: x + y, (N, 1))
        X_l = X.asarray()
        assert np.array_equal(Xnp, X_l)

        rnp = anp + Xnp
        r = a + X
        r_l = r.asarray()
        assert np.array_equal(rnp, r_l)

    def test2(self):
        N = 100

        a = ramba.arange(N)
        anp = np.arange(N)
        a_l = a.asarray()
        assert np.array_equal(anp, a_l)

        X = ramba.fromfunction(lambda x, y: x + y, (N, 1))
        Xnp = np.fromfunction(lambda x, y: x + y, (N, 1))
        X_l = X.asarray()
        assert np.array_equal(Xnp, X_l)

        rnp = anp + Xnp
        r = a + X
        r_l = r.asarray()
        assert np.array_equal(rnp, r_l)


def rb_comparer(np_res, ramba_res, array_comp=np.array_equal):
    if isinstance(np_res, tuple):
        assert isinstance(ramba_res, tuple)
        assert len(np_res) == len(ramba_res)
        for i in range(len(np_res)):
            rb_comparer(np_res[i], ramba_res[i], array_comp)
    elif isinstance(np_res, np.ndarray):
        assert isinstance(ramba_res, ramba.ndarray)
        ramba_local = ramba_res.asarray()
        assert array_comp(np_res, ramba_local)
    else:
        #assert np_res == ramba_res
        assert array_comp(np_res,ramba_res)   # Note: this works for scalars as well


def run_both(func, *args, array_comp=np.array_equal):
    ramba_res = func(ramba, *args)
    ramba.sync()
    np_res = func(np, *args)
    rb_comparer(np_res, ramba_res, array_comp)


class TestOps:
    ops = ["+", "-", "*", "/", "//"]

    def test1(self):  # regular distributed + distributed
        def impl(app, op):
            a = app.ones((100, 100))
            b = app.ones((100, 100))
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test2(self):  # non-distributed + non-distributed
        def impl(app, op):
            a = app.ones((5, 5))
            b = app.ones((5, 5))
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test2_0d(self):  # non-distributed + non-distributed
        def impl(app, op):
            a = app.ones(())
            b = app.ones(())
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test3(self):  # ramba distributed + numpy
        def impl(app, op):
            a = app.ones((100, 100))
            b = np.ones((100, 100))
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test4(self):  # numpy + ramba distributed
        def impl(app, op):
            a = np.ones((100, 100))
            b = app.ones((100, 100))
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test5(self):
        def impl(app, op):  # ramba non-distributed + numpy
            a = app.ones((5, 5))
            b = np.ones((5, 5))
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test6(self):
        def impl(app, op):  # numpy + ramba non-distributed
            a = np.ones((5, 5))
            b = app.ones((5, 5))
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test5_0d(self):
        def impl(app, op):  # ramba non-distributed + numpy
            a = app.ones(())
            b = np.ones(())
            return eval("a" + op + "b")

        print("starting test5_0d")
        [run_both(impl, x) for x in TestOps.ops]

    def test6_0d(self):
        def impl(app, op):  # numpy + ramba non-distributed
            a = np.ones(())
            b = app.ones(())
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test7(self):
        def impl(app, op):  # ramba distributed + constant
            a = app.ones((100, 100))
            b = 7.0
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test8(self):
        def impl(app, op):  # constant + ramba distributed
            a = 7.0
            b = app.ones((100, 100))
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test9(self):
        def impl(app, op):  # ramba non-distributed + constant
            a = app.ones((5, 5))
            b = 7.0
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test10(self):
        def impl(app, op):  # constant + ramba non-distributed
            a = 7.0
            b = app.ones((5, 5))
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test9_0d(self):
        def impl(app, op):  # ramba non-distributed + constant
            a = app.ones(())
            b = 7.0
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test10_0d(self):
        def impl(app, op):  # constant + ramba non-distributed
            a = 7.0
            b = app.ones(())
            return eval("a" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test11(self):
        def impl(app, op):  # ramba distributed array + numpy.float
            a = app.ones((100, 100))
            b = np.ones(1)
            return eval("a" + op + "b[0]")

        [run_both(impl, x) for x in TestOps.ops]

    def test12(self):
        def impl(app, op):  # numpy.float + ramba distributed array
            a = np.ones(1)
            b = app.ones((100, 100))
            return eval("a[0]" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test13(self):
        def impl(app, op):  # ramba non-distributed array + numpy.float
            a = app.ones((5, 5))
            b = np.ones(1)
            return eval("a" + op + "b[0]")

        [run_both(impl, x) for x in TestOps.ops]

    def test14(self):
        def impl(app, op):  # numpy.float + ramba non-distributed array
            a = np.ones(1)
            b = app.ones((5, 5))
            return eval("a[0]" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]

    def test13_0d(self):
        def impl(app, op):  # ramba non-distributed array + numpy.float
            a = app.array(13)
            b = np.ones(1)
            return eval("a" + op + "b[0]")

        [run_both(impl, x) for x in TestOps.ops]

    def test14_0d(self):
        def impl(app, op):  # numpy.float + ramba non-distributed array
            a = np.ones(1)
            b = app.array(13)
            return eval("a[0]" + op + "b")

        [run_both(impl, x) for x in TestOps.ops]



class TestDgemm:
    def test_2Dx1D(self):
        def impl(app, i, j):
            X = app.fromfunction(lambda x, y: x + y, (i, j))
            theta = app.fromfunction(lambda x: x, (j,), dtype=X.dtype)
            res = X @ theta
            return res

        for i in range(4, 50):
            for j in range(4, 20):
                run_both(impl, i, j)

    def test_2Dx2D(self):
        def impl(app, i, j, k):
            X = app.fromfunction(lambda x, y: x + y, (i, j))
            theta = app.fromfunction(lambda x, y: x + y, (j, k), dtype=X.dtype)
            res = X @ theta
            return res

        for i in range(4, 20):
            for j in range(4, 20):
                for k in range(1, 10):
                    run_both(impl, i, j, k)

    def test_2Dx1DT(self):  # 2D x transposed 1D
        def impl(app, i, j):
            XnonT = app.fromfunction(lambda x, y: x + y, (j, i))
            X = XnonT.T
            theta = app.fromfunction(lambda x: x, (j,), dtype=X.dtype)
            res = X @ theta
            return res

        for i in range(4, 50):
            for j in range(4, 20):
                run_both(impl, i, j)

    def test_2DTx2DT(self):  # transposed 2D x transposed 2D
        def impl(app, i, j, k):
            X = app.fromfunction(lambda x, y: x + y, (j, i))
            theta = app.fromfunction(lambda x, y: x + y, (k, j), dtype=X.dtype)
            X = X.T
            theta = theta.T
            res = X @ theta
            return res

        for i in range(4, 20):
            for j in range(4, 20):
                for k in range(1, 10):
                    run_both(impl, i, j, k)

    def test_2Dx1D_slice(self):
        def impl(app, i, j, k, l):
            X = app.fromfunction(lambda x, y: x + y, (i, j))
            theta = app.fromfunction(lambda x: x, (j,), dtype=X.dtype)
            res = X[:, l : l + k] @ theta[l : l + k]
            return res

        for i in range(4, 50):
            for j in range(4, 20):
                for k in range(1, j):
                    for l in range(j - k):
                        run_both(impl, i, j, k, l)



class TestBasic:
    def test1(self):
        def impl(app):
            return app.ones((100, 100))

        run_both(impl)

    def test2(self):
        def impl(app):
            return app.ones((5, 5))

        run_both(impl)

    def test4(self):
        def impl(app):
            return app.arange(120)

        run_both(impl)

    def test5(self):
        a = np.arange(120)
        b = ramba.fromarray(a)
        a2 = a * a
        b2 = b * b
        b2l = b2.asarray()
        assert np.array_equal(a2, b2l)

    def test6(self):
        def impl(app):
            a = app.arange(120)
            b = a * a
            c = 1000 - b
            return c

        run_both(impl)

    def test7(self):
        def impl(index):
            return index[0] * 100

        try:
            a = ramba.init_array(120, impl)
            ramba.sync()
        except:
            assert 0

    def test8(self):
        @numba.njit
        def impl(index):
            return index[0] * 100

        try:
            a = ramba.init_array(120, impl)
        except:
            assert 0
        ramba.sync()

    def test9(self):
        def impl(index):
            return (index[0] + index[1]) * 5

        a = ramba.init_array((120, 100), impl)
        ramba.sync()

    def test10(self):
        @numba.njit
        def impl(index):
            return (index[0] * index[1]) + 7

        a = ramba.init_array((120, 100), impl)
        ramba.sync()

    def test11(self):
        try:
            a = ramba.init_array(120, lambda x: x[0] * 100)
            ramba.sync()
        except:
            assert 0

    def test12(self):
        try:
            a = ramba.init_array((120, 100), lambda x: (x[0] * x[1]) + 5)
            ramba.sync()
        except:
            assert 0

    def test13(self):
        def impl(app):
            a = app.arange(120)
            a -= 7
            a = abs(a)
            return a

        run_both(impl)

    def test_identity(self):
        def impl(app):
            a = app.identity(100)
            return a

        run_both(impl)

    def test_eye1(self):
        def impl(app):
            a = app.eye(20,k=-3)
            return a

        run_both(impl)

    def test_eye2(self):
        def impl(app):
            a = app.eye(20,30,k=7)
            return a

        run_both(impl)

    def test_asarray_after_base_update(self):
        def impl():
            a = ramba.ones(120)
            b = a[10:50]
            a += 7
            return b.asarray()

        res = impl()
        a = np.ones(120)
        b = a[10:50]
        a += 7
        print("asdf:", res, b)
        assert np.array_equal(b, res)

    """
    def test14(self):
        def impl(app):
            a = app.arange(120)
            b = app.ones(120)
            a = a * 7 + 3
            c = a + b + b
            d = -c
            breakpoint()
            b[45:55] = d[22:32] + a[48:58]
            return b

        run_both(impl)
    """
    def test_slice(self):
        # Test slice indexing / views
        def impl(app):
            a = app.arange(200)
            a[20:120] += 50
            b = a[40:140]
            b -= 20
            c = a[60:160] - 25
            d = b + a[80:180]
            return b+c+d

        run_both(impl)

    def test_skipslice(self):
        # Test slice indexing / views
        def impl(app):
            a = app.arange(200)
            a[20:120:3] += 50
            b = a[40:108:2]
            b -= 20
            c = a[60:196:4] - 25
            d = b + a[80:180:3]
            return b+c+d

        run_both(impl)

    def test_skipslice2(self):
        # Test slice indexing / views
        def impl(app):
            a = app.fromfunction(lambda i, j: i + j, (500, 50), dtype=int)
            b = a[40:340:3,20::2]
            b[b>50] -= 20
            c = app.broadcast_to(b.T,(70,15,100))
            d = c[15:25:2,2:7,::7] - c[20:30:2,6:11, 1::7] + 4
            e = app.sum(d)
            return d+e

        run_both(impl)

    def test_negative_skipslice(self):
        # Test slice indexing / views
        def impl(app):
            a = app.fromfunction(lambda i, j: i + j, (500, 50), dtype=int)
            b = a[340:40:-3,:20:-2]
            b[b>50] -= 20
            c = app.broadcast_to(b.T,(70,15,100))
            d = c[15:25:2,7:2:-1,::-3] - c[30:20:-2,6:11, -1::-3] + 4
            e = app.sum(d)
            return d+e

        run_both(impl)

    def test_smap1(self):
        a = ramba.arange(100)
        b = ramba.smap("lambda x: 3*x-7", a)
        c = ramba.smap(lambda x: 3*x-7, a)
        d = np.arange(100)*3-7
        rb_comparer(d, b)
        rb_comparer(d, c)

    def test_smap2(self):
        a = ramba.fromfunction(lambda i,j: i+j,(100,100))
        b = ramba.smap("lambda x: 3*x-7", a)
        c = ramba.smap(lambda x: 3*x-7, a)
        d = np.fromfunction(lambda i,j: i+j, (100,100))*3-7
        rb_comparer(d, b)
        rb_comparer(d, c)

    def test_smap3(self):
        a = ramba.arange(100)
        a2 = a*a
        b = ramba.smap("lambda x,y: 3*x-7*y", a2, a)
        c = ramba.smap(lambda x,y: 3*x-7*y, a2, a)
        d = np.arange(100)
        d = 3*d*d-7*d
        rb_comparer(d, b)
        rb_comparer(d, c)

    # NOTE: Should test smap, smp_index with slices, transpose, etc.

    def test_smap_index1(self):
        a = ramba.arange(100)-25
        b = ramba.smap_index("lambda i,x: 7*i+x", a)
        c = ramba.smap_index(lambda i,x: 7*i+x, a)
        d = np.arange(100)-25
        d = 7*np.arange(100)+d
        rb_comparer(d, b)
        rb_comparer(d, c)

    def test_smap_index2(self):
        a = ramba.ones((100,100))
        b = ramba.smap_index("lambda i,x: 7*i[0]-i[1]+x", a)
        c = ramba.smap_index(lambda i,x: 7*i[0]-i[1]+x, a)
        d = np.fromfunction(lambda i,j: 7*i-j+1,((100,100)))
        rb_comparer(d, b)
        rb_comparer(d, c)

    def test_smap_index3(self):
        a = ramba.arange(100)-25
        a2 = ramba.arange(100)*a
        b = ramba.smap_index("lambda i,x,y: 7*i+x-4*y", a,a2)
        c = ramba.smap_index(lambda i,x,y: 7*i+x-4*y, a,a2)
        d = np.arange(100)-25
        d2 = np.arange(100)*d
        d = 7*np.arange(100)+d-4*d2
        rb_comparer(d, b)
        rb_comparer(d, c)

    def test_masked(self):
        # Test boolean mask indexing
        def impl(app):
            a = app.arange(200)
            a[a%5==1] -= 50
            return a

        run_both(impl)

    def test_masked_reduction(self):
        # Test boolean mask indexing
        def impl(app):
            a = app.fromfunction(lambda i, j: i + j, (50, 50), dtype=int)
            return a[a<20].sum()

        run_both(impl)

    def test_where1(self):
        # Test of "where" in which cond,a,b are all the same size
        def impl(app):
            a = app.arange(200)
            b = app.ones(200)
            c = app.where(a > 133, a, b)
            return c

        run_both(impl)

    def test_where2(self):
        # Test of "where" in which b needs to be broadcast
        def impl(app):
            a = app.fromfunction(lambda i, j: i + j, (50, 50), dtype=int)
            b = app.ones((50, 1))
            c = app.where(a > 33, a, b)
            return c

        run_both(impl)

    def test_where3(self):
        # Test of "where" in which cond needs to be broadcast
        def impl(app):
            a = app.fromfunction(lambda i, j: i + j, (50, 50), dtype=int)
            e = app.fromfunction(lambda i, j: 500 + i + j, (50, 50), dtype=int)
            b = app.fromfunction(lambda i, j: i + 200, (50, 1), dtype=int)
            c = app.where(b > 233, a, e)
            return c

        run_both(impl)

    def test_triu1(self):
        # Test of "where" in which cond needs to be broadcast
        def impl(app):
            a = app.fromfunction(lambda i, j: i + j, (50, 50), dtype=int)
            return app.triu(a)

        run_both(impl)

    def test_triu2(self):
        # Test of "where" in which cond needs to be broadcast
        def impl(app):
            a = app.fromfunction(lambda i, j: i + j, (50, 50), dtype=int)
            return app.triu(a, k=-2)

        run_both(impl)

    def test_triu3(self):
        # Test of "where" in which cond needs to be broadcast
        def impl(app):
            a = app.fromfunction(lambda i, j: i + j, (50, 50), dtype=int)
            return app.triu(a, k=2)

        run_both(impl)

    def test_transpose_default_2(self):
        def impl(app):
            a = app.fromfunction(lambda i, j: i + j, (10, 20), dtype=int)
            return a.transpose()

        run_both(impl)

    def test_transpose_default_3(self):
        def impl(app):
            a = app.fromfunction(lambda i, j, k: i + j + k, (5, 10, 20), dtype=int)
            return a.transpose()

        run_both(impl)

    def test_transpose_tuple_3(self):
        def impl(app):
            a = app.fromfunction(lambda i, j, k: i + j + k, (5, 10, 20), dtype=int)
            return a.transpose((1, 2, 0))

        run_both(impl)

    def test_transpose_separate_3(self):
        def impl(app):
            a = app.fromfunction(lambda i, j, k: i + j + k, (5, 10, 20), dtype=int)
            return a.transpose(1, 2, 0)

        run_both(impl)

    def test_concatenate_1(self):
        def impl(app):
            shape = (20, 4)
            a = app.fromfunction(lambda i, j: i + j, shape, dtype=int)
            b = app.fromfunction(lambda i, j: i + j, shape, dtype=int)
            return app.concatenate([a, b], axis=0)

        run_both(impl)

    def test_concatenate_2(self):
        def impl(app):
            shape = (20, 4)
            a = app.fromfunction(lambda i, j: i + j, shape, dtype=int)
            b = app.fromfunction(lambda i, j: i + j, shape, dtype=int)
            return app.concatenate([a, b], axis=1)

        run_both(impl)

    def test_linspace_1(self):
        def impl(app):
            return app.linspace(1, 5, num=10)

        run_both(impl, array_comp=np.allclose)

    def test_linspace_2(self):
        def impl(app):
            return app.linspace(1, 5, num=200)

        run_both(impl, array_comp=np.allclose)

    def test_linspace_3(self):
        def impl(app):
            return app.linspace(1, 5, num=10, endpoint=False, retstep=True)[0]

        run_both(impl, array_comp=np.allclose)

    def test_linspace_4(self):
        def impl(app):
            return app.linspace(1, 5, num=200, endpoint=False, retstep=True)[0]

        run_both(impl, array_comp=np.allclose)

    def test_linspace_3_2(self):
        def impl(app):
            return app.linspace(1, 5, num=10, endpoint=False, retstep=True)[1]

        run_both(impl)

    def test_linspace_4_2(self):
        def impl(app):
            return app.linspace(1, 5, num=200, endpoint=False, retstep=True)[1]

        run_both(impl)

    def test_linspace_5(self):
        def impl(app):
            return app.linspace(10, 30, dtype=int)

        run_both(impl, array_comp=np.allclose)

    def test_mgrid_1(self):
        def impl(app):
            S = 20
            return app.mgrid[0:S, 0:S]

        run_both(impl)

    def test_mgrid_2(self):
        def impl(app):
            S = 20
            m, _ = app.mgrid[0:S, 0:S]
            return m

        run_both(impl)

    def test_mgrid_3(self):
        def impl(app):
            S = 5
            return app.mgrid[0:S, 0:S]

        run_both(impl)

    def test_mgrid_4(self):
        def impl(app):
            S = 5
            m, _ = app.mgrid[0:S, 0:S]
            return m

        run_both(impl)

    def test_reshape_1(self):
        shape = (50, 4)
        a = ramba.fromfunction(lambda i, j: i + j, shape, dtype=int)
        anp = np.fromfunction(lambda i, j: i + j, shape, dtype=int)
        al = a.asarray()
        assert np.array_equal(anp, al)

        b = a.reshape_copy((20, 10))
        bnp = anp.copy().reshape((20, 10))
        bl = b.asarray()
        assert np.array_equal(bnp, bl)

    def test_reshape_2(self):
        shape = (50,)
        a = ramba.fromfunction(lambda i: i, shape, dtype=int)
        anp = np.fromfunction(lambda i: i, shape, dtype=int)
        al = a.asarray()
        assert np.array_equal(anp, al)

        b = a.reshape((50, 1))
        bnp = anp.reshape((50, 1))
        bl = b.asarray()
        assert np.array_equal(bnp, bl)

    def test_pad1(self):
        shape = 200
        all_tests = [((0,1), {}),
                     ((2,0), {}),
                     ((3,4), {}),
                     ((0,3), {"constant_values":((0,7),)}),
                     ((5,0), {"constant_values":((5,0),)}),
                     ((1,2), {"constant_values":((3,4),)})]
        modes = ["constant", "edge", "wrap"]
        for mode in modes:
            for one_test in all_tests:
                if "constant_values" in one_test[1]:
                    if mode != "constant":
                        continue
                a = ramba.arange(shape)
                anp = np.arange(shape)
                b = ramba.pad(a, one_test[0], mode=mode, **one_test[1])
                bnp = np.pad(anp, one_test[0], mode=mode, **one_test[1])
                bl = b.asarray()
                if not np.array_equal(bnp, bl):
                    print(f"Fail: mode={mode}, pad={one_test}")
                assert np.array_equal(bnp, bl)

    def test_pad2(self):
        shapes = [(20, 30), (400, 1), (1, 300)]

        all_tests = [(((0,1), (0,1)), {}),
                     (((2,0), (3,0)), {}),
                     (((2,0), (0,3)), {}),
                     (((0,2), (3,0)), {}),
                     (((2,2), (0,3)), {}),
                     (((0,2), (3,3)), {}),
                     (((4,2), (3,3)), {})]
                     #((0,3), {"constant_values":((0,7),)}),
                     #((5,0), {"constant_values":((5,0),)}),
                     #((1,2), {"constant_values":((3,4),)})]
        for shape in shapes:
            modes = ["constant", "edge", "wrap"]
            for mode in modes:
                for one_test in all_tests:
                    if "constant_values" in one_test[1]:
                        if mode != "constant":
                            continue
                    a = ramba.fromfunction(lambda i, j: i + j, shape, dtype=int)
                    anp = np.fromfunction(lambda i, j: i + j, shape, dtype=int)
                    b = ramba.pad(a, one_test[0], mode=mode, **one_test[1])
                    bnp = np.pad(anp, one_test[0], mode=mode, **one_test[1])
                    bl = b.asarray()
                    if not np.array_equal(bnp, bl):
                        print(f"Fail: shape={shape}, mode={mode}, pad={one_test}")
                    assert np.array_equal(bnp, bl)

    def test_pad1_slice(self):
        orig_shape = 300
        sstart = 25
        shape = 200
        os_slice = slice(sstart, sstart + shape)
        all_tests = [((0,1), {}),
                     ((2,0), {}),
                     ((3,4), {}),
                     ((0,3), {"constant_values":((0,7),)}),
                     ((5,0), {"constant_values":((5,0),)}),
                     ((1,2), {"constant_values":((3,4),)})]
        modes = ["constant", "edge", "wrap"]
        for mode in modes:
            for one_test in all_tests:
                if "constant_values" in one_test[1]:
                    if mode != "constant":
                        continue
                o = ramba.arange(orig_shape)
                onp = np.arange(orig_shape)
                a = o[os_slice]
                anp = onp[os_slice]
                b = ramba.pad(a, one_test[0], mode=mode, **one_test[1])
                bnp = np.pad(anp, one_test[0], mode=mode, **one_test[1])
                bl = b.asarray()
                if not np.array_equal(bnp, bl):
                    print(f"Fail: mode={mode}, pad={one_test}")
                assert np.array_equal(bnp, bl)

    # Test pad with reduced dimension, increased dimension, transpose.
    # reduced dimension = a[slice, slice, constant]
    # increased dimension = a[slice, slice, np.newaxis] or expand_dims

class TestReduction:
    ops = ["sum", "prod", "min", "max"]

    def testFull(self):
        def impl(app, op):
            a = app.fromfunction(lambda i,j,k: 0.01*i+0.7*j+0.3*k+1, (8,6,4))
            return eval("a."+op+"()")

        [run_both(impl, x, array_comp=np.allclose) for x in TestReduction.ops]

    def testAxis1(self):
        def impl(app, op):
            a = app.fromfunction(lambda i,j,k: 10*i+7*j+k+1, (8,6,4))
            return eval("a."+op+"(axis=1)")

        [run_both(impl, x, array_comp=np.allclose) for x in TestReduction.ops]

    def testAxis2(self):
        def impl(app, op):
            a = app.fromfunction(lambda i,j,k: 10*i+7*j+k+1, (8,6,4))
            return eval("a."+op+"(axis=(1,0))")

        [run_both(impl, x, array_comp=np.allclose) for x in TestReduction.ops]

    def test_transpose_redcution(self):
        def impl(app):
            s1 = app.sum(app.ones((200,100), dtype=int),axis=0)
            s2 = app.sum(app.ones((200,100), dtype=int),axis=1)
            s3 = app.sum(app.ones((200,100), dtype=int).T,axis=0)
            s4 = app.sum(app.ones((200,100), dtype=int).T,axis=1)
            s5 = app.sum(app.ones((200,100), dtype=int))
            s6 = app.sum(app.ones((200,100), dtype=int).T)
            return s1,s2,s3,s4,s5,s6

        run_both(impl)

    def test_transpose_slice_redcution(self):
        def impl(app):
            s1 = app.sum(app.ones((200,100), dtype=int)[50:170],axis=0)
            s2 = app.sum(app.ones((200,100), dtype=int)[50:170],axis=1)
            s3 = app.sum(app.ones((200,100), dtype=int)[50:170].T,axis=0)
            s4 = app.sum(app.ones((200,100), dtype=int)[50:170].T,axis=1)
            s5 = app.sum(app.ones((200,100), dtype=int)[50:170])
            s6 = app.sum(app.ones((200,100), dtype=int)[50:170].T)
            return s1,s2,s3,s4,s5,s6

        run_both(impl)

    def test_transpose_slice_redcution2(self):
        def impl(app):
            s1 = app.sum(app.ones((200,100), dtype=int)[150:170],axis=0)
            s2 = app.sum(app.ones((200,100), dtype=int)[150:170],axis=1)
            s3 = app.sum(app.ones((200,100), dtype=int)[150:170].T,axis=0)
            s4 = app.sum(app.ones((200,100), dtype=int)[150:170].T,axis=1)
            s5 = app.sum(app.ones((200,100), dtype=int)[150:170])
            s6 = app.sum(app.ones((200,100), dtype=int)[150:170].T)
            return s1,s2,s3,s4,s5,s6

        run_both(impl)


class TestRandom:
    def test1(self):
        shape = (1000, 10)
        rs = ramba.random.RandomState(1337)
        X1 = rs.normal(loc=5.0, size=shape)
        ramba.sync()


class TestDel:
    def test1del(self):
        def impl(app):
            a = app.ones(100)
            del a

        run_both(impl)

    def test2del(self):
        def impl(app):
            a = app.ones(100)
            b = a + 3
            del a
            return b

        run_both(impl)

    def test3del(self):
        def impl(app):
            s = 0

            c = app.ones(100)
            d = c * 3
            del c

            for i in range(100):
                a = app.ones(200)
                b = a[37:137]
                c = b * 3
                s += c[42]

            d += s
            return d

        run_both(impl)

"""
class TestGeneric:
    def test6(self):
        ntests=10
        testsize=100

        def run_test(count):
            failcount=0
            for i in range(count):
                fail = False
                x = random.randint(1,100)
                y = random.randint(1,100)
                z = random.randint(1,100)
                al = np.random.randint(200,size=(x,y))*0.1
                bl = np.random.randint(200,size=(x,y))*0.1
                a = ramba.fromarray(al)
                b = ramba.fromarray(bl)
                cl = 2*(al+bl)
                c = 2*(a+b)
                dl = np.random.randint(200,size=(y,x))*0.1
                d = ramba.fromarray(dl)
                el = dl.T * bl - al
                e = d.T * b - a
                fl = al @ dl + 7
                f = a @ d + 7
                ax = a.asarray()
                bx = b.asarray()
                cx = c.asarray()
                dx = d.asarray()
                ex = e.asarray()
                fx = f.asarray()
                if not np.array_equal(al,ax):
                    assert(0)
                if not np.array_equal(bl,bx):
                    assert(0)
                if not np.array_equal(cl,cx):
                    assert(0)
                if not np.array_equal(dl,dx):
                    assert(0)
                if not np.array_equal(el,ex):
                    assert(0)
                if not np.allclose(fl,fx): fail=True
                if fail:
                    failcount += 1
            return failcount

        fails=0
        for i in range(ntests):
            fails += run_test(testsize)
        assert(fails == 0)

    def test6b(self):
        ntests=10
        testsize=100

        def get_rand_slices(k):
            a = random.randint(0,k-1)
            b = random.randint(a,k)
            s = b-a
            c = random.randint(0,k-s)
            d = c+s
            return slice(a,b),slice(c,d)

        def run_test(count):
            failcount=0
            Al = np.random.randint(20, size=(200,300))
            Bl = np.random.randint(20, size=(400,500))
            A = ramba.fromarray(Al)
            B = ramba.fromarray(Bl)
            for i in range(count):
                fail = False
                sl1a, sl1b = get_rand_slices(100)
                sl2a, sl2b = get_rand_slices(100)
                a = A[sl1a, sl2a]
                b = B[sl1b, sl2b]
                c = 2*(a+b)
                d = B[sl2b, sl1b]
                e = d.T * b - a
                f = a @ d + 7
                g = b @ a.T *4
                h = (d+7).T @ b.T
                al = Al[sl1a, sl2a]
                bl = Bl[sl1b, sl2b]
                cl = 2*(al+bl)
                dl = Bl[sl2b, sl1b]
                el = dl.T * bl - al
                fl = al @ dl + 7
                gl = bl @ al.T *4
                hl = (dl+7).T @ bl.T
                ax = a.asarray()
                bx = b.asarray()
                cx = c.asarray()
                dx = d.asarray()
                ex = e.asarray()
                fx = f.asarray()
                gx = g.asarray()
                hx = h.asarray()
                if not np.array_equal(al,ax):
                    assert(0)
                if not np.array_equal(bl,bx):
                    assert(0)
                if not np.array_equal(cl,cx):
                    assert(0)
                if not np.array_equal(dl,dx):
                    assert(0)
                if not np.array_equal(el,ex):
                    assert(0)
                if not np.allclose(fl,fx): fail=True
                if not np.allclose(gl,gx): fail=True
                if not np.allclose(hl,hx): fail=True
                if fail:
                    failcount += 1
            return failcount

        fails=0
        for i in range(ntests):
            fails += run_test(testsize)
        assert(fails == 0)
"""
