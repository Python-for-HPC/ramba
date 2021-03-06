import ramba
import numpy as np
import random
import numba


@ramba.stencil
def stencil1(a):
    return a[-2, -2] + a[0, 0] + a[2, 2]


class TestStencil:
    def test1(self):
        a = ramba.ones((20, 20), local_border=3)
        b = ramba.sstencil(stencil1, a)
        c = ramba.copy(b)


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


def rb_comparer(np_res, ramba_res, array_comp):
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
        assert np_res == ramba_res


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

    def test9(self):
        def impl(index):
            return (index[0] + index[1]) * 5

        a = ramba.init_array((120, 100), impl)

    def test10(self):
        @numba.njit
        def impl(index):
            return (index[0] * index[1]) + 7

        a = ramba.init_array((120, 100), impl)

    def test11(self):
        try:
            a = ramba.init_array(120, lambda x: x[0] * 100)
        except:
            assert 0

    def test12(self):
        try:
            a = ramba.init_array((120, 100), lambda x: (x[0] * x[1]) + 5)
        except:
            assert 0

    def test13(self):
        def impl(app):
            a = app.arange(120)
            a -= 7
            a = abs(a)
            return a

        run_both(impl)

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


    def test_masked(self):
        # Test boolean mask indexing
        def impl(app):
            a = app.arange(200)
            a[a%5==1] -= 50
            return a

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

            c = ramba.ones(100)
            d = c * 3
            del c

            for i in range(100):
                a = ramba.ones(200)
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
