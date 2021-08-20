import ramba
import numpy as np
import random

class TestBroadcast:
    def test1(self):
        N = 10

        a = ramba.arange(N)
        anp = np.arange(N)
        a_l = a.asarray()
        assert(np.array_equal(anp, a_l))

        X = ramba.fromfunction(lambda x,y: x+y, (N,1))
        Xnp = np.fromfunction(lambda x,y: x+y, (N,1))
        X_l = X.asarray()
        assert(np.array_equal(Xnp, X_l))

        rnp = anp + Xnp
        r = a + X
        r_l = r.asarray()
        assert(np.array_equal(rnp, r_l))


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
