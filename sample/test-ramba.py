# test-numpy.py
import ramba as np
import time

np.sync()
t0 = time.time()
A = np.arange(100*1000*1000)/1000.0
t1 = time.time()
np.sync()
print ("Intitialize array time:",t1-t0)

for i in range(5):
    t0 = time.time()
    B = np.sin(A)
    C = np.cos(A)
    D = B*B + C**2
    np.sync()
    t1 = time.time()
    print ("Iteration",i+1,"time:",t1-t0)

