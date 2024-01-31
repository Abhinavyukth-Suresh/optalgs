import numpy as np 
import ctypes ,os
import time
n=10
A  = np.random.random((n,n))

c_float_p = ctypes.POINTER(ctypes.c_double)
data = A.flatten()
data = data.astype(np.float64)
data_p = data.ctypes.data_as(c_float_p)
path = os.getcwd()

cdll = ctypes.CDLL(path+'/./test.so',winmode=0)
cdll.MATRIX_INV.restype = ctypes.POINTER(ctypes.c_double * (n*n))
t = time.time()
DATA = cdll.MATRIX_INV(data_p,ctypes.c_int(n))
t2= time.time()

k = [i for i in DATA.contents ]

#print(k)
b = np.array(k).reshape(n,n)
if n<17:print(b.dot(A).round(2))
print(t2-t)