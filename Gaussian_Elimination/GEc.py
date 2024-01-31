import numpy as np 
import ctypes ,os
import time

# best 33.689 X faster 
####################3
def GaussianElimination(A):
    n,_ = A.shape
    """
    for i in range(1,n):
        for j in range(0,i):
            coeff = A[i,j]/A[j,j]
            A[i] -= A[j]*coeff
    """
    for i in range(0,n):
        for j in range(i+1,n):
            coeff = A[i,j]/A[i,i]
            A[:,j] -= A[:,i]*coeff
    #print(A.round(2))
    return A

def GaussianElimination2(A):
    n,_ = A.shape
    for i in range(1,n):
        for j in range(0,i):
            coeff = A[i,j]/A[j,j]
            A[i] -= A[j]*coeff
    
    return A
####################
n=800
N = 10

A  = np.random.random((n,n))

T1 = 0
for i in range(N):
    t = time.time()
    C = GaussianElimination(A)
    t2= time.time()
    T1 += t2-t
    print(f"python GE1 Trans time in iter-{i}/{N} :",t2-t)
T1/=N
print("Time GE1 :",T1)

T3 = 0
for i in range(N):
    t = time.time()
    C = GaussianElimination(A)
    t2= time.time()
    T3 += t2-t
    print(f"python GE2 time in iter-{i}/{N} :",t2-t)
T3/=N
print("Time GE2 :",T3)

c_float_p = ctypes.POINTER(ctypes.c_double)
data = A.flatten()
data = data.astype(np.float64)
data_p = data.ctypes.data_as(c_float_p)
path = os.getcwd()

cdll = ctypes.CDLL(path+'/./GE.so',winmode=0)
cdll.GE.restype = ctypes.POINTER(ctypes.c_double * (n*n))
T2 = 0
for i in range(N):
    t = time.time()
    DATA = cdll.GE(data_p,ctypes.c_int(n))
    t2= time.time()
    print(f"CDLL time in iter-{i}/{n} :",t2-t)
    T2 += t2-t
T2 /= N
print("Time GE3 c :",T2)
print("gain : GE1",T1/T2," and GE2",T3/T2) #34.086 and 39.875
k = [i for i in DATA.contents ]

#print(k)
l = 5
b = np.array(k).reshape(n,n)
#print(b[:l,:l])
#print(C[:l,:l])
#print(t2-t)