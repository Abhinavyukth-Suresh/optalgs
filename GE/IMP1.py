"""
            GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING

1. Normal pure-python based implementation of Gaussian elimination with partial pivoting 

2. C-based implementation were the arrays taken in column major array to improve caching.
   a. change of memory arrange ment for effective caching 
   b. Vectorization with SIMD based parallelization with Advanced Vector Extensions and Streaming SIMD Extensions
   c. +compiler opotimizations -O3, ffast-math, loop-unrolling
MANUAL BUILD:
compile : gcc GE_PP.c -o GE_PP.so -O3 -msse -mavx2 -shared -fPIC -ffast-math -funroll-all-loops
source-code in GE_CP.c file for reference

Maximum Performace improvement of 215.96385X for 100x100 matrix on CDLL based implementation
on INTEL(11th Gen) i5-1135G7.

1.Python based implementation :
    inputs matrix A augmented matrix of order (N x N+1)
    step 1. from given matrix/sub-matrix, calculate max value of cloumn.
            Find argmax i_max of column
    step 2. Bring  i_max row to i (current pivot row)
    setp 3. reduce succesive elements of cloumn to 0 via row operations.
            (Tradtional GE step)
    step 4. if N_sub_matrix - 1 != N: jmp setp 1. else return A  

2. C based implementation :
    inputs matrix AT, Transpose of augmented matrix A, of order (N+1,N)
    memory allocated in Fortran style array
    step 1. Calculate max vale and argmax (i_max,j_max) of the current pivoting row
    step 2. Bring  i_max row to i (current pivot row)
    setp 4. reduce succesive elements of row to 0 via column operations.
            by redifing the loops, such that function performs operation in contagious 
            memory location as much as possible to imporve memory caching involving 
            SIMD based parallelism.
    step 5. if N_sub_matrix - 1 != N: jmp setp 1. else return A

    

Author: Abhonav Yukth S
        IMS21007
"""
############             REQ LIBS           #############
import numpy as np
import ctypes ,os,platform 
from timeit import default_timer as timer

print('\033[95m')
print("TEST 1.... System 1")
print('\033[0m',end="")

################# EXTRACRING DATA ####################
with open("system.txt","r") as f:
    data = f.read()
    f.close()
lines = data.split("\n")
data = []
for i in lines:
    if i.strip()[0]=="#":
        pass
    elif "," in i:
        [row,clm] = i.split(",")
    else:
        l = i.split("  ")
        k = []
        for i,s in enumerate(l) :
            if s.strip()!="":
                k.append(float(s.strip()))
        data.append(k)

row = int(row)
clm = int(clm)
A = np.array(data)
a = A.T.flatten()

############       NAIVE IMPLEMENTATION      #############
# GAUSSIAN ELIMINATION WITH PARTIAL PIVOTING (GE PP)
def GE_PP(aug_mat):
    n = aug_mat.shape[0]
    for i in range(n):
        mri = i 
        for j in range(i + 1, n):
            if abs(aug_mat[j, i]) > abs(aug_mat[mri, i]):
                mri = j

        aug_mat[[i, mri]] = aug_mat[[mri, i]]
        for j in range(i + 1, n):
            factor = aug_mat[j, i] / aug_mat[i, i]
            aug_mat[j, i:] -= factor * aug_mat[i, i:]
    return aug_mat

############       GET SOLUTION FROM AUG MATRIX      #############
def GetSln(aug_mat): 
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (aug_mat[i, -1] - np.dot(aug_mat[i, i+1:n], x[i+1:])) / aug_mat[i, i]
    return x

############          TESTING         #############
n = row
start = timer()
sln1 = GE_PP(A)
end = timer()
solution = GetSln(sln1)

print('\033[92m')
print("Implementation 1 | python")
print("Solution:", solution)
print('\033[0m')

t1 = end-start
print("time els on python based impl : ",t1,end="\n\n")

#####     IMPPLEMENTATION 2    ####
#SOURCE GE_PP.c file
                        ###############        SETUP        ###################
path = os.getcwd()
libname = path + "\\GE_PP.so"
start = '\033[94m'
end = '\033[0m'
if not os.path.isfile(libname):
    print(start)
    print("BULIDING RESOURCES...")
    print("_"*60)
    print("checking for resources ...")
    if not os.path.isfile(path + "\\GE_PP.c"):
        print( '\033[91m')
        print("COMPILATION FAILED !")
        raise FileExistsError("file GE_PP.c file does not exist.\033[0m")
    print("processor info :",platform.processor())
    print("compiling shared object file...")
    print("checking for GCC")
    print("checking GCC version ...")
    ret = os.system("gcc --version")
    if ret != 0:
        print( '\033[91m')
        raise Exception("GCC not found!'\033[0m'")
    print("compiling resources ...")
    print("gcc GE_PP.c -o GE_PP.so -O3 -msse -mavx2 -shared -fPIC -ffast-math -funroll-all-loops")
    ret = os.system("gcc GE_PP.c -o GE_PP.so -O3 -msse -mavx2 -shared -fPIC -ffast-math -funroll-all-loops")
    if ret != 0:
        print( '\033[91m')
        raise Exception("compilation unsuccesful!'\033[0m'")
    else:
        print("compilation successfully completed!")
    print("_"*60)
    print("\n")
    print(end)

####################  TESTING CDLLL BASED IMPLEMENTATION   ##########################
c_float_p = ctypes.POINTER(ctypes.c_double)
data = a.flatten()
data = data.astype(np.float64)
data_p = data.ctypes.data_as(c_float_p)
path = os.getcwd()

try:
    cdll = ctypes.CDLL(path+'/./GE_PP.so',winmode=0)
except FileNotFoundError:
    print(" SHARED LIB NOT COMPILED")
    raise FileExistsError("SHARED LIB NOT COMPILED\nshared library not found!\ncompilie the code GE_CP.c with gcc\ngcc GE_PP.c -o GE_PP.so -O3 -msse -mavx2 -shared -fPIC -ffast-math -funroll-all-loops")
    sys.exit()
cdll.GE.restype = ctypes.POINTER(ctypes.c_double * (n*n+n))

start = timer()
DATA = cdll.GE(data_p,ctypes.c_int(n))
end = timer()

sln = np.array([i for i in DATA.contents ]).reshape(n+1,n).T
solution = GetSln(sln)

print('\033[92m')
print("Implementation 2 | python extended with C-based dynamically linked shared function")
print("Solution: ",solution)
print('\033[0m')
t2 = end-start

print("time els on CDLL based impl : ",end-start,end="\n")
print("_"*50)
print("performance improvement t1/t2 = ",t1/t2)
print("_"*50,"\n\n")

                    #####################################################
                            ##############  TEST 2 ##############
print('\033[95m')
print("TEST 2.... on System 2")
print('\033[0m',end="")
with open("system2.txt","r") as f:
    data = f.read()
    f.close()
lines = data.split("\n")
data = []
for i in lines:
    if i.strip()[0]=="#":
        pass
    elif "," in i:
        [row,clm] = i.split(",")
    else:
        l = i.split("  ")
        k = []
        for i,s in enumerate(l) :
            if s.strip()!="":
                k.append(float(s.strip()))
        data.append(k)

row = int(row)
clm = int(clm)
A = np.array(data)
a = A.T.flatten()

############     TESTING SYSTEM 2  ON PYTHON BASED IMPLEMENTATION       #############
start = timer()
sln1 = GE_PP(A)
end = timer()
solution = GetSln(sln1)

print('\033[92m')
print("Implementation 1 | python")
print("Solution:", solution)
print('\033[0m')
t1 = end-start
print("time els on python based impl : ",t1,end="\n\n")

##################      TESTING SYSTEM 2 ON CDLL BASED IMPLEMENTATION       #######################
c_float_p = ctypes.POINTER(ctypes.c_double)
data = a.flatten()
data = data.astype(np.float64)
data_p = data.ctypes.data_as(c_float_p)
path = os.getcwd()

start = timer()
DATA = cdll.GE(data_p,ctypes.c_int(n))
end = timer()

sln = np.array([i for i in DATA.contents ]).reshape(n+1,n).T
solution = GetSln(sln)

print('\033[92m')
print("Implementation 2 | python extended with C-based dynamically linked shared function")
print("Solution: ",solution)
print('\033[0m')
t2 = end-start
print("time els on CDLL based impl : ",end-start,end="\n")
print("_"*50)
print("performance improvement t1/t2 = ",t1/t2)
print("_"*50,"\n\n")

# END PROGRAM 
