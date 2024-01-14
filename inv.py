import numpy as np
import time
t=time.time()
n = 700
A = np.random.random((n,n))
c = A.copy()

ortho = np.zeros((n,n))
ortho[0] = A[0]/np.linalg.norm(A[0])

for i in range(n):
    ortho[:,i] = A[:,i]
    for j in range(i):
        ortho[:,i] -= ortho[:,j].dot(A[:,i])*ortho[:,j]
    ortho[:,i] /= np.linalg.norm(ortho[:,i])

a = np.zeros((n,n))

a = ortho.T.dot(A)*np.triu(np.ones((n,n)))
b = np.identity(n)

for i in range(n):
    b[i] *= 1/a[i,i]
    a[i] *= 1/a[i,i]
    
for j in range(1,n):
    for i in range(j):
        
        coeff = a[i,j]/a[j,j]
        a[i] -= coeff*a[j]
        b[i] -= coeff*b[j]


QT = ortho.T
R_inv = b
ainv = np.dot(R_inv,QT)
t2 = time.time()
print(t2-t)
#print(np.round(np.dot(R_inv,QT).dot(A),2))

#####################################################

import numpy as np
import time
t=time.time()

#n = 80

A = np.random.random((n,n))*1000
c = A.copy()

ortho = np.zeros((n,n))
ortho[0] = A[0]/np.linalg.norm(A[0])

for i in range(n):
    ortho[i] = A[i]
    for j in range(i):
        ortho[i] -= ortho[j].dot(A[i])*ortho[j]
    ortho[i] /= np.linalg.norm(ortho[i])

a = A.dot(ortho.T)*np.triu(np.ones((n,n))).T
b = np.identity(n)

k = 1/np.diag(a).reshape(n,1)
a*=k
b*=k
del k
#for i in range(n):
#    b[i] *= 1/a[i,i]
#    a[i] *= 1/a[i,i]


for i in range(n):
    for j in range(i):
        coeff = a[i,j]/a[j,j]
        a[i] -= coeff*a[j]
        b[i] -= coeff*b[j]

QT = ortho.T
R_inv = b

ainv = np.dot(QT,R_inv)
t2 = time.time()
print(t2-t)
#print(np.round(np.dot(QT,R_inv).dot(A),1))"""

'''for j in range(n):
    for i in range(j+1,n):
        print(i,j)
        
for i in range(n):
    for j in range(i):
        print(i,j)
'''
#for i in range()
