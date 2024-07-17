import numpy as np
import ctypes as cty
import pyXLR8
import time
import random
def generate_random_string(length):
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    random_string = "".join(random.choice(characters) for _ in range(length))
    return random_string

lib = pyXLR8.py2C_linkfile("substr_lib.c",flags=["-fprefetch-loop-arrays","-Ofast"])
lib.compile(globals())

import time
from tqdm import tqdm
import matplotlib.pyplot as plt

iters = 1000
T1 = []
T2 = []
x = []
for i in tqdm(range(1,500000,5000)):
    n = i
    ns = 64
    string = generate_random_string(n-ns-1)
    substr = generate_random_string(ns)
    string=string+substr+"h"
    start = time.time()
    for _ in range(iters):
        issubstr(string.encode('utf-8'),n,substr.encode('utf-8'),ns)
    end = time.time()
    T2.append((end-start)/iters)
    x.append(n)

plt.plot(x,T2)
plt.show()