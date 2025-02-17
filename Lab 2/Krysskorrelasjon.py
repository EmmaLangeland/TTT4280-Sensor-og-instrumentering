import numpy as np
import matplotlib.pyplot as plt
from sympy import *
from scipy import signal

fs = 3000

x = [0,1,2,3,4]
y = [2,3,4,5,6]

#Slower
#r_xy = np.correlate(x,y, mode='full')
#r_xy_abs = np.abs(r_xy)

#Faster
r_xy_s = signal.correlate(x,y, mode='full')
r_xy_s_abs = np.abs(r_xy_s)

l_max = np.argmax(r_xy_s_abs) - (len(x) - 1)
delta_t = l_max/fs

print(delta_t)
