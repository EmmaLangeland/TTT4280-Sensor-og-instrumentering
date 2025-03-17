import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sc 

G = -10 #20dB 20*log(10) = 20

R_2 = 100000 #100kOhm

R_1 = - R_2/G

f_L = 3.5
f_H =2.8e3 #Hz

C_1 = (1/(2*np.pi*R_1*f_L)) *1e6 #microfarad
C_2 = (1/(2*np.pi*R_2*f_H)) *1e12 #picofarad


print("R_1: ", R_1)
print("R_2: ", R_2)

print("C_1: ", C_1)
print("C_2: ", C_2)
