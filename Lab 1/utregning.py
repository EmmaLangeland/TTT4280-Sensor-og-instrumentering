import numpy as np

C1= 100e-6
C2= 470e-6
C3= 100e-9
L= 100e-3
def H(f):
    return 1/(1-(2*np.pi*f)**2*L*Ctot)

print(np.sqrt(2.4)*21)
Ctot= C1 + C2 + C3
print(f"total C: {Ctot}")

print(f"ved 100 Hz {np.abs(H(100))}")
print(f"ved 26 Hz {np.abs(H(26))}")
print(f"ved 21 Hz {np.abs(H(21))}")

def H_db(f):
    return 20*np.log10(np.abs(1/(1-(2*np.pi*f)**2*L*Ctot)))

print(f"ved 100 Hz {H_db(100)}")
print(f"ved 26 Hz {H_db(26)}")
print(f"ved 21 Hz {H_db(21)}")
print(f"ved 32 Hz {H_db(32)}")

'''
def H_db_norm(f):
    return 20*np.log10(np.abs(1/(1-(2*np.pi*f)**2*L*Ctot))) - np.max(20*np.log10(np.abs(1/(1-(2*np.pi*f)**2*L*Ctot))))

print(f"ved 100 Hz {H_db_norm(100)}")
print(f"ved 26 Hz {H_db_norm(26)}")
print(f"ved 21 Hz {H_db_norm(21)}")'''




