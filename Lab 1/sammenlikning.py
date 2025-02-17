import numpy as np
import matplotlib.pyplot as plt
from raspi_import import raspi_import
import scipy.signal as sc

#--------------------------------------------------------------------------------
#plot av testsignal med f=1kHz, A=1V og DC-offset 1.65V

Fs= 31250
T= 1/Fs
f=1000


sample_period, data = raspi_import('Test3.bin')

k= 3.3/4096 #2^12 pga 12-bits ADC bit_signal = Signal/k
data = data[1:,:]*k 
data = data - np.mean(data, axis=0) #fjerner DC-komponenten til signalet

print(data.shape) #(31250, 5)
print(sample_period) #3.2e-05
print(np.argmax(data))

print(data.shape[0])
print(data.shape[0]*sample_period)

time = np.linspace(0, data.shape[0]*sample_period, data.shape[0])

plt.plot(time, data[:,1], label=f'ADC {1}')
plt.legend(loc='upper right')
plt.grid()



time = np.linspace(0, data.shape[0]*sample_period, data.shape[0]+1000000)
signal = 1*np.sin(2*np.pi*f*time)
plt.plot(time, signal, label='Ideell sinus')
#plt.title(f"")
plt.xlabel("tid [s]")
plt.ylabel("Amplitude [V]")
plt.legend()
plt.xlim(0, 0.006)

plt.show()