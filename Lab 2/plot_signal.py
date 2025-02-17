import numpy as np
import matplotlib.pyplot as plt
from raspi_import import raspi_import
import scipy.signal as sc



#hente inn data fra måling
#Sinus_100Hz, Sinus_500Hz, Sinus_1500Hz, Sinus_2000Hz, Sinus_1000Hz

sample_period, data = raspi_import('1000Hz_sinus.bin')

k= 3.3/4096 #2^12 pga 12-bits ADC bit_signal = Signal/k
data = data[5000:,:]*k 
data = data - np.mean(data, axis=0) #fjerner DC-komponenten til signalet


print(data.shape) #(31250, 5)
print(sample_period) #3.2e-05
print(np.argmax(data))
print(data.shape[0])

time = np.linspace(0, (data.shape[0]*sample_period), data.shape[0])

#plotter de rekonstruerte sinussignalene hver for seg med en forskyvning i tid for penere illustrasjon
""" for i in range(data.shape[1]):
    plt.plot(time+ i/10000, data[:,i], label=f'ADC {i+1}') """

plt.plot(time, data[:,0], label=f'ADC {0}')

plt.xlabel('Tid [s]')
plt.ylabel('Amplitude [V]')
plt.title('Samplet sinussignal fra alle 5 AD-konvertere')
plt.legend(loc='upper right')
plt.grid()
#plt.xlim(0,0.01)
plt.show()
