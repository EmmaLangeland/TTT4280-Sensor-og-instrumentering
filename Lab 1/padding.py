import numpy as np
import matplotlib.pyplot as plt

f=1000
t = np.linspace(0,20, 10000)
signal = np.sin(2*np.pi*f*t)

plt.plot(t, signal)
plt.title("Sinusbølge med frekvens f= 1000Hz")
plt.ylabel("Amplitude [V]")
plt.xlabel("tid [s]")
plt.show()


padded = np.pad(signal, (0,10000))
t= np.linspace(0,40, 20000)
plt.plot(t, padded)
plt.title("Sinusbølge med frekvens f= 1000Hz")
plt.ylabel("Amplitude [V]")
plt.xlabel("tid [s]")
plt.show()