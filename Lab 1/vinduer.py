import numpy as np
import matplotlib.pyplot as plt

signal = np.ones(51)

plt.plot(signal)
plt.title("Rektangulært vindu")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.show()

window = np.hanning(len(signal))
hanning_window = signal * window
plt.plot(hanning_window)
plt.title("Hanning vindu")
plt.ylabel("Amplitude")
plt.xlabel("Sample")
plt.show()