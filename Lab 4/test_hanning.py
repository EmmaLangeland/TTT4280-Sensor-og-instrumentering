import numpy as np
import matplotlib.pyplot as plt

# Parametere
fs = 1000  # Samplingfrekvens i Hz
f = 50     # Frekvensen til sinusen i Hz
pad = 500  # Zero-padding
t = np.arange(0, 1, 1/fs)  # 1 sekund med 1000 samples

# Generer I- og Q-komponenter
IF_I = np.sin(2 * np.pi * f * t)  # I-komponent
IF_Q = np.cos(2 * np.pi * f * t)  # Q-komponent

# Dummy-kolonner
dummy = np.ones_like(t)
data = np.column_stack((IF_I, dummy, dummy, dummy, IF_Q))

# --- Hanning-funksjonen (fra deg) ---
def Hanning(data, pad):
    IF_I = data[:,0]  # ADC0
    IF_Q = data[:,4]  # ADC4
    window_I = np.hanning(len(IF_I))
    window_Q = np.hanning(len(IF_Q))
    hanning_window_I = IF_I * window_I
    hanning_window_Q = IF_Q * window_Q
    hanning_window_padded_I = np.pad(hanning_window_I, (0, pad))
    hanning_window_padded_Q = np.pad(hanning_window_Q, (0, pad))
    hanning_window_padded = [hanning_window_padded_I, 1, 1, 1, hanning_window_padded_Q]
    return hanning_window_padded

# --- FFT-funksjon (fra deg) ---
def Regn_ut_FFT_pad(data):
    IF_I = data[0]
    IF_Q = data[4]
    IQ = IF_I + 1j * IF_Q
    Nfft = len(IQ)
    FFT_doppler = np.fft.fft(IQ, Nfft)
    FFT_doppler = np.abs(FFT_doppler)
    Doppler_shift = np.fft.fftshift(FFT_doppler)
    freqs = np.fft.fftfreq(Nfft, 1/fs)
    freqs = np.fft.fftshift(freqs)
    return Doppler_shift, freqs

# --- FFT uten Hanning og padding ---
data_uten = [data[:,0], 1, 1, 1, data[:,4]]
fft_uten, freqs_uten = Regn_ut_FFT_pad(data_uten)

# --- FFT med Hanning og padding ---
data_med = Hanning(data, pad)
fft_med, freqs_med = Regn_ut_FFT_pad(data_med)

# --- Plot begge FFT-ene ---
plt.figure(figsize=(12, 6))
plt.plot(freqs_uten, fft_uten, label='Uten Hanning og padding')
plt.plot(freqs_med, fft_med, label='Med Hanning og padding', linestyle='--')
plt.title('FFT med og uten Hanning + padding')
plt.xlabel('Frekvens (Hz)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

