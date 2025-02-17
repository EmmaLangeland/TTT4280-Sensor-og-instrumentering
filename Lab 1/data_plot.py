import numpy as np
import matplotlib.pyplot as plt
from raspi_import import raspi_import
import scipy.signal as sc

#--------------------------------------------------------------------------------
#plot av testsignal med f=1kHz, A=1V og DC-offset 1.65V

Fs= 31250
T= 1/Fs
f=1000
t = np.linspace(0, 1, Fs)
signal = 1.65 + 1*np.sin(2*np.pi*f*t)

""" plt.plot(t, signal, label='x(t)')
plt.title(f"Sinussignal x(t) med $f$ = {f}Hz")
plt.xlabel("tid [s]")
plt.ylabel("Amplitude [V]")
plt.legend()
plt.ylim(0.5, 3)
plt.xlim(0, 0.006)
plt.show() """


#--------------------------------------------------------------------------------
#hente inn data fra måling
#Sinus_100Hz, Sinus_500Hz, Sinus_1500Hz, Sinus_2000Hz, Sinus_1000Hz

sample_period, data = raspi_import('Sinus_1000Hz.bin')

k= 3.3/4096 #2^12 pga 12-bits ADC bit_signal = Signal/k
data = data[1:,:]*k 
data = data - np.mean(data, axis=0) #fjerner DC-komponenten til signalet
#data = sc.detrend(data[:,1], -1, 'linear', 0, overwrite_data=False)

print(data.shape) #(31250, 5)
print(sample_period) #3.2e-05
print(np.argmax(data))

print(data.shape[0])

time = np.linspace(0, data.shape[0]*sample_period, data.shape[0])
#time = np.arange(0, 1, 1/31250)
#time = np.arange(data.shape[0])*sample_period
#time= np.arange(0, data.shape[0]*sample_period, sample_period/data.shape[0])

#plotter de rekonstruerte sinussignalene hver for seg med en forskyvning i tid for penere illustrasjon
for i in range(data.shape[1]):
    plt.plot(time+ i/10000, data[:,i], label=f'ADC {i+1}')

plt.xlim(0, 0.006)
plt.xlabel('Tid [s]')
plt.ylabel('Amplitude [V]')
plt.title('Samplet sinussignal fra alle 5 AD-konvertere')
plt.legend(loc='upper right')
plt.grid()
plt.show()


#------------------------------------------------------------------------
#hanning vindu med og uten padding 
def hanning(data, pad):
    window = np.hanning(len(data))  # Lager et Hanning-vindu med samme lengde som den data #erstatt med np.ones for å kun padde
    hanning_window = data * window
    hanning_window_padded = np.pad(hanning_window, (0,pad)) #zero padder bak
    return hanning_window_padded

#-------------------------------------------------------------------------
#Plot av fft

fs = 1/ sample_period
x_t = data[:,0] #x_t er dataen til en av ADC-ene, dersom jeg vil hente ut fra til, hiv på [0:87654] bakpå
print(f"lengden til x(t): {len(x_t)}")
N_fft= x_t.shape[0]

#sjekker at paddingen fungerer som ønsket ved å printe antall datapunkter og en verdi mot slutter av arrayet
print (f"N_fft: {N_fft} ")

X_f = np.fft.fft(x_t, N_fft) #fft av x_t
X_f = abs(X_f)
X_f = X_f[:int(N_fft/2)] #kutter spekteret i to for å kun vise det reelle signalet
X_f_db= 10*np.log10(X_f) 
X_f_db_normalisert = X_f_db - np.max(X_f_db)


#zero-padding
""" pad = 2**16#2**15 - len(x_t) #Juster antall pads etter ønske. 
x_t_zero = np.pad(x_t, (0,pad)) 
N_fft_zero= x_t_zero.shape[0]
print(f"lengden til det paddede signalet: {len(x_t_zero)}, skal være lik  {x_t_zero.shape[0]}")
X_f_zero = np.fft.fft(x_t_zero, N_fft_zero) #fft av x_t
X_f_zero = abs(X_f_zero)
X_f_zero = X_f_zero[:int(N_fft_zero/2)] #kutter spekteret i to for å kun vise det reelle signalet
X_f_db_zero= 10*np.log10(X_f_zero) 
X_f_db_normalisert_zero = X_f_db_zero - np.max(X_f_db_zero) """


#hanning og evt padding
x_t_hann = hanning(x_t, 2**16) #kaller på hanning vindu funksjonen. juster antall pads etter ønske. 2**15 - len(x_t)
N_fft_hann= x_t_hann.shape[0]
X_f_hann = np.fft.fft(x_t_hann, N_fft_hann) #fft av x_t
X_f_hann = abs(X_f_hann)
X_f_hann = X_f_hann[:int(N_fft_hann/2)] #kutter spekteret i to for å kun vise det reelle signalet
X_f_db_hann= 10*np.log10(X_f_hann) 
X_f_db_normalisert_hann = X_f_db_hann - np.max(X_f_db_hann)

frekvenser = np.linspace(0,fs/2,int(N_fft/2))

#frekvenser_zero = np.linspace(0,fs/2,int(N_fft_zero/2)) #for zero pad
frekvenser_hann = np.linspace(0,fs/2,int(N_fft_hann/2)) #for zero-padding og hanningvindu

#plt.plot(frekvenser_zero, X_f_db_normalisert_zero, label=f'Zero-padded', color='C1') #Matplotlib Default Colors (C1 = orange)
plt.plot(frekvenser, X_f_db_normalisert, label=f'Rektangulært', color='C0') #(C0 = blue)
plt.plot(frekvenser_hann, X_f_db_normalisert_hann, label=f'Hanning', color='C1')
plt.xscale('log')
plt.xlabel('Frekvenser i log skala [Hz]')
plt.ylabel('Relativ amplitude [dB]')
plt.title('Frekvensspektrum i dB skala')
plt.ylim(-100, 10)
plt.legend(loc='upper right')
plt.grid()
plt.show()

#-------------------------------------------------------------
#Plotter PSD og regner ut SNR

Effekttetthetsspektrum = (abs(X_f)**2) #PSD
PSD_log = 10*np.log10(Effekttetthetsspektrum)
PSD_normalisert = PSD_log - np.max(PSD_log) #normalisering

signal_sum = 0
noise_sum = 0
 

for i in range(len(Effekttetthetsspektrum)):
    if  frekvenser[np.argmax(Effekttetthetsspektrum)]-10 < frekvenser[i] < frekvenser[np.argmax(Effekttetthetsspektrum)]+10:
        #print(f"frekvenskomponent: {Effekttetthetsspektrum[i]}")
        signal_sum += Effekttetthetsspektrum[i]
    else:
        noise_sum += Effekttetthetsspektrum[i]

SNR = 10*np.log10(np.abs(signal_sum / noise_sum))
print(f"SNR for en enkelt kanal {SNR} i dB")


plt.plot(frekvenser, PSD_normalisert)
plt.xscale('log')
plt.xlabel('Frekvenser i log skala [Hz]')
plt.ylabel('Relativ amplitude [db]')
plt.title('Effekttetthetsspektrum i dB skala')
#plt.ylim(-160, 20)
plt.grid()
plt.show()
