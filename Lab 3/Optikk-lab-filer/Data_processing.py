import matplotlib.pyplot as plt
import numpy as np

#Import text file
data = np.loadtxt("Puls_test_1.txt")
data = data - np.mean(data,axis=0)
#Split into channels for Red, Green and Blue
Red = data[:,0]
Green = data[:,1]
Blue = data[:,2]
#Create a time axis
time = np.arange(len(data))

#Filtrere data??

def fft(data_kanal):
    Nfft = len(data_kanal)
    data_fft = np.fft.fft(data_kanal, Nfft)
    return data_fft

Green_fft = fft(Green)

plt.plot(Green_fft)
#plt.plot(Red, "r")
#plt.plot(Green, "g")
#plt.plot(Blue, "b")
plt.show()