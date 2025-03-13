import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sc
from scipy.fft import rfft, rfftfreq


# =================== Variables ===================
filename = "Puls_refl_5.txt" #Pulse_test_2.txt
#filnavn_lst = ["Puls_hvile_1.txt", "Puls_hvile_2.txt", "Puls_hvile_3.txt",  "Puls_hvile_4.txt", "Puls_hvile_5.txt"] #Transmittans
filnavn_lst = ["Puls_refl_1.txt", "Puls_refl_2.txt", "Puls_refl_3.txt",  "Puls_refl_4.txt", "Puls_refl_5.txt"] #Reflektans
fs = 30 #funnet ved å ta lengden på data og dele på tiden spilt inn, også mulig å lese av i terminalen etter kjørt roi.py filen

# =================== Import text file ===================
def file_to_data(filnavn):
    data = np.loadtxt(filnavn)
    data = data - np.mean(data,axis=0)
    #Split into channels for Red, Green and Blue
    #Full rådata lengde -----
    """ Red = data[:,0]
    Green = data[:,1]
    Blue = data[:,2] """
    #Kutter rådata lengde for å unngå lave frekvenser -----
    Red = data[300:450,0]
    Green = data[300:450,1]
    Blue = data[300:450,2]
    #Create a time axis
    #time = np.arange(len(data))
    data_kanaler = [Red, Green, Blue]
    return data_kanaler

# Lager x aksen vi bruker når vi plotter fft
def data_to_bpm(data_kanal):
    signallengde_frames = len(data_kanal)
    frames = np.arange(0, signallengde_frames)
    signallengde_tid = signallengde_frames/30 #fps
    frekvens_resolution = 1/signallengde_tid # 1/sek = Hz
    freqs = frekvens_resolution*frames #datapunkter i frames*Hz
    bpm = freqs*60 #konverterer fra frekvens til bmp
    bpm = bpm[:int(len(bpm)/2)]
    return bpm


# =================== Filtrere data med båndpass Butterworth filter ===================
def digitalt_filter(N, low_freq, high_freq, data_2B_filtered): #N er orden på filteret 
    nyquist = fs / 2
    low = low_freq / nyquist  #Normaliserer frekvensen
    high = high_freq / nyquist  
    filter_coefficients = sc.butter(N, [low, high], "bandpass") #python tupple with coeffs
    filtrert_data = sc.lfilter(filter_coefficients[0], filter_coefficients[1], data_2B_filtered)
    return filtrert_data #er et array


# Lavpassfilter
""" def digitalt_filter(high_freq, data_2B_filtered): 
    N = 4 #N er orden på filteret 
    nyquist = fs / 2  #Normaliserer frekvensen
    high = high_freq / nyquist  
    filter_coefficients = sc.butter(N, high, "low") #python tupple with coeffs
    filtrert_data = sc.lfilter(filter_coefficients[0], filter_coefficients[1], data_2B_filtered)
    return filtrert_data #er et array
 """

# =================== Regne ut puls med FFT ===================
def fft(data_kanal):
    #filter data
    #data_kanal = digitalt_filter(4, 0.3, 3, data_kanal)
    Nfft = len(data_kanal)
    data_fft = np.fft.fft(data_kanal, Nfft)
    #data_fft = digitalt_filter(4, 0.3, 3, data_fft)
    data_fft = data_fft[:int(len(data_fft)/2)]
    return np.abs(data_fft)

def find_puls_fft(data_kanal): #denne på jobbes litt mer med, den finner ikke puls nå uten å gjøre mer matematikk
    X_f = fft(data_kanal)
    #X_f = X_f[:len(data_kanal)//2]
    pulse_index = np.argmax(X_f)
    bpm = data_to_bpm(data_kanal)
    pulse = bpm[pulse_index]
    #print (f"dette er pulsen: {pulse}")
    return pulse



# =================== Regne ut puls med autocorrelasjon ===================
def autocorrelasjon(data_kanal):
    data_autocorr = sc.correlate(data_kanal, data_kanal, mode = 'full')
    data_autocorr = data_autocorr / np.max(data_autocorr)
    return data_autocorr

def find_puls_autokorr(data_kanal): #denne på jobbes litt mer med, den finner ikke puls nå uten å gjøre mer matematikk
    rxy = autocorrelasjon(data_kanal)
    peaks_indices = sc.find_peaks(rxy,height = 0.4, threshold=None)
    pulse = 1
    return pulse



# =================== Lage vektor med pulser ===================

def generate_pulse_vec(filnavn_liste, color_index): #data_kanaler[0] gir Rød, 0 = rød, 1 = grønn, 2 = blå
    puls_vec = []
    for filnavn in filnavn_liste: #5 ulike målinger
        data_kanaler = file_to_data(filnavn)
        puls_vec.append(find_puls_fft(data_kanaler[color_index])) 
    return puls_vec



# =================== Regne ut puls med standardavvik og varians av puls ===================
def standardavvik(puls_vec):
    std = np.std(puls_vec, ddof=1)
    return std


def varians(pulse_vec):
    var = np.var(pulse_vec)
    return var

def gjennomsnitt(pulse_vec):
    summen = np.sum(pulse_vec)
    snitt = summen/ len(pulse_vec)
    return snitt

# =================== Regner ut PSD og SNR===================


def PSD(data_kanal):
    X= fft(data_kanal)
    Effekttetthetsspektrum = (abs(X)**2) #PSD
    PSD_log = 10*np.log10(Effekttetthetsspektrum)
    PSD_normalisert = PSD_log - np.max(PSD_log) #normalisering
    PSD = [Effekttetthetsspektrum, PSD_normalisert]
    return PSD


def SNR(data_kanal):

    signal_sum = 0
    noise_sum = 0
    Effekttetthetsspektrum , norm = PSD(data_kanal)
    bpm = data_to_bpm(data_kanal)
    

    for i in range(len(Effekttetthetsspektrum)):
        if  np.argmax(Effekttetthetsspektrum)-50 < Effekttetthetsspektrum[i] < np.argmax(Effekttetthetsspektrum)+50:
            signal_sum += Effekttetthetsspektrum[i]
        else:
            noise_sum += Effekttetthetsspektrum[i]

    print(signal_sum)
    print(noise_sum)

    SNR = 10*np.log10(np.abs(signal_sum / noise_sum))
    return SNR
#bruke gjennomsnitt og peak



r,g,b = file_to_data(filename)

#FUNKSJONER FOR Å PLOTTE DATA ================
#Plot raw data
def plot_rådata():
    plt.plot(r, "r")
    plt.plot(g, "g")
    plt.plot(b, "b")
    plt.show()

#Plot raw data and takes inn filename
def plot_rådata_nr2(filename):
    r,g,b = file_to_data(filename)
    #plt.plot(r, "r")
    plt.plot(g, "g")
    #plt.plot(b, "b")
    plt.show()


#Plot av filtrert data
def plot_filtrert_data():
    Red_filtrert = digitalt_filter(3, r)
    Green_filtrert = digitalt_filter(3, g)
    Blue_filtrert = digitalt_filter(3, b)

    #plt.plot(Red_filtrert, "r")
    plt.plot(Green_filtrert, "g")
    #plt.plot(Blue_filtrert, "b")
    plt.show()


#Plot FFT
def plot_FFT(data_kanal):  
    bpm = data_to_bpm(data_kanal)
    
    plt.plot(bpm, fft(r), "r")
    plt.plot(bpm, fft(g), "g")
    plt.plot(bpm, fft(b), "b")
    plt.show()

# Plot fft with input filename
def plot_FFT_nr2(filename):
    r, g, b = file_to_data(filename)  
    bpm = data_to_bpm(r)
    
    #plt.plot(bpm, fft(r), "r")
    plt.plot(bpm, fft(g), "g")
    #plt.plot(bpm, fft(b), "b")
    plt.show()




#Plot Autocorrelasjon
def plot_autocorr(data_kanal1, data_kanal2, data_kanal3): #her kan man ta inn rå eller filtrert data
    Green_autocorr = autocorrelasjon(data_kanal1)
    Blue_autocorr = autocorrelasjon(data_kanal2)
    Red_autocorr = autocorrelasjon(data_kanal3)

    plt.plot(Blue_autocorr, "b")
    plt.plot(Red_autocorr, "r")
    plt.plot(Green_autocorr, "g")
    plt.show()

# *=*=*=*=*=*=*=*=*=*=*=*=*= Kjører funkjsonene under her *=*=*=*=*=*=*=*=*=*=*=*=*=
#DEFINER FILNAVNENE VI TESTER HER
#filnavn_lst = ["Puls_hvile_1.txt", "Puls_hvile_2.txt", "Puls_hvile_3.txt",  "Puls_hvile_4.txt", "Puls_hvile_5.txt"] #Transmittans
#filnavn_lst = ["Puls_refl_1.txt", "Puls_refl_2.txt", "Puls_refl_3.txt",  "Puls_refl_4.txt", "Puls_refl_5.txt"] #Reflektans
filnavn_lst = ["Puls_varm_1.txt", "Puls_varm_2.txt", "Puls_kald_1.txt",  "Puls_kald_2.txt", "Puls_run_1.txt", "Puls_run_2.txt"] #Robusthetstest

#Kjør for å plotte rådata til alle kanalene =======================
""" for filnavn in filnavn_lst:
    plot_rådata_nr2(filnavn) """

#Kjør for å plotte fft til alle kanalene ==========================
""" for filnavn in filnavn_lst:
    plot_FFT_nr2(filnavn) """

#Kjør for å regne ut puls til hver av kanalene ====================
rød_vec = generate_pulse_vec(filnavn_lst, 0)
grønn_vec =  generate_pulse_vec(filnavn_lst, 1)
blå_vec =  generate_pulse_vec(filnavn_lst, 2)

print(f"Pulser-Rød: {rød_vec} \n Pulser-Grønn: {grønn_vec} \n Pulser-Blå: {blå_vec} \n")

#Kjør for å plotte freq spekter med kuttet data ===================
#plot_FFT_nr2("Puls_hvile_1.txt")

# Plott PSD for alle kanalene
for filnavn in filnavn_lst:
    r, b, g = file_to_data(filnavn)
    plt.plot(PSD(r))
    plt.show()
    plt.plot(PSD(g))
    plt.show()
    plt.plot(PSD(b))
    plt.show()

#Kjør For å regne ut SNR, snitt og std ============================
""" snr = SNR(g)
print(f"SNR = {snr:.2f} dB")

rød_vec = generate_pulse_vec(filnavn_lst, 0)
rød_snitt = gjennomsnitt(rød_vec)
rød_std = standardavvik(rød_vec)
print(f"for rød kanal: snitt: {rød_snitt}, std: {rød_std}.")

grønn_vec =  generate_pulse_vec(filnavn_lst, 1)
grønn_snitt = gjennomsnitt(grønn_vec)
grønn_std = standardavvik(grønn_vec)
print(f"for grønn kanal: snitt: {grønn_snitt}, std: {grønn_std}.")

blue_vec =  generate_pulse_vec(filnavn_lst, 2)
blue_snitt = gjennomsnitt(blue_vec)
blue_std = standardavvik(blue_vec)
print(f"for blue kanal: snitt: {blue_snitt}, std: {blue_std}.")

rød_snr = SNR(r)
grønn_snr = SNR(g)
blue_snr = SNR(b)
print(f"SNR for de tre kanalene rød, grønn og blå for måling (Puls_hvile_1.txt) er hhv: \n Rød: {rød_snr} \n Grønn:  {grønn_snr} \n Blå:  {blue_snr} .") 
 """

