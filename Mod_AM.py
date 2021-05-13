# %%

from numpy.core.arrayprint import DatetimeFormat
from numpy.core.defchararray import mod
from numpy.core.function_base import linspace
from scipy import signal
from scipy.signal.filter_design import normalize
from suaBibSignal import *
# import peakutils  # alternativas  #from detect_peaks import *   #import pickle
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import time
from scipy.io import wavfile
import soundfile as sf


def calcFFT(signal, fs):
    # https://docs.scipy.org/doc/scipy/reference/tutorial/fftpack.html
    N = len(signal)
    W = window.hamming(N)
    T = 1/fs
    xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
    yf = fft(signal*W)
    return(xf, np.abs(yf[0:N//2]))


def generateSin(freq, amplitude, time, fs):
    n = time*fs
    x = np.linspace(0.0, time, n)
    s = amplitude*np.sin(freq*x*2*np.pi)
    return (x, s)


def filtro(y, samplerate, cutoff_hz):
  # https://scipy.github.io/old-wiki/pages/Cookbook/FIRFilter.html
    nyq_rate = samplerate/2
    width = 5.0/nyq_rate
    ripple_db = 60.0  # dB
    N, beta = sg.kaiserord(ripple_db, width)
    taps = sg.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    yFiltrado = sg.lfilter(taps, 1.0, y)
    return yFiltrado


def LPF(signal, cutoff_hz, fs):
    from scipy import signal as sg
    #####################
    # Filtro
    #####################
    # https://scipy.github.io/old-wiki/pages/Cookbook/FIRFilter.html
    nyq_rate = fs/2
    width = 5.0/nyq_rate
    ripple_db = 60.0  # dB
    N, beta = sg.kaiserord(ripple_db, width)
    taps = sg.firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
    return(sg.lfilter(taps, 1.0, signal))

# --> Audio gravado


fs = 48640
sound, sr = sf.read('SHEEESH.wav')

# sd.play(sound)
# sd.wait()

# --> Normalizando o áudio

normalizando = sound
maxi = np.max(normalizando)
min = np.min(normalizando)

for e, i in enumerate(normalizando):
    med = (maxi+min)/2
    ran = (maxi-min)/2
    normalizando[e] = (i-med)/ran


# --> Filtrando áudio

filtro = LPF(sound, 4000, fs)

# sd.play(filtro)
# sd.wait()

# --> Carrier

T = 4
t = np.linspace(0, T, fs)
X, Y = generateSin(20000, 1, T, fs)
carrier = Y[0:len(sound)]

# --> Modulação dos dados

modulado = filtro*carrier

# sd.play(modulado)
# sd.wait()

# --> Demodulação

min_x = np.min(X)
max_x = np.max(X)

z = 0

for e in range(0,16000):
    if Y[e] < 10:
        z += 1

if z > 15900 and np.max(X) < 24400:
    print("Sucesso! Sinal dentro da banda.")

demodulado = modulado*carrier

# --> Filtrou Demodulado

filtro_demod = LPF(demodulado, 3000, fs)

print("Comparando os sons:")

print("Áudio FILTRADO:")

sd.play(filtro)
sd.wait()

print("Áudio FILTRADO escutado.")

print("Áudio DEMODULADO FILTRADO:")

sd.play(filtro_demod)
sd.wait()

print("Áudio DEMODULADO FILTRADO escutado.")

#----------------------- GRÁFICOS --------------------------------------

# SINAL DE ÁUDIO ORIGINAL


plt.figure()
plt.plot(sound, color="red")
plt.title("Sinal do áudio original no tempo")
plt.xlim(0, 200000)

plt.figure()
X, Y = calcFFT(sound, fs)
plt.plot(X, np.abs(Y), color="red")
plt.xlim(0, 9000)
plt.title("Sinal do áudio original no domínio da frequência (Fourier)")

# SINAL DE ÁUDIO NORMALIZADO

plt.figure()
plt.plot(normalizando)
plt.title("Sinal do áudio normalizado no tempo")
plt.xlim(0, 200000)

plt.figure()
X, Y = calcFFT(normalizando, fs)
plt.plot(X, np.abs(Y))
plt.xlim(0, 9000)
plt.title("Sinal do áudio normalizado no domínio da frequência (Fourier)")

# SINAL DE ÁUDIO FILTRADO

plt.figure()
plt.plot(filtro, color="magenta")
plt.title("Sinal do áudio filtrado no tempo")
plt.xlim(0, 200000)

plt.figure()
X, Y = calcFFT(filtro, fs)
plt.plot(X, np.abs(Y), color="magenta")
plt.xlim(0, 9000)
plt.title("Sinal do áudio filtrado no domínio da frequência (Fourier)")

# SINAL DE ÁUDIO MODULADO

plt.figure()
plt.plot(modulado, color="yellow")
plt.title("Sinal do áudio modulado no tempo")
plt.xlim(0, 200000)

plt.figure()
X, Y = calcFFT(modulado, fs)
plt.plot(X, np.abs(Y), color="yellow")
plt.xlim(15000, 25000)
plt.title("Sinal do áudio modulado no domínio da frequência (Fourier)")
plt.show()

plt.figure()
plt.plot(demodulado, color="green")
plt.title("Sinal do áudio demodulado no tempo")
plt.xlim(0, 200000)

plt.figure()
X, Y = calcFFT(demodulado, fs)
plt.plot(X, np.abs(Y), color="green")
plt.xlim(0, 7500)
plt.title("Sinal do áudio demodulado no domínio da frequência (Fourier)")
plt.show()




# %%
