from json.encoder import ESCAPE_DCT

import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft, ifft, fftfreq, fftshift

def discrete_quadratic(T,N,tN):
    # create an array of zeros of length N
    V = np.zeros(N, dtype=np.float32)

# discretise time
    for k in range(0,N):
        dt = tN / N     # define time-step
        tk = k * dt     # discrete time an integer multiple of time-step

        # define V(t) on t in [0,2]
        if (tk % T) < T / 2:
            V[k] = (tk % T) ** 2
        else:
            V[k] = ((tk % T) - T) ** 2
    return V

tN = 8.0    # sampling time
samplerate = 512
N = int(tN * samplerate)    # total number of data points sampled

T = 2   # fundamental period

t = np.linspace(0,tN,N)     # define the time array
V = discrete_quadratic(T,N,tN)    # construct discretised signal

plt.plot(t,V)
plt.title('V(t) as a function of time t')
plt.xlabel('t (s)')
plt.ylabel('V(t)')
plt.xlim(0.0,8.0)   # set t to span sample time 0 to 8 seconds

# output file ./graphics/sig_disc.png
plt.show()

# evaluate FFT of signal V returns vector of Fourier coefficients
# option norm=forward to account for 1/N normalisation for discrete Fourier Transforms
V_fft = fft(V, norm='forward')

# set total number of data points
N = 4096

# define array for n ascending between -2048 and 2047
n = np.arange(-N/2,N/2,1)

# set conditions for positive and negative n
n1 = n > 0
n2 = n < 0

# define DFT array
V_tilde_n = np.zeros(N)

# set entry for n=0 equal to 1/3
V_tilde_n[n == 0] = 1/3

# set analytically-derived entries for positive and negative n, excluding static n=0
V_tilde_n[n1] = (2 * (-1) ** n[n1]) / (np.pi ** 2 * n[n1] ** 2)
V_tilde_n[n2] = (2 * (-1) ** n[n2]) / (np.pi ** 2 * n[n2] ** 2)

# define frequency arrays
f_num = n / tN
f_analytic = n / T

# map Fourier coefficients to discrete frequencies and plot against real-valued solution
plt.scatter(f_analytic, np.real(V_tilde_n), color='red', label='Exact')
plt.plot(f_num, fftshift(np.real(V_fft)), color='blue', label='Numerical')
plt.xlim(-5, 5)
plt.ylim(-0.23, 0.35)
plt.xlabel('Frequency, f')
plt.ylabel('Real part of Fourier Transforms')
plt.title('Contrasting analytical and numerical DFTs by their real parts')
plt.legend()

## output file ./graphics/dft_real.png
plt.show()

# map Fourier coefficients to discrete frequencies and plot against imaginary-valued solution
plt.scatter(f_analytic, np.imag(V_tilde_n), color='red', label='Exact')
plt.plot(f_num, fftshift(np.imag(V_fft)), color='blue', label='Numerical')
plt.xlim(-5, 5)
plt.ylim(-0.1, 0.1)
plt.xlabel('Frequency, f')
plt.ylabel('Imaginary part of Fourier Transforms')
plt.title('Contrasting analytical and numerical DFTs by their imaginary parts')
plt.legend()

## output file ./graphics/dft_imag.png
plt.show()

def dominant_freq(samplerate, V):
    N = len(V)  # total data points in discretised V(t)

    # fast Fourier transform on signal V(t)
    V_fft = fft(V, norm='forward')

    # sampling time
    tN = N / samplerate

    # multiply each Fourier coefficient of the FFT by its conjugate to give the ESD
    ESD = np.real(V_fft * np.conjugate(V_fft))

    # shift zero-frequency component to centre
    ESD_shift = fftshift(ESD)

    # extract index for maximum ESD, non-negative frequencies only
    ESD_max = ESD_shift[int(N/2):].argmax(axis=0) + int(N/2)

    # define the frequency array
    n = np.arange(-N/2,N/2,1)
    f = n / tN

    # match frequency to index n at max ESD
    dom_freq = f[ESD_max]

    return dom_freq