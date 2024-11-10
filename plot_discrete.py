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

