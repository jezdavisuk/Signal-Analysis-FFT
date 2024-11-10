import numpy as np
import matplotlib.pyplot as plt

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