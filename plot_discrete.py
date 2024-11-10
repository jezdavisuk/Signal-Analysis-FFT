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