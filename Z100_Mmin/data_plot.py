import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = np.load("MUG-varyN_AllImitate_Z100_n1000_beta10_Mmin.npz")
P_pq_t = data["strategies"]
pay_pq_t = data["payoffs"]
N_arr = data["GroupSize"]

print(np.shape(P_pq_t))

Pmean_pq_t = np.mean(np.mean(np.mean(
    P_pq_t[:,:,-250:,:,:], axis=1), axis=1), axis=1)

print(np.shape(Pmean_pq_t))

plt.plot(N_arr, Pmean_pq_t[:,0], 'bo-',
         label=r'$<<<p>_{i=1}^{Z}>_{t=750}^{1000}>_{trial=1}^{25}$')
plt.plot(N_arr, Pmean_pq_t[:,1], 'go-',
         label=r'$<<<q>_{i=1}^{Z}>_{t=750}^{1000}>_{trial=1}^{25}$')
plt.legend(framealpha=0)
plt.xlabel("N - group size")
plt.title("M=1 Z=100")
plt.savefig("Z100_Mmin_qpVSN.png")
plt.show()
