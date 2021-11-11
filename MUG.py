import numpy as np
import numba as nb
import matplotlib.pyplot as plt

@nb.njit
def strategy_update(payoff_arr, strategy_arr, Z, beta, mu, eps=0.05):

    pl = np.random.choice(Z, 2)
    plCentral = pl[0]
    
    if np.random.random()<mu:
        strategy_arr[plCentral,0] = np.random.random()
        strategy_arr[plCentral,1] = np.random.random()
    else:
        payoff1 = payoff_arr[plCentral]
        payoff2 = payoff_arr[pl[1]]
    
    prob = 1./(1+np.exp((payoff1-payoff2)*beta))
    
    if np.random.random()<prob:
        strategy_arr[plCentral,:] = strategy_arr[pl[1],:] + eps*(np.random.random()-0.5)
        
    return strategy_arr


@nb.njit
def single_MUG(strategy_arr, N, M, Z):

    payoff_arr = np.zeros((Z,))
    acceptance_arr = np.zeros((N,1))
    selP = np.random.choice(Z, N, replace=False)
    RespP = np.zeros((N-1,))
    index = -1
    for PropP in selP:
        index = index + 1
        RespP = np.delete(selP, index)
        p = strategy_arr[PropP,0]
        acceptInd = p>=strategy_arr[RespP,1]
        acceptancerate = np.sum(acceptInd)
        if acceptancerate>=M:
            payoff_arr[PropP] = payoff_arr[PropP] + (1-p)
            payoff_arr[RespP] = payoff_arr[RespP] + p/(N-1)
        else:
            pass
    return(payoff_arr)


if __name__ == "__main__":

    import time
    
    Z = 1024
    gen = 4000
    nMUG = 10000
    N = 9
    M = 7
    beta = 10

    #strategies = np.array([[0.5,0.5],[0.6,0.1],[0.2,0.1],[0.4,0.2]])
    strategies = np.random.random((Z,2))
    tt = time.time()
    for g in range(gen):
        payoffs = np.zeros((Z,))
        for i in range(nMUG):
            payoff = single_MUG(strategies, N, M, Z)
            payoffs = payoffs + payoff
        strategies = strategy_update(payoffs, strategies, Z, beta, 1./Z)
        print(g)
        print(np.mean(strategies, axis=0))
    print(time.time()-tt)
    print(payoffs)
