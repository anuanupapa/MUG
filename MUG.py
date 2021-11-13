import numpy as np
import numba as nb
import matplotlib.pyplot as plt

'''
#Implement B.C.
1. Reflective
2. Damping
3. Random
4. Cyclic
'''
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
    
    #prob = 1./(1+np.exp((payoff1-payoff2)*beta))
    if np.random.random()<1./(1+np.exp((payoff1-payoff2)*beta)):

        p_copy = strategy_arr[pl[1],0] + 2*eps*(np.random.random()-0.5)
        if p_copy <= 1 and p_copy >= 0:
            strategy_arr[plCentral,0] = p_copy
        elif p_copy > 1:
            strategy_arr[plCentral,0] = 2 - p_copy            
        else:
            strategy_arr[plCentral,0] = -1*p_copy
            
        q_copy = strategy_arr[pl[1],1] + 2*eps*(np.random.random()-0.5)
        if q_copy <= 1 and q_copy >= 0:
            strategy_arr[plCentral,1] = q_copy
        elif q_copy > 1:
            strategy_arr[plCentral,1] = 2 - q_copy            
        else:
            strategy_arr[plCentral,1] = -1*q_copy
            
    else:
        pass

    return strategy_arr


#Implement B.C
@nb.njit
def strategy_update_all_async(payoff_arr, strategy_arr,
                               Z, beta, mu, Centre, eps=0.05):

    #pl = np.random.choice(Z, 2)
    #plCentral = pl[0]
    plCentral = Centre
    pl = np.random.choice(Z, 2)
    
    if np.random.random()<mu:
        strategy_arr[plCentral,0] = np.random.random()
        strategy_arr[plCentral,1] = np.random.random()
    else:
        payoff1 = payoff_arr[plCentral]
        payoff2 = payoff_arr[pl[1]]

    #prob = 1./(1+np.exp((payoff1-payoff2)*beta))
    if np.random.random()<1./(1+np.exp((payoff1-payoff2)*beta)):

        p_copy = strategy_arr[pl[1],0] + 2*eps*(np.random.random()-0.5)
        if p_copy <= 1 and p_copy >= 0:
            strategy_arr[plCentral,0] = p_copy
        elif p_copy > 1:
            strategy_arr[plCentral,0] = 2 - p_copy            
        else:
            strategy_arr[plCentral,0] = -1*p_copy
            
        q_copy = strategy_arr[pl[1],1] + 2*eps*(np.random.random()-0.5)
        if q_copy <= 1 and q_copy >= 0:
            strategy_arr[plCentral,1] = q_copy
        elif q_copy > 1:
            strategy_arr[plCentral,1] = 2 - q_copy            
        else:
            strategy_arr[plCentral,1] = -1*q_copy
            
    else:
        pass
        
    return strategy_arr


# Playing a single MUG in a group of N selected from Z individuals with a minimum of M(<N) required for the proposal to pass.
@nb.njit
def single_MUG(strategy_arr, N, M, Z):

    payoff_arr = np.zeros((Z,))
    acceptance_arr = np.zeros((N,1))

    # Select N from the Z players
    selP = np.random.choice(Z, N, replace=False)
    RespP = np.zeros((N-1,))

    # Cycle through the proposals of each player in the group
    index = -1
    for PropP in selP:
        index = index + 1
        RespP = np.delete(selP, index) #Responders
        p = strategy_arr[PropP,0]
        acceptInd = p>=strategy_arr[RespP,1] #array of True False
        acceptancerate = np.sum(acceptInd)
        if acceptancerate>=M: #M is required for group acceptance
            payoff_arr[PropP] = payoff_arr[PropP] + (1-p)
            payoff_arr[RespP] = payoff_arr[RespP] + p/(N-1)
        else:
            pass
    return(payoff_arr)


if __name__ == "__main__":

    import time
    tt = time.time()

    '''
    np.random.seed(1)
    # MUG check
    strategies = np.array([[0.5,0.5],[0.6,0.1],[0.2,0.1],[0.4,0.2]])
    payoff = single_MUG(strategies, 3, 2, 4)
    print(payoff)    
    '''
    
    Z = 1000
    gen = 4000
    nMUG = 10000
    N = 15
    M = 1
    beta = 10
    strategy_space1D = np.linspace(0,1,101)
    strategies = np.random.choice(strategy_space1D, (Z,2))
    strategy_evolution = np.zeros((gen,Z,2))
    payoff_evolution = np.zeros((gen,Z))
    
    for g in range(gen):
        payoffs = np.zeros((Z,))
        
        for i in range(nMUG):
            payoff = single_MUG(strategies, N, M, Z)
            payoffs = payoffs + payoff    
        
        #Everyone is getting a chance to update
        playas = np.arange(0,Z,1)
        np.random.shuffle(playas)
        
        for i in playas:
            strategies = strategy_update_all_async(
                payoffs, strategies, Z, beta, 1./Z, i)
        '''
        strategies = strategy_update(
            payoffs, strategies, Z, beta, 1./Z)
        '''
        #Store data
        strategy_evolution[g,:,:] = strategies
        payoff_evolution[g,:] = payoffs        
        print(g)
        #print(np.mean(strategies, axis=0))

    print(time.time()-tt)

    strategy_mean = np.mean(strategy_evolution, axis=1)
    strategy_std = np.std(strategy_evolution, axis=1)
    t_arr = np.arange(1,gen+1,1)
    
    plt.plot(t_arr, strategy_mean[:,0],'bo-')
    plt.fill_between(t_arr,
                     strategy_mean[:,0]-strategy_std[:,0],
                     strategy_mean[:,0]+strategy_std[:,0],
                     alpha=0.3, color="b")
    
    plt.plot(t_arr, strategy_mean[:,1],'go-')
    plt.fill_between(t_arr,
                     strategy_mean[:,1]-strategy_std[:,1],
                     strategy_mean[:,1]+strategy_std[:,1],
                     alpha=0.3, color="g")

    plt.title("M="+str(M)+" N="+str(N)+" Z="+str(Z))
    plt.show()
    
    '''
    np.savez("MUGdata_AllImitate_n50000_beta1000_N10_M1.npz",
             Z=Z, generation=gen, nMUG=nMUG,
             GroupSize=N, GroupCutoff=M, beta=beta,
             payoffs=payoff_evolution, strategies=strategy_evolution)
    '''
