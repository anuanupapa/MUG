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
    runs = 25
    Z = 100
    gen = 1000
    nMUG = 1000
    N = 12
    beta = 10
    N_arr = np.arange(2,16,1)
    strategy_evolution = np.zeros((len(N_arr),runs,gen,Z,2))
    payoff_evolution = np.zeros((len(N_arr),runs,gen,Z))

    for N in N_arr:
        M = 1

        for run in range(runs):
            strategy_space1D = np.linspace(0,1,101)
            strategies = np.random.choice(strategy_space1D, (Z,2))

            for g in range(gen):
                payoffs = np.zeros((Z,))
                print("trial:", run, " gen:", g)

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
                #Evolutionary dynamics timescale 
                #much longer than MUG timescale
                strategies = strategy_update(
                    payoffs, strategies, Z, beta, 1./Z)
                '''
                #Store data
                strategy_evolution[N-2,run,g,:,:] = strategies
                payoff_evolution[N-2,run,g,:] = payoffs

        strategy_mean = np.mean(np.mean(
            strategy_evolution[N-2,:,:,:,:], axis=2), axis=0)
        strategy_std = np.std(np.mean(
            strategy_evolution[N-2,:,:,:,:], axis=2), axis=0)
        t_arr = np.arange(1,gen+1,1)

        plt.clf()
        plt.plot(t_arr, strategy_mean[:,0],'bo-', label='<p>')
        plt.fill_between(t_arr,
                         strategy_mean[:,0]-strategy_std[:,0],
                         strategy_mean[:,0]+strategy_std[:,0],
                         alpha=0.3, color="b")
    
        plt.plot(t_arr, strategy_mean[:,1],'go-', label='<q>')
        plt.fill_between(t_arr,
                         strategy_mean[:,1]-strategy_std[:,1],
                         strategy_mean[:,1]+strategy_std[:,1],
                         alpha=0.3, color="g")

        plt.legend()
        plt.title("M="+str(M)+" N="+str(N)+" Z="+str(Z))
        plt.savefig("Z100_Mmin/pblue-qgreen"
                    +"_N"+str(N)+"_M"+str(M)+"_Z"+str(Z)
                    +"allupdate.png")
        
    np.savez("MUG-varyN_AllImitate_Z"+str(Z)+"_n"+str(nMUG)+
             "_beta"+str(beta)+"_Mmin.npz",
             Z=Z, generation=gen, nMUG=nMUG,
             GroupSize=N_arr, GroupCutoff_arr=N_arr-1, beta=beta,
             payoffs=payoff_evolution, strategies=strategy_evolution)
    
    print(time.time()-tt)
    plt.clf()
    strategyM_mean = np.mean(np.mean(np.mean(
        strategy_evolution[:,:,-250:,:], axis=1), axis=1), axis=1)
    plt.plot(N_arr, strategyM_mean[:,0], 'bo-', label="<<<p>>>")
    plt.plot(N_arr, strategyM_mean[:,1], 'go-', label="<<<q>>>")
    plt.title("Z="+str(Z)+" N="+str(N)+" min cutoff")
    plt.xlabel("N - group size")
    plt.legend()
    plt.savefig("Z100_Mmin_qpVSN.png")
    plt.show()
