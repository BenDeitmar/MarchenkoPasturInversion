import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from math import ceil
from cmath import exp


import MarchenkoPasturInversion as MPI
import EigenInference_ElKaroui as EI_a

if __name__ == "__main__":

######################
    #hyperparameters

    #dimension and quotient c=d/n
    d = 10000
    c = 1/10

    PopEV = [0.5 for i in range(d//2)]+[0.5+i/d for i in range(d-d//2)]

    sig2=1

    d1,d2 = 0.5,1

    positions = np.arange(-0.2,1.2*sig2,0.005)
    #positions = np.arange(0,1.2*sig2,0.001)
    #positions = np.arange(0.49,1*sig2,0.005)
######################
    
    fig, ax = plt.subplots(1,1,layout='constrained')

    n = ceil(d/c)

    T = np.matrix(np.diag(np.sqrt(PopEV)))
    Y_matr = np.random.normal(size=(d,n))
    X_matr = T@Y_matr
    S_matr = X_matr@X_matr.H/n
    SampEV = np.linalg.eigh(S_matr)[0]

    ax.plot(positions,[sum([1 if x>lam else 0 for lam in PopEV])/d for x in positions],alpha=0.5,color='green',linewidth=3)
    
    positions,weights = MPI.H_Estimation(X_matr,dist=d1,positions=positions,sig2=sig2,N=10,options={},PopEV=None)
    ax.plot(positions,np.cumsum(weights),color='orange',alpha=0.5,linewidth=3)


    positions,weights = MPI.H_Estimation(X_matr,dist=d2,positions=positions,sig2=sig2,N=10,options={},PopEV=None)
    ax.plot(positions,np.cumsum(weights),color='orange',alpha=0.5,linewidth=3,linestyle='dashed')

    
    positions,weights = EI_a.H_Estimation(X_matr,sig2=sig2,positions=positions,options={})
    ax.plot(positions,np.cumsum(weights),color='blue',alpha=0.4,linewidth=3)




    plt.show()


