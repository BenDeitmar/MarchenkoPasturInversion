import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from math import ceil
from cmath import exp

import MarchenkoPasturInversion as MPI



if __name__ == "__main__":

######################
    #hyperparameters

    #dimension and quotient c=d/n
    d_List = [i*10 for i in range(1,101)]
    c = 1/10

    sig2=1

    #defining oracle curve
    gammaLeft,gammaTop,gammaRight = -0.1, 0.5, 1.5
######################
    
    fig, ax = plt.subplots(1,2,layout='constrained')

    ax[0].set_title('f(z) = z^5')
    ax[1].set_title('f(z) = exp(z)')

    f1 = lambda x: x**5
    f2 = lambda x: exp(x)

    Diff1_f1_List = []
    Diff2_f1_List = []
    Diff1_f2_List = []
    Diff2_f2_List = []
    for d in d_List:
        print('d=',d)
        n = ceil(d/c)
        PopEV = [0.5 for i in range(d//2)]+[0.5+i/d for i in range(d-d//2)]
        ActualIntegral1 = sum([f1(lam) for lam in PopEV])/d
        ActualIntegral2 = sum([f2(lam) for lam in PopEV])/d

        dist = 4*sig2*(1+c)
        gamma1 = MPI.makeGamma_Legendre(gammaLeft,gammaTop,gammaRight,N=10,show=False)
        gamma2 = MPI.makeGamma_Legendre(-dist,dist,sig2+dist,N=10,show=False)

        T = np.matrix(np.diag(np.sqrt(PopEV)))
        Y_matr = np.random.normal(size=(d,n))
        X_matr = T@Y_matr
        S_matr = X_matr@X_matr.H/n
        SampEV = np.linalg.eigh(S_matr)[0]

        sValues1 = MPI.Vectorwise_MPI(np.array([tup[0] for tup in gamma1]),SampEV,d,n)
        sValues2 = MPI.Vectorwise_MPI(np.array([tup[0] for tup in gamma2]),SampEV,d,n)

        I1f1 = MPI.CurveIntegral(f1,gamma1,sValues1)
        I2f1 = MPI.CurveIntegral(f1,gamma2,sValues2)

        I1f2 = MPI.CurveIntegral(f2,gamma1,sValues1)
        I2f2 = MPI.CurveIntegral(f2,gamma2,sValues2)

        Diff1_f1_List.append(abs(I1f1-ActualIntegral1))
        Diff2_f1_List.append(abs(I2f1-ActualIntegral1))

        Diff1_f2_List.append(abs(I1f2-ActualIntegral2))
        Diff2_f2_List.append(abs(I2f2-ActualIntegral2))

    ax[0].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)
    ax[1].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)

    ax[0].plot(d_List,Diff1_f1_List,color='blue',alpha=0.3)
    ax[0].plot(d_List,Diff2_f1_List,color='blue',linestyle='dashed',alpha=0.3)

    ax[1].plot(d_List,Diff1_f2_List,color='blue',alpha=0.3)
    ax[1].plot(d_List,Diff2_f2_List,color='blue',linestyle='dashed',alpha=0.3)

    plt.show()


