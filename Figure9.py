import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from math import ceil
from cmath import exp
import time

import MarchenkoPasturInversion as MPI
import EigenInference_KongValiant as EI_c

if __name__ == "__main__":

######################
    #hyperparameters

    #dimension and quotient c=d/n
    d_List = [10*i for i in range(1,31)]
    c = 1/10

    #Moment to be estimated
    K = 5

    sig2=1

    N = 50
######################
    
    fig, ax = plt.subplots(1,2,layout='constrained')

    f = lambda z: z**K

    ax[0].set_title('Estimation accuracy')
    ax[1].set_title('Time in seconds')

    dist = 4*sig2*(1+c)
    dist = 0.5
    gamma = MPI.makeGamma_Legendre(-dist,dist,sig2+dist,N=10,show=False)

    AvgDiff_List_MPI = []
    AvgDiff_List_c = []
    AvgTime_List_MPI = []
    AvgTime_List_c = []
    for d in d_List:
        print('d=',d)
        PopEV = [0.5 for i in range(d//2)]+[0.5+i/d for i in range(d-d//2)]
        n = ceil(d/c)

        ActualMoment = sum([f(lam) for lam in PopEV])/d

        SumDiff1 = 0
        SumDiff2 = 0
        TimeDiff1 = 0
        TimeDiff2 = 0


        for i in range(N):
            T = np.matrix(np.diag(np.sqrt(PopEV)))
            Y_matr = np.random.normal(size=(d,n))
            X_matr = T@Y_matr

            #our method
            start = time.time()
            S_matr = X_matr@X_matr.H/n
            SampEV = np.linalg.eigh(S_matr)[0]

            sValues = MPI.Vectorwise_MPI(np.array([tup[0] for tup in gamma]),SampEV,d,n)

            I = MPI.CurveIntegral(f,gamma,sValues)
            end = time.time()
            TimeDiff1 += end - start

            Diff = abs(I-ActualMoment)
            SumDiff1 += Diff

            #Kong-Valiant method (c)
            start = time.time()
            I = EI_c.MomentEstimator(X_matr,K)
            end = time.time()
            TimeDiff2 += end - start

            Diff = abs(I-ActualMoment)
            SumDiff2 += Diff

        AvgDiff_List_MPI.append(SumDiff1/N)
        AvgTime_List_MPI.append(TimeDiff1/N)
        AvgDiff_List_c.append(SumDiff2/N)
        AvgTime_List_c.append(TimeDiff2/N)


    ax[0].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)
    ax[1].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)

    ax[0].plot(d_List,AvgDiff_List_MPI,color='orange',alpha=0.3)
    ax[0].plot(d_List,AvgDiff_List_c,color='purple',alpha=0.3)

    ax[1].plot(d_List,AvgTime_List_MPI,color='orange',alpha=0.3)
    ax[1].plot(d_List,AvgTime_List_c,color='purple',alpha=0.3)

    plt.show()


