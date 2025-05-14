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
    d_List = [i*10 for i in range(1,51)]
    c = 1/10

    sig2=1

    N = 50
######################
    
    fig, ax = plt.subplots(1,2,layout='constrained')

    ax[0].set_title('f(z) = z^5')
    ax[1].set_title('f(z) = exp(z)')

    f1 = lambda x: x**5
    f2 = lambda x: exp(x)

    dist = 4*sig2*(1+c)
    gamma = MPI.makeGamma_Legendre(-dist,dist,sig2+dist,N=10,show=False)

    AvgDiff_f1_List = []
    AvgDiff_f2_List = []
    MaxDiff_f1_List = []
    MaxDiff_f2_List = []
    for d in d_List:
        print('d=',d)
        n = ceil(d/c)
        PopEV = [0.5 for i in range(d//2)]+[0.5+i/d for i in range(d-d//2)]
        ActualIntegral1 = sum([f1(lam) for lam in PopEV])/d
        ActualIntegral2 = sum([f2(lam) for lam in PopEV])/d

        SumDiff1 = 0
        SumDiff2 = 0
        MaxDiff1 = 0
        MaxDiff2 = 0
        for i in range(N):
            T = np.matrix(np.diag(np.sqrt(PopEV)))
            Y_matr = np.random.normal(size=(d,n))
            X_matr = T@Y_matr
            S_matr = X_matr@X_matr.H/n
            SampEV = np.linalg.eigh(S_matr)[0]

            sValues = MPI.Vectorwise_MPI(np.array([tup[0] for tup in gamma]),SampEV,d,n)

            I_f1 = MPI.CurveIntegral(f1,gamma,sValues)
            I_f2 = MPI.CurveIntegral(f2,gamma,sValues)

            Diff1 = abs(I_f1-ActualIntegral1)
            Diff2 = abs(I_f2-ActualIntegral2)

            SumDiff1 += Diff1
            SumDiff2 += Diff2

            if Diff1 > MaxDiff1:
                MaxDiff1 = Diff1
            if Diff2 > MaxDiff2:
                MaxDiff2 = Diff2

        AvgDiff_f1_List.append(SumDiff1/N)
        AvgDiff_f2_List.append(SumDiff2/N)
        MaxDiff_f1_List.append(MaxDiff1)
        MaxDiff_f2_List.append(MaxDiff2)

    Ignore = 2
    C1Max = sum([y/x for x,y in zip(d_List[Ignore:],MaxDiff_f1_List[Ignore:])])/sum([1/x**2 for x,y in zip(d_List[Ignore:],MaxDiff_f1_List[Ignore:])])
    C2Max = sum([y/x for x,y in zip(d_List[Ignore:],MaxDiff_f2_List[Ignore:])])/sum([1/x**2 for x,y in zip(d_List[Ignore:],MaxDiff_f2_List[Ignore:])])
    C1Avg = sum([y/x for x,y in zip(d_List[Ignore:],AvgDiff_f1_List[Ignore:])])/sum([1/x**2 for x,y in zip(d_List[Ignore:],AvgDiff_f1_List[Ignore:])])
    C2Avg = sum([y/x for x,y in zip(d_List[Ignore:],AvgDiff_f2_List[Ignore:])])/sum([1/x**2 for x,y in zip(d_List[Ignore:],AvgDiff_f2_List[Ignore:])])

    ax[0].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)
    ax[1].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)

    ax[0].plot(d_List,AvgDiff_f1_List,color='blue',alpha=0.3)
    ax[0].plot(d_List,[C1Avg/d for d in d_List],color='green',alpha=0.3)

    ax[1].plot(d_List,AvgDiff_f2_List,color='blue',alpha=0.3)
    ax[1].plot(d_List,[C2Avg/d for d in d_List],color='green',alpha=0.3)

    ax[0].plot(d_List,MaxDiff_f1_List,color='red',alpha=0.3)
    ax[0].plot(d_List,[C1Max/d for d in d_List],color='orange',alpha=0.3)

    ax[1].plot(d_List,MaxDiff_f2_List,color='red',alpha=0.3)
    ax[1].plot(d_List,[C2Max/d for d in d_List],color='orange',alpha=0.3)

    plt.show()


