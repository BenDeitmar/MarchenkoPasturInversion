import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from math import ceil

import MarchenkoPasturInversion as MPI

def DBoundary(epsilon,theta,PopEV,d,n,xMin,xMax,etaMax,stepSize=0.01,iterationDepth=10):
    c = d/n
    Stil = lambda z : sum([1/(lam-z) for lam in PopEV])/d
    N = ceil((xMax-xMin)/stepSize)
    xList = [xMin+(xMax-xMin)*i/N for i in range(N)]
    bList = []
    if True:
        J = 0
        for x in xList:
            J+=1
            LastNeg = 0
            LastPos = etaMax
            for i in range(iterationDepth):
                eta = (LastPos+LastNeg)/2
                c = d/n
                z = x+1j*eta
                if np.imag((1-c-c*z*Stil(z))*z) > epsilon and c*abs(z)*np.imag(z*Stil(z))/np.imag((1-c-c*z*Stil(z))*z) < theta:
                    LastPos = eta
                else:
                    LastNeg = eta
            bList.append(LastPos)
    return(xList,bList)



if __name__ == "__main__":

######################
    #hyperparameters

    #dimension and quotient c=d/n
    d = 10
    c = 1/10

    #population eigenvalues
    PopEV = [0.5 for i in range(d//2)]+[0.5+i/d for i in range(d-d//2)]

    #defining oracle curve
    gammaLeft,gammaTop,gammaRight = -0.1, 0.5, 1.5
######################

    n = ceil(d/c)
    sig2 = 1

    dist = 4*sig2*(1+c)
    gamma1 = MPI.makeGamma_Legendre(gammaLeft,gammaTop,gammaRight,N=10,show=False)
    gamma2 = MPI.makeGamma_Legendre(-dist,dist,sig2+dist,N=10,show=False)

    xMin,etaMax,xMax = -1.1*dist,1.1*dist,1.1*(sig2+dist)

    T = np.matrix(np.diag(np.sqrt(PopEV)))
    Y_matr = np.random.normal(size=(d,n))
    X_matr = T@Y_matr
    S_matr = X_matr@X_matr.H/n
    SampEV = np.linalg.eigh(S_matr)[0]


    fig, ax = plt.subplots(1,2,layout='constrained')
    
    ax[0].set_title('Curve visualization')
    ax[1].set_title('Curve approximation')

    print('finding boundaries')
    xList,zImagList = DBoundary(0,np.inf,PopEV,d,n,xMin,xMax,etaMax,stepSize=0.01)
    _,zThetaImagList = DBoundary(0,1,PopEV,d,n,xMin,xMax,etaMax,stepSize=0.01)
    print('done')

    ax[0].plot([xMin,xMax],[0,0],color='gray',alpha=0.2)
    ax[1].plot([xMin,xMax],[0,0],color='gray',alpha=0.2)

    ax[0].plot(xList,zImagList,alpha=0.3,color='green')
    ax[0].plot(xList,zThetaImagList,alpha=0.3,color='green',linestyle='dashed')
    ax[1].plot(xList,zImagList,alpha=0.3,color='green')
    ax[1].plot(xList,zThetaImagList,alpha=0.3,color='green',linestyle='dashed')

    ax[0].scatter(PopEV,[0]*d,alpha=1,color='orange',s=10)
    ax[1].scatter(PopEV,[0]*d,alpha=1,color='orange',s=10)

    ax[0].plot([gammaLeft,gammaLeft,gammaRight,gammaRight],[0,gammaTop,gammaTop,0],color='blue',alpha=0.5)
    ax[0].plot([-dist,-dist,sig2+dist,sig2+dist],[0,dist,dist,0],color='blue',alpha=0.5)

    gamma1_points = [tup[0] for tup in gamma1]
    gamma2_points = [tup[0] for tup in gamma2]
    ax[1].scatter(np.real(gamma1_points),np.imag(gamma1_points),color='blue',alpha=0.5)
    ax[1].scatter(np.real(gamma2_points),np.imag(gamma2_points),color='blue',alpha=0.5)


    ax[0].set_xlim([xMin, xMax])
    ax[0].set_ylim([-0.05*etaMax, etaMax])

    ax[1].set_xlim([xMin, xMax])
    ax[1].set_ylim([-etaMax, etaMax])
    plt.show()


