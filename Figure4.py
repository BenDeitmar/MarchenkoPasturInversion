import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from math import ceil

def MPInversion_StepCounting(z,SampEV,d,n,PopEV,maxRepetitions=100,epsilon=10**(-6)):
    c = d/n
    LastW = 0
    w = 1j
    for i in range(maxRepetitions):
        if abs(w-LastW) < epsilon:
            return ((w-1)/z,i)
        LastW = w
        w = sum([lam/(lam-(1-c*w)*z) for lam in SampEV])/d
    return (np.nan,np.nan)

def MPInversion(z,SampEV,d,n,maxRepetitions=100,epsilon=10**(-6),verbose=False):
    c = d/n
    LastW = 0
    w = 1j
    for i in range(maxRepetitions):
        #if np.imag(w)<-0.2:
            #w = np.random.normal()+1j*abs(np.random.normal())
        if verbose:
            print(i,(w-1)/z)
        if abs(w-LastW) < epsilon:
            return (w-1)/z
        LastW = w
        w = sum([lam/(lam-(1-c*w)*z) for lam in SampEV])/d
    return 0

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
    d = 500
    c = 2

    #population eigenvalues
    PopEV = [0.5 for i in range(d//2)]+[0.5+i/d for i in range(d-d//2)]

    #(left edge, height, right edge) of the window
    xMin,etaMax,xMax = -1, 2, 3.5

    #resolution
    delta = 0.02
######################

    n = ceil(d/c)
    sig2 = max(PopEV)

    T = np.matrix(np.diag(np.sqrt(PopEV)))
    Y_matr = np.random.normal(size=(d,n))
    X_matr = T@Y_matr
    S_matr = X_matr@X_matr.H/n
    SampEV = np.linalg.eigh(S_matr)[0]

    x = np.arange(xMin, xMax, delta)
    y = np.arange(0.001, etaMax, delta/2)
    X, Y = np.meshgrid(x, y)

    Z = X+1j*Y
    nr, nc = Z.shape
    Steps = np.zeros((nr,nc))
    Differences = np.zeros((nr,nc))

    for i in range(nr):
        print(i,'/',nr)
        for j in range(nc):
            z = Z[i,j]
            val,steps = MPInversion_StepCounting(z,SampEV,d,n,PopEV)
            Steps[i,j] = steps
            if np.isnan(val):
                Differences[i,j] = np.nan
            else:
                Differences[i,j] = min(10,abs(val-sum([1/(lam-z) for lam in PopEV])/d))

    Steps = np.ma.array(Steps)
    Differences = np.ma.array(Differences)

    fig, ax = plt.subplots(1,2,layout='constrained')
    CS0 = ax[0].contourf(X, Y, Steps, 30, cmap='Blues',vmin=0,vmax=100)
    lev_exp = np.arange(-4,1.1,0.25)
    levs = np.power(10, lev_exp)
    CS1 = ax[1].contourf(X, Y, Differences, levs, norm=colors.LogNorm(), cmap='Reds')

    ax[0].set_title('Steps until convergence')
    ax[1].set_title('Difference to true Stieltjes transform')
    ax[0].set_xlim([xMin, xMax])
    ax[0].set_ylim([-0.05*etaMax, etaMax])
    ax[1].set_xlim([xMin, xMax])
    ax[1].set_ylim([-0.05*etaMax, etaMax])

    cbar0 = fig.colorbar(CS0)
    cbar1 = fig.colorbar(CS1)

    print('finding boundaries')
    xList,zImagList = DBoundary(0,np.inf,PopEV,d,n,xMin,xMax,etaMax,stepSize=delta)
    _,zThetaImagList = DBoundary(0,1,PopEV,d,n,xMin,xMax,etaMax,stepSize=delta)
    print('done')

    ax[0].plot([xMin,xMax],[0,0],color='gray',alpha=0.2)
    ax[1].plot([xMin,xMax],[0,0],color='gray',alpha=0.2)

    ax[0].plot(xList,zImagList,alpha=0.3,color='green')
    ax[1].plot(xList,zImagList,alpha=0.3,color='green')
    ax[0].plot(xList,zThetaImagList,alpha=0.3,color='green',linestyle='dashed')
    ax[1].plot(xList,zThetaImagList,alpha=0.3,color='green',linestyle='dashed')

    ax[0].scatter(PopEV,[0]*d,alpha=1,color='orange',s=10)
    ax[1].scatter(PopEV,[0]*d,alpha=1,color='orange',s=10)

    plt.show()


