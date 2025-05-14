import numpy as np
from math import log, e, pi
from math import factorial, gamma
from math import floor, ceil, sqrt
import matplotlib.pyplot as plt
from scipy.optimize import minimize,linprog

#Implements Algorithms 1 and 2 from https://doi.org/10.1214/16-AOS1525

def choose(n,k):
    return factorial(n)//factorial(k)//factorial(n-k)

#Algorithm 1
def MomentEstimator(X,Moment):
    d,n = X.shape
    A = X.T@X
    G = A.copy()
    G[np.tril_indices(n)] = 0
    for k in range(Moment-1):
        A = G@A
    return np.trace(A)/d/choose(n,Moment)

#Algorithm 1 to estimate multiple Moments at once:
def MomentEstimator_consecutive(X,maxMoment):
    d,n = X.shape
    A = X.T@X
    if sig2 is None:
        sig2 = max(np.linalg.eigh(A)[0])/n
    G = A.copy()
    for i in range(n):
        for j in range(i+1):
            G[i,j] = 0
    MomentEstimators = [np.trace(A)/d/choose(n,1)]
    for k in range(2,maxMoment+1):
        A = G@A
        momEst = np.trace(A)/d/choose(n,k)
        if verbose:
            print('Estimated {}-th moment as {}'.format(k,momEst))
        MomentEstimators.append(momEst)
    MomentEstimators = np.array(MomentEstimators)

#Algorithm 2
def MomentInference(X,maxMoment=7,sig2=None,givenMoments=None,positions=None,verbose=False,options={}):
    d,n = X.shape
    MomentEstimators = MomentEstimator_consecutive(X,maxMoment)

    if givenMoments is not None:
        MomentEstimators = givenMoments
        maxMoment = len(MomentEstimators)

    epsilon = 1/100#max(d,n)
    if positions is None:
        positions = np.arange(0,sig2,epsilon)
    V = np.flip(np.vander(np.array(positions),maxMoment+1),axis=1)[:,1:].T
    p0 = np.array([i/len(positions) for i in range(len(positions))])
    def functionToMinimize(p):
        #Loss = np.sum(np.abs(V@p-MomentEstimators))
        Loss = 0
        Diff = np.abs(V@p-MomentEstimators)
        for k in range(maxMoment):
            ci = (2*k)**(2*k)*max(d**(k/2-1),1)/n**(k/2)
            Loss += Diff[k]/MomentEstimators[k]/ci
        if verbose:
            print(Loss)
        return Loss
    StartingLoss = functionToMinimize(p0)
    #functionToMinimize = lambda p: np.sum(np.abs(V@p-MomentEstimators))
    Constraints = [{'type': 'eq', 'fun': lambda p: 1-sum(p)}]#+[{'type': 'ineq', 'fun': lambda p : p}]

    sol = minimize(
        lambda p: functionToMinimize(p)/StartingLoss*100,
        p0,
        bounds=[(0,None)],
        constraints = (Constraints),
        options = options,
        )

    weights = sol.x

    return positions,weights/np.sum(weights)




if __name__ == "__main__":
######################
    #hyperparameters

    #dimension and quotient c=d/n
    d = 1000
    c = 1/2

    #Moment to be estimated
    K = 4

    #population eigenvalues
    PopEV = [(i+1)/d for i in range(d)]
    #PopEV = [0.5 for i in range(d//2)]+[1 for i in range(d-d//2)]
######################

    n = ceil(d/c)
    sig2 = max(PopEV)

    T = np.matrix(np.diag(np.sqrt(PopEV)))
    Y = np.random.normal(size=(d,n))
    X = T@Y

    print('actual moment:',sum([lam**K for lam in PopEV])/d)

    print('estimated moment:', MomentEstimator(X,K))


    PopMoments = [sum([lam**k for lam in PopEV])/d for k in range(1,8)]
    print(PopMoments)

    #positions,weights = MomentInference(X,b=1.3,givenMoments=np.array(PopMoments))
    positions,weights = MomentInference(X,sig2=1.3,maxMoment=7)

    plt.plot(positions,[sum([1 if x>lam else 0 for lam in PopEV])/d for x in positions],alpha=0.5,color='green')
    plt.plot(positions,[sum(weights[:i+1]) for i in range(len(positions))],alpha=0.5,color='purple')
    plt.show()