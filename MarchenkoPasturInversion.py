import numpy as np
import scipy as sp
from math import log, e, pi
from math import factorial, gamma
from math import floor, ceil, sqrt
from cmath import exp,sin,cos
import os
import matplotlib.pyplot as plt
from itertools import product
from collections import Counter
from scipy.optimize import minimize
import cvxopt

def Vectorwise_MPI(zVec,SampEV,d,n,maxRepetitions=100,epsilon=10**(-6),verbose=False):
    c = d/n
    LastW = np.zeros(len(zVec))
    zVec = zVec[np.newaxis]
    SampVec = np.array(SampEV)[np.newaxis]
    w = np.sum(1/(SampVec.T-zVec),axis=0)/d
    for i in range(maxRepetitions):
        if max(np.abs(w-LastW)) < epsilon:
            if verbose:
                print('convergence after {} steps'.format(i))
            sol = (w-1)/zVec
            return sol[0,:]
        LastW = w
        w = np.sum(SampVec.T/(SampVec.T-(1-c*w)*zVec),axis=0)/d
    print('Error: did not converge')
    return np.zeros(len(zVec))


def MPInversion(z,SampEV,d,n,maxRepetitions=100,epsilon=10**(-6)):
    c = d/n
    LastW = 0
    w = sum([1/(lam-z) for lam in SampEV])/d
    for i in range(maxRepetitions):
        if abs(w-LastW) < epsilon:
            print('convergence after {} steps at {}'.format(i,z))
            return (w-1)/z
        LastW = w
        w = sum([lam/(lam-(1-c*w)*z) for lam in SampEV])/d
    print('Error: did not converge')
    return 0

def makeGamma(left,top,right,N=30,show=False):
    LeftList = [left + 1j*(-top+(i+0)*2*top/N) for i in range(N)]
    TopList = [left+(i+0)*(right-left)/N + 1j*top for i in range(N)]
    RightList = [right + 1j*(top-(i+0)*2*top/N) for i in range(N)]
    BottomList = [right-(i+0)*(right-left)/N - 1j*top for i in range(N)]

    vLeft = 1j*2*top/N
    TupleList = [(p,vLeft) for p in LeftList]
    vTop = (right-left)/N
    TupleList += [(p,vTop) for p in TopList]
    vRight = -1j*2*top/N
    TupleList += [(p,vRight) for p in RightList]
    vBottom = -(right-left)/N
    TupleList += [(p,vBottom) for p in BottomList]

    if show:
        plt.scatter(np.real([tup[0] for tup in TupleList]),np.imag([tup[0] for tup in TupleList]))
        plt.scatter(np.real([tup[0]+0.5*tup[1] for tup in TupleList]),np.imag([tup[0]+0.5*tup[1] for tup in TupleList]))
        plt.show()
        plt.clf()

    return TupleList

def makeGamma_Legendre(left,top,right,N=30,show=False):
    LegendreNodes, LegendreWeights = np.polynomial.legendre.leggauss(N)

    LeftList = [left + 1j*(-top+(x+1)/2*2*top) for x in LegendreNodes]
    TopList = [left+(x+1)/2*(right-left) + 1j*top for x in LegendreNodes]
    RightList = [right + 1j*(top-(x+1)/2*2*top) for x in LegendreNodes]
    BottomList = [right-(x+1)/2*(right-left) - 1j*top for x in LegendreNodes]

    #SideWeights = LegendreWeights*top
    #TopWeights = LegendreWeights/2*(right-left)

    SideWeights = LegendreWeights/2*len(LegendreWeights)
    TopWeights = LegendreWeights/2*len(LegendreWeights)

    vLeft = 1j*2*top/N
    TupleList = [(LeftList[i],vLeft*SideWeights[i]) for i in range(len(LeftList))]
    vTop = (right-left)/N
    TupleList += [(TopList[i],vTop*TopWeights[i]) for i in range(len(TopList))]
    vRight = -1j*2*top/N
    TupleList += [(RightList[i],vRight*SideWeights[i]) for i in range(len(RightList))]
    vBottom = -(right-left)/N
    TupleList += [(BottomList[i],vBottom*TopWeights[i]) for i in range(len(BottomList))]


    if show:
        plt.scatter(np.real([tup[0] for tup in TupleList]),np.imag([tup[0] for tup in TupleList]))
        plt.scatter(np.real([tup[0]+0.5*tup[1] for tup in TupleList]),np.imag([tup[0]+0.5*tup[1] for tup in TupleList]))
        plt.show()
        plt.clf()

    return TupleList

def CurveIntegral(f,gamma,sValues):
    I = 0
    for i in range(len(gamma)):
        z,v = gamma[i]
        s = sValues[i]
        I += v*f(z)*s
    return I/(2*pi*1j)


def H_Estimation_old(X,dist=None,sig2=None,PopEV=None,positions=None,verbose=False,N=None,options={}):
    d,n = X.shape
    c = d/n
    S = X@X.T/n
    SampEV = np.linalg.eigh(S)[0]
    if sig2 is None:
        sig2 = max(SampEV)
    if dist is None:
        dist = 4*sig2*(1+c)
    if N is None:
        N=max(100,d,n//2)
    #gamma = makeGamma(-dist,dist,sig2+dist,N=1000)
    gamma = makeGamma_Legendre(-dist,dist,sig2+dist,N=N)
    zVec = np.array([tup[0] for tup in gamma if np.imag(tup[0])>0])
    sValues = Vectorwise_MPI(zVec,SampEV,d,n,epsilon=1e-12)

    if PopEV is not None:
        sValues = np.array([sum([1/(lam-z) for lam in PopEV])/d for z in zVec])

    epsilon = 1/100#max(d,n)
    if positions is None:
        positions = np.arange(0,sig2,epsilon)
    NrPositions = len(positions)
    positions = positions[np.newaxis]
    zVec = zVec[np.newaxis]
    V = 1/(positions-zVec.T)
    #print(V.shape,NrPositions)
    p0 = np.array([i/NrPositions for i in range(NrPositions)])

    def functionToMinimize(p):
        Loss = np.sum(np.abs(V@p-sValues))
        #Loss *= 1e-4
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
        method='SLSQP',
        )

    weights = sol.x

    return positions[0],weights/np.sum(weights)


def H_Estimation(X,dist=None,sig2=None,PopEV=None,positions=None,verbose=False,N=None,options={}):
    d,n = X.shape
    c = d/n
    S = X@X.T/n
    SampEV = np.linalg.eigh(S)[0]
    if sig2 is None:
        sig2 = max(SampEV)
    if dist is None:
        dist = 4*sig2*(1+c)
    if N is None:
        N=max(100,d,n//2)
    #gamma = makeGamma(-dist,dist,sig2+dist,N=1000)
    gamma = makeGamma_Legendre(-dist,dist,sig2+dist,N=N)
    zVec = np.array([tup[0] for tup in gamma if np.imag(tup[0])>0])
    sValues = Vectorwise_MPI(zVec,SampEV,d,n,epsilon=1e-12)

    if PopEV is not None:
        sValues = np.array([sum([1/(lam-z) for lam in PopEV])/d for z in zVec])

    epsilon = 1/100#max(d,n)
    if positions is None:
        positions = np.arange(0,sig2,epsilon)
    NrPositions = len(positions)
    positions = positions[np.newaxis]
    zVec = zVec[np.newaxis]
    V = np.matrix(1/(positions-zVec.T))
    #print(V.shape,NrPositions)
    x0 = np.array([i/NrPositions for i in range(NrPositions)])

    Q = np.real(V).T@np.real(V)+np.imag(V).T@np.imag(V)
    p = -np.real(V).T@np.real(sValues)-np.imag(V).T@np.imag(sValues)
    b = np.matrix(1)
    A = np.ones((1,len(x0)))
    G = -np.eye(len(x0))
    h = np.zeros(len(x0))


    Q = Q.astype('float')
    p = p.astype('float')
    A = A.astype('float')
    b = b.astype('float')
    G = G.astype('float')
    h = h.astype('float')

    #print(p)

    Q = cvxopt.matrix(Q)
    p = cvxopt.matrix(p).T
    A = cvxopt.matrix(A)
    b = cvxopt.matrix(b)
    G = cvxopt.matrix(G)
    h = cvxopt.matrix(h)
    #[np.newaxis].T

    tol = 1e-14
    cvxopt.solvers.options['reltol']=tol
    cvxopt.solvers.options['abstol']=tol
    cvxopt.solvers.options['maxiters']=1000
    cvxopt.solvers.options['feastol']=tol
    sol=cvxopt.solvers.qp(Q, p, G, h, None, None)

    weights = sol['x']

    return positions[0],weights/np.sum(weights)



if __name__ == "__main__":
    None


