import numpy as np
from math import pi
from math import ceil
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import cvxopt

#implements the algorithm from Subsection 3.2.2 of https://doi.org/10.1214/07-AOS581

def H_Estimation_old(X,sig2=None,positions=None,verbose=False,options={}):
    d,n = X.shape
    c = d/n
    epsilon = 1/100
    SampEV = np.linalg.eigh(X@X.T/n)[0]
    if sig2 is None:
        sig2 = max(SampEV)

    zVec = np.array([lam+0.05*1j for lam in SampEV]+[lam-0.05*1j for lam in SampEV])[np.newaxis]
    SampVec = np.array(SampEV)[np.newaxis]
    M = (1-c)*(-1/zVec.T)+c/(SampEV-zVec.T)
    vVec = np.sum(M,axis=1)/d

    if positions is None:
        positions = np.arange(0,sig2,epsilon)
    positions = positions[np.newaxis]
    vVec = vVec[np.newaxis]
    M = positions/(1+positions*vVec.T)
    w0 = np.ones(len(positions[0]))/len(positions[0])
    def functionToMinimize(w):
        #print(vVec.shape,zVec.shape,w.shape,w0.shape,M.shape,(w*M).shape)
        #assert 0==1
        #eVec = 1/vVec+zVec-c*np.sum(w*M,axis=1)
        eVec = 1/vVec+zVec-c*M@w
        Loss = np.sum(np.abs(eVec)**2)
        if verbose:
            print(Loss)
        return Loss
    StartingLoss = functionToMinimize(w0)

    Constraints = [{'type': 'eq', 'fun': lambda p: 1-sum(p)}]

    sol = minimize(
        lambda p: functionToMinimize(p)/StartingLoss*100,
        w0,
        bounds=[(0,None)],
        constraints = (Constraints),
        options = options,
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

    if PopEV is not None:
        sValues = np.array([sum([1/(lam-z) for lam in PopEV])/d for z in zVec])

    epsilon = 1/100#max(d,n)
    if positions is None:
        positions = np.arange(0,sig2,epsilon)
    NrPositions = len(positions)
    positions = positions[np.newaxis]


    zVec = np.array([lam+0.05*1j for lam in SampEV]+[lam-0.05*1j for lam in SampEV])[np.newaxis]
    SampVec = np.array(SampEV)[np.newaxis]
    M = (1-c)*(-1/zVec.T)+c/(SampEV-zVec.T)
    vVec = np.sum(M,axis=1)/d
    vVec = vVec[np.newaxis]
    M = positions/(1+positions*vVec.T)

    V = c*M
    v = zVec+1/vVec
    v = v[0,:]



    Q = np.real(V).T@np.real(V)+np.imag(V).T@np.imag(V)
    p = -np.real(V).T@np.real(v)-np.imag(V).T@np.imag(v)
    b = np.matrix(1)
    A = np.ones((1,NrPositions))
    G = -np.eye(NrPositions)
    h = np.zeros(NrPositions)


    Q = Q.astype('float')
    p = p.astype('float')
    A = A.astype('float')
    b = b.astype('float')
    G = G.astype('float')
    h = h.astype('float')

    #print(p)

    Q = cvxopt.matrix(Q)
    p = cvxopt.matrix(p)
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
    sol=cvxopt.solvers.qp(Q, p, G, h, A, b)

    weights = sol['x']

    return positions[0],weights/np.sum(weights)




if __name__ == "__main__":
######################
    #hyperparameters

    #dimension and quotient c=d/n
    d = 200
    c = 1/2

    #population eigenvalues
    PopEV = [(i+1)/d for i in range(d)]
    #PopEV = [0.5 for i in range(d//2)]+[1 for i in range(d-d//2)]
######################

    n = ceil(d/c)
    sig2 = max(PopEV)

    T = np.matrix(np.diag(np.sqrt(PopEV)))
    Y = np.random.normal(size=(d,n))
    X = T@Y

    positions,weights = H_Estimation(X,sig2=1.3)

    #print(positions.shape,weights.shape)
    #print(weights)

    plt.plot(positions,[sum([1 if x>lam else 0 for lam in PopEV])/d for x in positions],alpha=0.5,color='green')
    plt.plot(positions,[sum(weights[:i+1]) for i in range(len(positions))],alpha=0.5,color='blue')
    plt.show()