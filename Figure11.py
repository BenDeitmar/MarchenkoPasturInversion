import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker, colors
from math import ceil
from cmath import exp
import os
import pandas as pd
data_path = ''.join(map(lambda string: string+'\\', os.path.realpath(__file__).split('\\')[:-1]))+'data\\'


import MarchenkoPasturInversion as MPI
import EigenInference_ElKaroui as EI_a

if __name__ == "__main__":

######################
    #hyperparameters

    #dimension and quotient c=d/n
    d = 1000
    c = 10

    PopEV = [0.5 for i in range(d//2)]+[0.5+i/d for i in range(d-d//2)]

    sig2=1

    d1,d2 = 1.25,2.5

    positions = np.arange(-0.2,1.2*sig2,0.005)
    #positions = np.arange(0,1.2*sig2,0.001)
    #positions = np.arange(0.49,1*sig2,0.005)
######################
    
    fig, ax = plt.subplots(1,1,layout='constrained')

    n = ceil(d/c)

    ax.plot(positions,[sum([1 if x>lam else 0 for lam in PopEV])/d for x in positions],alpha=0.5,color='green',linewidth=3)
    
    try:
        EstimatedEigenvalues = pd.read_csv(data_path+'EV_Estimation_d={}_c={}.csv'.format(d,c), sep=',')
        EstimatedEigenvalues = EstimatedEigenvalues['Estimator'].values.tolist()
        ax.plot(positions,[sum([1 if x>lam else 0 for lam in EstimatedEigenvalues])/d for x in positions],color='orange',alpha=0.5,linewidth=3)
    except:
        print('#############################')
        print("Error: could not load the results")
        print("try running EigenvalueEstimation.R with the same values for d and c")
        print("before running this script again")
        print("looking for: ",data_path+'EV_Estimation_d={}_c={}.csv'.format(d,c))
        print('#############################')


    try:
        EstimatedEigenvalues = pd.read_csv(data_path+'EV_Estimation_LedoitWolf_d={}_c={}.csv'.format(d,c), sep=',')
        EstimatedEigenvalues = EstimatedEigenvalues['Estimator_LedoitWolf'].values.tolist()
        ax.plot(positions,[sum([1 if x>lam else 0 for lam in EstimatedEigenvalues])/d for x in positions],color='black',alpha=0.5,linewidth=3)
    except:
        print('#############################')
        print("Error: could not load the results")
        print("try running EigenvalueEstimation.R with the same values for d and c")
        print("before running this script again")
        print("looking for: ",data_path+'EV_Estimation_LedoitWolf_d={}_c={}.csv'.format(d,c))
        print('#############################')




    plt.show()


