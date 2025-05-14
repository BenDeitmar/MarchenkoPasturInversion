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

    
    fig, ax = plt.subplots(1,2,layout='constrained')

    ax[0].set_title('Estimation accuracy')
    ax[1].set_title('Time in seconds')

    #try:
    if True:
        DataMatrix = pd.read_csv(data_path+'Fig12_Results.csv', sep=',')
        d_List = DataMatrix['d_List'].values.tolist()
        AvgDiffList1 = DataMatrix['AvgDiffList1'].values.tolist()
        AvgDiffList2 = DataMatrix['AvgDiffList2'].values.tolist()
        AvgTimeList1 = DataMatrix['AvgTimeList1'].values.tolist()
        AvgTimeList2 = DataMatrix['AvgTimeList2'].values.tolist()
        ax[0].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)
        ax[1].plot(d_List,[0]*len(d_List),color='gray',alpha=0.3)

        ax[0].plot(d_List,AvgDiffList1,color='orange',alpha=0.5,linewidth=2)
        ax[0].plot(d_List,AvgDiffList2,color='black',alpha=0.5,linewidth=2)
        ax[1].plot(d_List,AvgTimeList1,color='orange',alpha=0.5,linewidth=2)
        ax[1].plot(d_List,AvgTimeList2,color='black',alpha=0.5,linewidth=2)
    #except:
        print('#############################')
        print("Error: could not load the results")
        print("try running EigenvalueEstimation_Fig12.R with the same values for d and c")
        print("before running this script again")
        print("looking for: ",data_path+'Fig12_Results.csv')
        print('#############################')






    plt.show()


