from .common import getSetup, subplotLabel
import seaborn as sns
import matplotlib.pyplot as plt
from ..functional import *


"""
def makeFigure():
    ax, f = getSetup((8, 4), (1, 2))
    
    R = 4
    R2X, R2Y = plsr(R)

    sns.heatmap(R2X.reshape((R,1)), cmap="PiYG",  cbar=True, ax=ax[0])
    sns.heatmap(R2Y.reshape((R,1)), cmap="PiYG",  cbar=True, ax=ax[1])

    ax[0].set_title('R2X')
    ax[1].set_title('R2Y')
    
    return f 

"""

def makeFigure():
    ax, f = getSetup((8, 4), (1, 1))
    R = 5
    R2X, R2Y = plsr(R)

    ax[0].scatter(np.arange(R), R2X, label='R2X')
    ax[0].scatter(np.arange(R), R2Y, label='R2Y')

    ax[0].set_xlabel("Components")
    ax[0].legend()
    ax[0].set_xticklabels(ax[0].get_xmajorticklabels())

    return f