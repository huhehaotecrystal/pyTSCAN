from MST import MST
from Sc_data import Sc_data
from pca_plot import pca_plot
from save_N_load import save_scdata, load_scdata

import os
import os.path
import time
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

def pyTSCAN(expr, pData = None, fData = None, save_path = "."):
    scdata = Sc_data(expr, pData = pData, fData = fData)
    scdata.filter_data()
    scdata.normlize()
    print("Performing mclust...")
    scdata.preprocessing()
    print("Performing dimension reduction...")
    scdata.pca()
    print("Performing clustering...")
    scdata.gmm()
    scdata.calc_centroid()
    scdata.mst()
    print("Cell projecting...")
    scdata.project_cells()
    pca_plot(scdata, show_pseudotime = True, show_cell_order = True)
    plt.savefig(os.path.join(save_path, "pseudotime_plot.pdf"), dpi = 100)
    save_scdata(scdata, file = os.path.join(save_path, "tscan.pydata"))
    scdata.pData.to_csv(os.path.join(save_path, "tscan_pData.csv"))
    print("Done!")
    return scdata