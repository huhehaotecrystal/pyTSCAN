from MST import MST
from Sc_data import Sc_data

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

def pca_plot(scdata, show_pseudotime = False, show_cell_order = False):
    plt.figure(figsize = (10, 10))
    if not show_pseudotime:
        plt.scatter(scdata.pca_expr[:, 0], scdata.pca_expr[:, 1])
    else:
        cluster = scdata.pData["gmm_cluster"]
        for tp in np.unique(cluster):
            data_part = scdata.pca_expr[cluster == tp, :]
            plt.scatter(data_part[:, 0], data_part[:, 1], label = tp)
        leg_prop = {"weight" : "bold", "size" : 20}
        plt.legend(bbox_to_anchor=(1.05,1.0), prop = leg_prop, title = "State")
        if not scdata.gmm_centroid == []:
            for i in range(scdata.gmm_centroid.shape[0]):
                plt.scatter(scdata.gmm_centroid[i, 0], scdata.gmm_centroid[i, 1], marker = "^", s = 48, color = "grey")
                plt.text(x = scdata.gmm_centroid[i, 0], y = scdata.gmm_centroid[i, 1], s = str(i+1), fontsize = 20)
        if scdata.mst_pairs:
            for pair in scdata.mst_pairs:
                plot_line = scdata.gmm_centroid[[pair[0] - 1, pair[1] - 1], :2]
                plt.plot(plot_line[:, 0], plot_line[:, 1], color = "black")
    if show_cell_order:
        for cell in range(scdata.pData.shape[0]):
            plt.text(x = scdata.pca_expr[cell, 0], y = scdata.pca_expr[cell, 1], 
                     s = str(scdata.pData["cell_order"][cell]), fontsize = 6, color = "black")
            
    plt.xlabel("PC1(%s%%)" % round(scdata.evr[0] * 100, 2))
    plt.ylabel("PC1(%s%%)" % round(scdata.evr[1] * 100, 2))