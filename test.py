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

from MST import MST
from Sc_data import Sc_data
from pca_plot import pca_plot
from save_N_load import save_scdata, load_scdata
from pyTSCAN import pyTSCAN

if __name__ == "__main__":
	expr = pd.read_csv("./test_data/deng/deng_expr.csv", index_col = 0)
	phenoData = pd.read_csv("./test_data/deng/celltypes.csv", index_col = 0)
	phenoData.index = list(expr.columns)
	scdata = pyTSCAN(expr, pData = phenoData, save_path = "./test_result/")