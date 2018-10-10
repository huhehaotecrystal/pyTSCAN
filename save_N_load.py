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

def save_scdata(scdata, file = None):
    if file is None:
        print("Please enter a file name!")
    else:
        with open(file, "wb") as f:
            pickle.dump(scdata, f)

def load_scdata(file):
    return pickle.load(open(file, "rb"))