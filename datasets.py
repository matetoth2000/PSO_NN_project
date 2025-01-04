import numpy as np
import sklearn.datasets as sklds
import pandas as pd
from sklearn.utils import Bunch
from urllib.request import urlopen

########## selected datasets from https://archive.ics.uci.edu/ ############
def load_glass_data():
    print("== Loading Glass Data ...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
    columns = ["Id", "RI", "Na", "Mg", "Al", "Si", "K", "Ca", "Ba", "Fe", "Type"]
    data = pd.read_csv(url, header=None, names=columns)
    return build_dataset(data, columns)

def load_magic_gamma_data():
    print("== Loading Magic Gamma Data ...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/magic/magic04.data"
    columns = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
    data = pd.read_csv(url, header=None, names=columns)
    return build_dataset(data, columns)

def load_pen_digits_data():
    print("== Loading Digits Data ...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra"
    columns = [f"feature_{i}" for i in range(1, 17)] + ["class"]
    data = pd.read_csv(url, header=None, names=columns)
    return build_dataset(data, columns)

def load_wine_quality_data():
    print("== Loading Wine Quality Data ...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    data = pd.read_csv(url, delimiter=';')
    return build_dataset(data, data.columns)

def build_dataset(data, columns):
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("== [ {} Instances, {} Dimensions, {} classes ]".format(len(X), X.shape[1], len(np.unique(y))))
    print("== ===========================================") 
    return Bunch(data=X, target=y, feature_names=columns[:-1], target_names=list(map(str, range(10))), DESCR="Pen-Based Recognition of Handwritten Digits dataset")

########## selected datasets from  sklearn.datasets ############

def load_iris_data():
    print("== Loading Iris Data ...")
    data = sklds.load_iris()
    return build_skl_dataset(data)

def load_breast_cancer():
    print("== Loading Breast Cancer Data ...")
    data = sklds.load_breast_cancer()
    return build_skl_dataset(data)

def build_skl_dataset(data):
    X = data['data']
    y = data['target']
    print("== [ {} Instances, {} Dimensions, {} classes ]".format(len(X), X.shape[1], len(np.unique(y))))  
    print("== ===========================================") 
    return data
