import pennylane as qml
from pennylane import numpy as qnp
import jax
import optax
import sklearn as skl
import pandas as pd

num_wires = 5
num_rot = 3
num_layers = 8

dev = qml.device("default.qubit", wires = num_wires)

def get_data(dataset):
    df = pd.read_csv(dataset, dtype = int, header= None, index_col = None)
    y_train = qnp.array(df.iloc[:,0])
    x = df.iloc[:,1:]
    x_train = qnp.array([qnp.array(x.iloc[:,2*i]) + qnp.array(1j*x.iloc[:,2*i + 1]) for i in range(x.shape[1]//2)]).T
    return x_train, y_train

get_data("data/train.csv")