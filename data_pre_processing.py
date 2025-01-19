import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
def clean(filepath):
    df = pd.read_csv(filepath, dtype = int, header= 0, index_col = None)
    return df.loc[df['label'] < 2]
clean("raw_mnist/mnist_train.csv").to_csv('data/train.csv', header=False, index=False)
clean("raw_mnist/mnist_test.csv").to_csv('data/test.csv', header=False, index=False)