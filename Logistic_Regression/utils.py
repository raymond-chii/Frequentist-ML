import pandas as pd
import numpy as np

class data:
    ''' Class used to store/split the data in a reproducable way'''
    def __init__(self, df):
        rows, col = df.shape
        df = df.sample(frac=1, replace=False, random_state=42)
        self.train = df.iloc[0:int(np.ceil(rows*0.8))]
        self.valid = df.iloc[int(np.ceil(rows*0.8)):int(np.ceil(rows*0.9))]
        self.test = df.iloc[int(np.ceil(rows*0.9)):rows]
