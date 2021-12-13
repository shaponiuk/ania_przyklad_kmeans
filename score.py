import numpy as np
import pickle
import pandas as pd
from sklearn.datasets.samples_generator import make_blobs

df = pd.read_csv("data_1.csv")

with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

y_preds = model.predict(df.drop('y', axis=1))

print(y_preds)
