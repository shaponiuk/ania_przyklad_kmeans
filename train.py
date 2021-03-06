import pickle
import pandas as pd
from sklearn.cluster import KMeans

df = pd.read_csv("data_1.csv")

ktest = KMeans(n_clusters=3)
ktest.fit(df.drop('y', axis=1)) #fitting the model to X
y_pred = ktest.predict(df.drop('y', axis=1)) #predicting labels (y) and saving to y_pred

with open('model.pkl', 'wb') as handle:
    pickle.dump(ktest, handle)

