import pickle
import pandas as pd

df = pd.read_csv("data_1.csv")

with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

y_preds = model.predict(df.drop('y', axis=1))

y_preds_series = pd.Series(y_preds)
y_preds_series.to_csv("result.csv")