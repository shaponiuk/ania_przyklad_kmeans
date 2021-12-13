import pickle
import pandas as pd

df = pd.read_csv("data_1.csv")

with open('model.pkl', 'rb') as handle:
    model = pickle.load(handle)

y_preds = model.predict(df.drop('y', axis=1))

y_preds_df = pd.DataFrame(y_preds.reshape(-1, 1), columns = ['cluster_id'])
X = df.drop('y', axis=1)
result_df = pd.concat([X, y_preds_df], axis=1)
result_df.to_csv("result.csv")