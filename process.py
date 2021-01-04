import os
import numpy as np
import pandas as pd
from sklearn import preprocessing

FILE = "nn/ce889_dataCollection_g3.csv"


def get_data(file_name):
    df = pd.read_csv(file_name, names=[
                     "X Distance", "Y Distance", "New Vel Y", "New Vel X"])
    df = df.sample(frac=1).reset_index(drop=True)
    scaler = preprocessing.MinMaxScaler()
    inputs = df.iloc[:, 0:2]
    outputs = df.iloc[:, 2:]
    scaled_inputs = scaler.fit_transform(inputs)
    scaled_outputs = scaler.fit_transform(outputs)
    df = pd.DataFrame(scaled_inputs)
    df2 = pd.DataFrame(scaled_outputs)
    return pd.concat([df, df2], axis=1)


def get_extremes(file_name):
    extremes = []
    df = pd.read_csv(file_name, names=[
                     "X Distance", "Y Distance", "New Vel Y", "New Vel X"])
    extremes.append(df["X Distance"].min())
    extremes.append(df["X Distance"].max())
    extremes.append(df["Y Distance"].min())
    extremes.append(df["Y Distance"].max())
    extremes.append(df["New Vel Y"].min())
    extremes.append(df["New Vel Y"].max())
    extremes.append(df["New Vel X"].min())
    extremes.append(df["New Vel X"].max())
    return extremes


data = get_data(FILE)
print(data)
data.to_csv("nn/processed.csv", index=False, header=False)
with open("nn/processed.csv", "r") as f:
    with open("nn/tmp.csv", "w") as f2:
        f2.write("{},{},{},{},{},{},{},{}\n".format(*get_extremes(FILE)))
        f2.write(f.read())
os.remove('nn/processed.csv')
os.rename('nn/tmp.csv', "nn/processed.csv")
