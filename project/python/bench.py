import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

files = list(map(lambda x: os.path.join("bench/", x), os.listdir("./bench")))
colnames = ["file", "time"]

for f in files:
    df = pd.read_csv(f, names=colnames)
    xticks = df["file"]
    plt.xticks(np.arange(len(xticks)), xticks, rotation=45)
    plt.plot(df["time"])
    plt.show()