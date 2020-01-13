import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import cv2

files = sorted(list(map(lambda x: os.path.join("bench/", x), os.listdir("./bench"))))
colnames = ["file", "time"]

for f in files:
    df = pd.read_csv(f, names=colnames)
    xticks = df["file"]
    plt.xticks(np.arange(len(xticks)), xticks, rotation=45)
    plt.plot(df["time"], label=f)
    plt.scatter(np.arange(len(df['time'])), df["time"], linestyle='-', marker='o')
plt.legend()
plt.show()
