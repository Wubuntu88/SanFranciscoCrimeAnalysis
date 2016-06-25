#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
df1 = pd.read_csv("train.csv", header=0)
df2 = df1[df1['X'].apply(lambda x: x < -121.5)]
df3 = df1[df1['X'].apply(lambda x: x < -121.5)]
districts = df1.PdDistrict.unique()
print("len: " + str(len(districts)))
colors = ["red", "blue", "brown", "teal", "orange",
          "silver", "burlywood", "coral", "beige",
          "olive"]
dist_colors = {}
for i in range(0, len(districts)):
    dist_colors[districts[i]] = colors[i]
for dist in districts:
    temp_df = df2[df2["PdDistrict"] == dist]
    plt.scatter(x=temp_df['X'], y=temp_df['Y'], c=dist_colors[dist])
plt.show()
