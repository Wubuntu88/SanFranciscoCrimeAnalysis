#!/usr/bin/env python

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def plot_and_show_crime_type_histogram(dataframe):
    """
    Shows the histogram where each crime type is shown with the number
    of times that it was committed.
    :param dataframe: The dataframe that holds the data.
        It should have the column named "Category"
    :result: shows a histogram in a popup window.
    """
    #df2.Category.value_counts().plot(kind='bar')
    sns.countplot(y="Category", data=dataframe, palette="Greens_d")
    plt.suptitle("Crime Type Instances", fontsize=30)
    plt.ylabel("Type of Crime", fontsize=26)
    plt.xlabel("Number of Crimes Committed", fontsize=26)
    plt.show()

'''--------Start of Program-------------'''
df1 = pd.read_csv("train.csv", header=0)
#gr = df1.groupby(["PdDistrict"]).count
#creating a second dataframe "df2" which does not have the outlier locations
#The locations that are less than -121.5 (for the 'x') are outlier locations
#(they mess up the map and make everything tiny dots)
df2 = df1[df1['X'].apply(lambda x: x < -121.5)]
#df2.plot.scatter(x='X', y='Y', c=df2["PdDistrict"])
#plt.scatter(x=df2['X'], y=df2['Y'], c=df2["PdDistrict"])
#plt.show()
plot_and_show_crime_type_histogram(df2)
'''
cats = df2.Category.unique()
print("num categories: ", len(cats))
print(cats)
'''
'''
districts = df2.PdDistrict.unique()
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
'''

