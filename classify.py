#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model


def create_feature_matrix_and_label_vector(dataframe):
    """
    Takes the dataframe that has categorical variables as strings
    and turns that into a purely numerical feature matrix
    :param dataframe: the SF crime dataframe
    :return: a 2d numpy array representing a feature matrix
    """
    weekdays = pd.get_dummies(data=df["DayOfWeek"]).as_matrix()
    police_districts = pd.get_dummies(data=df["PdDistrict"]).as_matrix()
    resolutions = pd.get_dummies(data=df["Resolution"]).as_matrix()
    x = df["X"].as_matrix()
    x = x[:, np.newaxis]
    y = df["Y"].as_matrix()
    y = y[:, np.newaxis]
    concatenated_feature_matrix = np.concatenate((weekdays, police_districts, resolutions, x, y), axis=1)
    return concatenated_feature_matrix

''' ********** START OF SCRIPT ********** '''

df = pd.read_csv("train.csv", header=0)
feature_matrix = create_feature_matrix_and_label_vector(dataframe=df)
#print(feature_matrix)

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df["Category"])
label_vector = label_encoder.transform(df["Category"])
label_vector = label_vector.reshape((len(label_vector), 1))
#print(label_vector[:20])

#print(feature_matrix.shape)
#print(label_vector.shape)


''' ********** CLASSIFICATION ********** '''
#'''
logistic_classifier = linear_model.LogisticRegression()
logistic_classifier.fit(feature_matrix, label_vector)
score = logistic_classifier.score(feature_matrix, label_vector)

print("Logistic regression model prediction success: ", score)
#'''
#leave addresses out for now
# print(df.Address.unique()[:20])
#addresses = pd.get_dummies(data=df["Address"])
#print(addresses[:20])