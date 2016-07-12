#!/usr/bin/env python
import pandas as pd
import numpy as np
from sklearn import preprocessing, linear_model
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import time


def create_feature_matrix(dataframe):
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
feature_matrix = create_feature_matrix(dataframe=df)

label_encoder = preprocessing.LabelEncoder()
label_encoder.fit(df["Category"])
label_vector = label_encoder.transform(df["Category"])
label_vector = label_vector.reshape((len(label_vector), 1))

TEST_PERCENT = 0.9

x_train, x_test, y_train, y_test = \
    train_test_split(feature_matrix, label_vector,
                     test_size=TEST_PERCENT)
''' ********** CLASSIFICATION ********** '''
time1 = time.time()
logistic_classifier = linear_model.LogisticRegression()
logistic_classifier.fit(x_train, y_train)
time2 = time.time()
print('took %0.2f s' % ((time2 - time1)))
model_score = logistic_classifier.score(x_train, y_train)
print("Logistic regression model prediction success: ", model_score)


















