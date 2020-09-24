# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np

dataset = pd.read_csv("heart_failure_clinical_records_dataset.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1:].values

from sklearn.preprocessing import MinMaxScaler

X_scaler = MinMaxScaler()
X = X_scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(units = 8, kernel_initializer = "uniform", activation = "relu", input_dim = 12))
classifier.add(Dense(units = 8, kernel_initializer = "uniform", activation = "relu"))
classifier.add(Dense(units = 1, activation = "sigmoid"))

classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])

classifier.fit(X_train, y_train, epochs = 1000, batch_size = 32)

y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
