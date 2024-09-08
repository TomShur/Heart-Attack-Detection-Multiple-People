import numpy as np
import os
import tensorflow as tf
from supervision.dataset.utils import train_test_split
from sympy import false

"""from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D"""

#from tensorflow.keras import layers, models

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, InputLayer
from tensorflow.keras.models import load_model

import pandas as pd
from sklearn.utils import shuffle
from keras import backend as K


def convert(string):
    #string = ast.literal_eval(string)
    #arr = string.split("[")[1].split("]")[0].split()

    point = string.split("[")
    point = point[1]
    point = point.split("]")
    point = point[0]
    point = point.split()

    #return np.array([float(point[0]), float(point[1]), float(point[2])])
    return np.array([float(point[0]), float(point[1])])


def prepare_data(path_none, path_inf, is_neck=false):
    # None:
    df_none = pd.read_csv(path_none, header=0)  # the DataFrame
    if(is_neck):
        #df_none = df_none.drop([10], axis=1)
        df_none = df_none.iloc[:, :10]

    df_none = df_none.iloc[:, 1:]
    print("none:")
    print(df_none)
    print("************")

    skel_none = np.empty((df_none.shape[0], df_none.shape[1], 2))
    for i in range(df_none.shape[0]):
        for j in range(df_none.shape[1]):
            temp = convert(df_none.iloc[i][j])
            skel_none[i][j][0] = temp[0]
            skel_none[i][j][1] = temp[1]
            #skel_none[i][j][2] = temp[2]


    print(skel_none)
    print(skel_none.shape)

    # Inf:
    df_inf = pd.read_csv(path_inf, header=0)  # the DataFrame
    if (is_neck):
        #df_inf = df_inf.drop([10], axis=1)
        df_inf = df_inf.iloc[:, :10]

    df_inf = df_inf.iloc[:, 1:]
    print("inf:")
    print(df_inf)
    print("************")

    skel_inf = np.empty((df_inf.shape[0], df_inf.shape[1], 2))
    for i in range(df_inf.shape[0]):
        for j in range(df_inf.shape[1]):
            temp = convert(df_inf.iloc[i][j])
            skel_inf[i][j][0] = temp[0]
            skel_inf[i][j][1] = temp[1]
            #skel_inf[i][j][2] = temp[2]

    print(skel_inf)
    print(skel_inf.shape)

    skel_train = np.concatenate((skel_none, skel_inf))

    print(f"data shape = {skel_train.shape}")

    labels = np.concatenate((np.zeros(skel_none.shape[0]), (np.ones(skel_inf.shape[0]))))
    print(f"labels.shape={labels.shape}")
    skel_train, labels = shuffle(skel_train, labels, random_state=20)

    print(f"len = {len(skel_train)}")
    print("data:")
    print(skel_train)
    print("labels:")
    print(labels)
    print("***********")

    return skel_train, labels

"""
path_test_none = 'testNoneDataZ.csv'
path_test_inf = 'testInfarctDataZ.csv'

skel_test, labels_test = prepare_data(path_test_none, path_test_inf, is_neck=True)

for i in range(10):
    model = load_model('model_1D_Z/0' + str(i) + '_model_1D_Z.h5')
    loss, accuracy = model.evaluate(skel_test, labels_test)
    print(f'Z {i} Test Loss: {loss}')
    print(f'Z {i} Test Accuracy: {accuracy}')

"""

path_test_none = 'testNoneData.csv'
path_test_inf = 'testInfarctData.csv'

skel_test, labels_test = prepare_data(path_test_none, path_test_inf, is_neck=True)

for i in range(10):
    model = load_model('model_1D/0' + str(i) + '_model_1D.h5')
    loss, accuracy = model.evaluate(skel_test, labels_test)
    print(f'2D {i} Test Loss: {loss}')
    print(f'2D {i} Test Accuracy: {accuracy}')