import numpy as np
import os
import tensorflow as tf
from keras.src.layers import Conv1D
from supervision.dataset.utils import train_test_split
from sympy import false

"""from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D"""

#from tensorflow.keras import layers, models

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten, InputLayer

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

    return np.array([float(point[0]), float(point[1]), float(point[2])])

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

    skel_none = np.empty((df_none.shape[0], df_none.shape[1], 3))
    for i in range(df_none.shape[0]):
        for j in range(df_none.shape[1]):
            temp = convert(df_none.iloc[i][j])
            skel_none[i][j][0] = temp[0]
            skel_none[i][j][1] = temp[1]
            skel_none[i][j][2] = temp[2]


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

    skel_inf = np.empty((df_inf.shape[0], df_inf.shape[1], 3))
    for i in range(df_inf.shape[0]):
        for j in range(df_inf.shape[1]):
            temp = convert(df_inf.iloc[i][j])
            skel_inf[i][j][0] = temp[0]
            skel_inf[i][j][1] = temp[1]
            skel_inf[i][j][2] = temp[2]

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


path_train_none = 'trainNoneDataZ.csv'
path_train_inf = 'trainInfarctDataZ.csv'

path_val_none = 'valNoneDataZ.csv'
path_val_inf = 'valInfarctDataZ.csv'

#path_test_none = 'testNoneData.csv'
#path_test_inf = 'testInfarctData.csv'

skel_train, labels_train = prepare_data(path_train_none, path_train_inf)
skel_val, labels_val = prepare_data(path_val_none, path_val_inf)
#skel_test, labels_test = prepare_data(path_test_none, path_test_inf, is_neck=True)

NUM_CHECK_POINT = 10
EPOCH_CHECK_POINT = 2  # How many epoch til save next checkpoint


OUTPUT_DIR = "./model_1D_Z/"
if not os.path.exists(OUTPUT_DIR):
    os.mkdir(OUTPUT_DIR)


# THE NN
K.clear_session()

model = Sequential()
model.add(Conv1D(4, 2, strides=1, padding='valid', activation='relu', input_shape=(9, 3)))
model.add(Conv1D(8, 3, strides=1, padding='same', activation='relu'))

model.add(Conv1D(16, 3, strides=1, padding='same', activation='relu'))
#model.add(Conv2D(1, (3, 3), strides=(1, 1), padding='same', activation='relu'))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

for i in range(NUM_CHECK_POINT):
    history = model.fit(skel_train, labels_train,
                        epochs=EPOCH_CHECK_POINT,
                        #batch_size=32,
                        validation_data=(skel_val, labels_val))

    print('Saving model: {:02}.'.format(i))
    model.save(OUTPUT_DIR + "{:02}_model_1D_Z.h5".format(i))











"""
model = Sequential()

# Add input layer
model.add(InputLayer(input_shape=(9, 2, 1)))  # Adjust input shape as needed

# Add convolutional layers
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))

# Add dropout layer
model.add(Dropout(0.5))

# Flatten the output
model.add(Flatten())

# Add dense layers
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))

# Add output layer
model.add(Dense(1, activation='sigmoid'))"""






"""model = Sequential()
model.add(Conv2D(128, (3, 3), activation='relu', input_shape=(9, 2, 1)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))"""


"""model = models.Sequential([
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(9, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(9, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', input_shape=(9, 2)),
    layers.Dropout(0.5),
    layers. Flatten(),
    #layers.Dense(128, activation='relu', input_shape=(skel_train.shape[1],)),
    layers.Dense(128, activation='relu'),

    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid'),

    #layers.Dense(128, activation='relu', input_shape=(skel_train.shape[1],)),
    #layers.Dropout(0.5),
    #layers.Dense(64, activation='relu'),
    #layers.Dropout(0.5),
    #layers.Dense(1, activation='sigmoid')
])
"""


"""for i in range(1, df_none.shape[0]):
    for j in range(1, df_none.shape[1]):
        df_none.iloc[i,j] = convert(df_none.iloc[i,j])

skel_train_none = df_none.iloc[1:,1:].to_numpy()
print("conversion for none done")"""

"""print(skel_train_none)
print(skel_train_none.shape)
print(skel_train_none[0][0])
print(skel_train_none[0][0][0])"""


"""
df_inf = pd.read_csv('trainNoneData.csv', header=None) # the DataFrame
for i in range(1, df_inf.shape[0]):
    for j in range(1, df_inf.shape[1]):
        df_inf.iloc[i,j] = convert(df_inf.iloc[i,j])
skel_train_inf = df_none.iloc[1:,1:].to_numpy()
print("conversion for inf done")






df2 = df_none.iloc[:4,:]
str = df2.iloc[0,0]"""


"""
data = {'points': ['[[1.1, 2.2], [3.3, 4.4]]', '[[5.5, 6.6], [7.7, 8.8]]']}
df = pd.DataFrame(data)
print(df)"""


"""skel_train_none = df_none.to_numpy()
print(skel_train_none)
print(skel_train_none[0][0])
print(type(skel_train_none[0][0]))"""

"""
"""


