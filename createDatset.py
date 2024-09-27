import cv2
import time
import PoseModule as pm
import os
import numpy as np
import pandas as pd

def create_data_frame(points,paths):  #  create a dataFrame containing the skeletons list with the image paths as names for the skeletons
    #Points and paths should be of the same length
    #Points should be numpy arrays
    #Points should still have in each point the index as well as x,y,z
    #remove_index = points[:,:,1:]
    names = ["index" + str(i) for i in range(points.shape[1])]
    df = pd.DataFrame.from_records(points, columns=names)
    df.insert(0, "path", paths)
    return df


path = "dataset/test/01Infarct"
detector = pm.PoseDetector(True)  # mode=True so it will treat the images as static and not video stream
pointsList = np.empty((0, 9, 3))  # initialize empty array, to contain 9 points each of 3 values (this version was designed to handle x,y,z coordinates)
print(pointsList.shape)
print(pointsList)

img = 0
paths = os.listdir(path)  # this is the list of paths of all the images in the dataset

for img_path in os.listdir(path):  # iterate through each image path
    img = cv2.imread(path + "/" + img_path)  # read the image

    img = detector.findPose(img)
    temp = detector.getPosition(img)  # the landmarks of the current image

    temp = temp[:, 1:]  # this is from the time we saved also the id's of the landmarks, so we want discard them
    temp = np.array([temp])  # for the concatenate
    if(len(temp[0]) != 0):
        pointsList = np.concatenate((pointsList, temp), axis=0)  # add the current skeleton points to the list
    else:
        paths.remove(img_path)  # if a skeleton was not identified, dont add the points and remove the current image's path from the paths list

    print(img_path + ": ")
    print(temp)

df = create_data_frame(pointsList, paths)
df.to_csv('testInfarctDataZ.csv', index=False)  # save the dataFrame as .csv file
