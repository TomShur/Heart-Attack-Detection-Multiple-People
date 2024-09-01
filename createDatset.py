import cv2
import time
import PoseModule as pm
import os
import numpy as np
import pandas as pd

def create_data_frame(points,paths):
    #Points and paths should be of the same length
    #Points should be numpy arrays
    #Points should still have in each point the index as well as x,y,z
    #remove_index = points[:,:,1:]
    names = ["index" + str(i) for i in range(points.shape[1])]
    df = pd.DataFrame.from_records(points, columns=names)
    df.insert(0, "path", paths)
    return df


train_none_path = "dataset/test/01Infarct"
detector = pm.PoseDetector(True) # mode=True so it will treat the images as static and not video stream
pointsList = np.empty((0, 10, 2))
print(pointsList.shape)
print(pointsList)

img = 0
paths = os.listdir(train_none_path)

for img_path in os.listdir(train_none_path):
    img = cv2.imread(train_none_path + "/" + img_path)

    #print(type(img))
    img = detector.findPose(img)
    temp = detector.getPosition(img)
    #cv2.imshow("Image", img)
    #cv2.waitKey(0)

    temp = temp[:, 1:]
    temp = np.array([temp])
    #print(temp.shape)
    if(len(temp[0]) != 0):
        pointsList = np.concatenate((pointsList, temp), axis=0)
    else:
        paths.remove(img_path)


    #print("pointsList:")
    #print(pointsList)


    print(img_path + ": ")
    print(temp)

df=create_data_frame(pointsList, paths)
df.to_csv('testInfarctData.csv', index=False)




"""
#print(os.listdir(train_none_path))

#img_path = os.listdir(train_none_path)[0]
img_path = "_531_9994162.png"
print(img_path)
img = cv2.imread(train_none_path + "/" + img_path)
#img = cv2.imread("dataset/train/00None/_0_180076.png")

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""



#print(pointsList)
"""
cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()"""



"""img = cv2.imread("falldataset/images/train/fall001.jpg")
#print(type(img))
img = detector.findPose(img)
temp = detector.getPosition(img)
pointsList.append(temp)

print(pointsList[0])
print(len(pointsList))
print(len(pointsList)/33)

cv2.imshow('img', img)
cv2.waitKey(0)

#closing all open windows
cv2.destroyAllWindows()


#print(len("fall_dataset/images/train"))"""