import numpy as np
import cv2

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


import cv2
import time
import PoseModule as pm

"""cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.PoseDetector()
while True:
    success, img = cap.read()
    #img = cv2.imread("dataset/train/00None/_531_9994162.png")
    img = detector.findPose(img)
    lmList = detector.getPosition(img)
    pointsList = [i[1:] for i in lmList]
    print(pointsList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    left_wrist = Point(pointsList[14][0],pointsList[14][1])
    right_wrist = Point(pointsList[15][0],pointsList[15][1])
    polygon = Polygon([pointsList[10],pointsList[11],pointsList[22] ,pointsList[23]])
    print(polygon.contains(left_wrist) or polygon.contains(right_wrist))
    print(polygon.boundary)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
#point = Point(0.5, 0.5)
#Add drawing the polygon and the point we want to get the answer for"""

"""import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import PoseModule as pm
detector = pm.PoseDetector()

# Load the image
image = cv2.imread("dataset/train/00None/_531_9994162.png")
image = detector.findPose(image)
lmList = detector.getPosition(image)
pointsList = [i[1:] for i in lmList]
print(pointsList)
vertices = [pointsList[12], pointsList[11], pointsList[23], pointsList[24]]


codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO]
path = Path(vertices, codes)

# Convert the Path to a set of points
points = path.vertices.astype(np.int32)

# Draw the path on the image
cv2.polylines(image, [points], isClosed=True, color=(255, 160, 0), thickness=3)
#cv2.circle(image,tuple(pointsList[11][0]+10,pointsList[11][]),)
# Convert the image to RGB (since OpenCV loads images in BGR)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the image with the path overlaid
plt.imshow(image_rgb)
plt.axis('off')
plt.show()
"""
import matplotlib.path as mpath
import matplotlib.pyplot as plt

def inpolygon(x, y, polygon):
    path = mpath.Path(polygon)
    return path.contains_point((x, y))

# Example usage:



cap = cv2.VideoCapture(0)
pTime = 0
detector = pm.PoseDetector()
while True:
    success, img = cap.read()
    #img = cv2.imread("dataset/train/00None/_531_9994162.png")
    img = detector.findPose(img)
    lmList = detector.getPosition(img)
    pointsList = [i[1:] for i in lmList]
    print(pointsList)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
    #left_wrist = set(pointsList[14][0],pointsList[14][1])
    #right_wrist = set(pointsList[15][0],pointsList[15][1])
    polygon = [pointsList[12], pointsList[11], pointsList[23], pointsList[24]] #Code may be sensitive to the order in which the points here in the polygon are entered
    hands_on_chest = inpolygon(pointsList[15][0],pointsList[15][1],polygon) or inpolygon(pointsList[16][0],pointsList[16][1],polygon)
    print(hands_on_chest)
    cv2.putText(img, str(hands_on_chest), (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

    #cv2.polylines(img, [polygon], isClosed=True, color=(255, 0, 0), thickness=3)

    # Convert the image to RGB (since OpenCV loads images in BGR)
    #image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Display the image with the path overlaid
    #plt.imshow(image_rgb)
    #plt.imshow(image_rgb)
    #plt.axis('off')
    #plt.show()
    cv2.imshow("Image", img)
    cv2.waitKey(1)
#point = Point(0.5, 0.5)
#Add drawing the polygon and the point we want to get the answer for
#Only works for one hand -nevermind fixed. Should note that it currently checks if the wrist is in the polygon and not a more sophisticated average of the hand, also the polygon is shoulders and hips and not something like קטע אמצעים which is more accurate for the area of the hand??? also code crashes when it doesn't recognize a person
#Code may be sensitive to the order in which the points here in the polygon are entered