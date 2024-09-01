import cv2
import mediapipe as mp
import time
import numpy as np
from scipy import ndimage

class PoseDetector():
    def __init__(self, mode=False):
        self.mode = mode

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        #self.pose = self.mpPose.Pose(self.mode)
        self.pose = self.mpPose.Pose(self.mode, model_complexity=2)


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        """#
        img = cv2.circle(img, (1, 1), radius=2, color=(0, 0, 255), thickness=-10)
        #"""

        return img

    def getPosition(self, img, draw=True):

        #lmList = np.empty((0, 4), int)
        lmList = np.empty((0, 3))



        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                #if id not in [1,2,3,4,5,6,7,8,9,10]:
                if id in [0,11,12,13,14,15,16,23,24]:

                    h, w, c = img.shape

                    #print(id, lm)

                    #cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)

                    cx, cy = int(lm.x * w), int(lm.y * h)
                    #cx, cy = int(lm.x), int(lm.y)



                    #lmList = np.append(lmList, [[id, cx, cy, cz]], axis=0)
                    lmList = np.append(lmList, [[id, cx, cy]], axis=0)


                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

            nose = lmList[0, 1:3]
            left_shoulder = lmList[1, 1:3]
            right_shoulder = lmList[2, 1:3]
            neck = (nose + left_shoulder + right_shoulder) / 3
            cx_neck, cy_neck= neck[0], neck[1]

            lmList = np.append(lmList, [[33, cx_neck, cy_neck]], axis=0) # estimate neck as the mean of the nose and the 2 shoulders

            # translation (mean):
            for point in lmList:
                point[1] = point[1] - neck[0]
                point[2] = point[2] - neck[1]

            # Dont forget Rotation !!!


            dist_to_left_shoulder = np.linalg.norm(neck - left_shoulder) # not the z
            dist_to_right_shoulder = np.linalg.norm(neck - right_shoulder)
            furthest_shoulder = [0,0]
            if(dist_to_left_shoulder > dist_to_right_shoulder):
                furthest_shoulder = left_shoulder
            else:
                furthest_shoulder = right_shoulder


            #furthest_shoulder = left_shoulder




            temp = (furthest_shoulder[1] - neck[1]) / (furthest_shoulder[0] - neck[0])
            alpha = np.arctan([temp])
            alpha = alpha[0]

            #alpha = abs(alpha)
            #alpha = -alpha

            """#
            alpha = 0.1
            #"""

            print("before Rotation:")
            print(lmList)
            #their rotation:
            """for point in lmList:
                #point[1] = (point[1]) * np.cos(alpha) # for x
                #point[2] = (point[2]) * np.sin(alpha) # for x

                point[1] = (point[1] - neck[0]) * np.cos(alpha) + neck[0] # for x
                point[2] = (point[2] - neck[1]) * np.sin(alpha) + neck[1] # for y"""

            #scipy rotation:
            """for point in lmList:

                point[1:3] = ndimage.rotate(point[1:3], alpha)"""
            #alpha must be in degrees
            #lmList[:,1:3] = ndimage.rotate(lmList[:,1:3], alpha)

            #rotation matrix rotation:
            """rotMatrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                                     [np.sin(alpha), np.cos(alpha)]])"""



            #alpha = 45 * (np.pi / 180)
            for point in lmList:
                """temp = point[1:3].T
                point[1:3] = (rotMatrix @ temp).T"""
                x = point[1]
                y = point[2]
                """point[1] = (x - neck[0]) * np.cos(alpha) - (y - neck[1]) * np.sin(alpha) + neck[0]
                point[2] = (x - neck[0]) * np.sin(alpha) + (y - neck[1]) * np.cos(alpha) + neck[1]"""
                point[1] = x * np.cos(alpha) - y * np.sin(alpha)
                point[2] = x * np.sin(alpha) + y * np.cos(alpha)

            # Normalization (already done translation)
            for point in lmList:
                point[1] = point[1] / np.linalg.norm(neck - furthest_shoulder)
                point[2] = point[2] / np.linalg.norm(neck - furthest_shoulder)

            # to show on the screen:
            """for point in lmList:
                point[1] = point[1] + neck[0]
                point[2] = point[2] + neck[1]"""





            # Normalisation of the Distances:

            """for point in lmList:
                dist_to_left_shoulder = np.linalg.norm(neck - left_shoulder)
                dist_to_right_shoulder = np.linalg.norm(neck - right_shoulder)

                dist_to_furthest_shoulder = max(dist_to_left_shoulder, dist_to_right_shoulder)
                point[1:4] = (point[1:4] - neck) / dist_to_furthest_shoulder"""

            # jhgkfg
            """print("left soulder: ")
            print(left_shoulder)

            print("right soulder: ")
            print(right_shoulder)

            print("alpha: ")
            print(alpha)"""

            # REMEMBER: THE Y AXIS GOES DOONWARDS, SO THE Y OF THE NOSE IS ACTUALLY < THE Y OF THE SHOULDERS, FOR EXAMPLE.



        return lmList

def main():
    cap = cv2.VideoCapture('a.mp4')
    pTime = 0
    detector = PoseDetector()
    while True:
        success, img = cap.read()
        img = detector.findPose(img)
        lmList = detector.getPosition(img)
        print(lmList)

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()