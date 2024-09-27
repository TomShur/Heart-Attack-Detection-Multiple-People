import cv2
import mediapipe as mp
import numpy as np


# for further details on the mediapipe pose estimation, see:
# [https://github.com/google-ai-edge/mediapipe/blob/master/docs/solutions/pose.md]


class PoseDetector:  # this is a class that is used to initialize a detector to output the skeleton of the person in the frame

    # these are the id's of the relevant points
    # the mediapipe model outputs 33 points, but we only need 9 of them
    LANDMARKS = [0, 11, 12, 13, 14, 15, 16, 23, 24]
    # 0:nose 11:left_shoulder 12:right_shoulder 13:left_elbow 14:right_elbow 15:left_wrist 16:right_wrist 23:left_hip 24:right_hip

    def __init__(self, mode=False, model_complexity=0):  # initialize the detector
        self.MODEL_COMPLEXITY = model_complexity  # a natural number between 0 and 2
        # the higher the value the more accurate the model is, but also slower

        self.mode = mode  # if set to false, the model treats the input images as a video stream
        # else, the model treats the input images as unrelated static images

        self.mpDraw = mp.solutions.drawing_utils  # to draw the points on the image

        self.mpPose = mp.solutions.pose  # this is the mediapipe pose solution

        self.pose = self.mpPose.Pose(self.mode, model_complexity=self.MODEL_COMPLEXITY)  # the mediapipe pose model, initialized with the wanted parameters

    def calculate_neck(self, nose, left_shoulder, right_shoulder):  # estimate the neck point as the mean of the nose and the shoulders
        return (nose + left_shoulder + right_shoulder) / 3

    def center(self, points, neck):  # centralize the points around the neck
        centered_points = points
        for point in centered_points:
            point[0] = point[0] - neck[0]
            point[1] = point[1] - neck[1]
        return centered_points

    def calculate_furthest_shoulder(self, neck, left_shoulder, right_shoulder):  # calculate the furthest shoulder from the neck
        dist_to_left_shoulder = np.linalg.norm(neck - left_shoulder)
        dist_to_right_shoulder = np.linalg.norm(neck - right_shoulder)
        furthest_shoulder = 0
        if (dist_to_left_shoulder > dist_to_right_shoulder):
            furthest_shoulder = left_shoulder
        else:
            furthest_shoulder = right_shoulder
        return furthest_shoulder

    def calculate_angle(self, neck, furthest_shoulder):  # calculate the angle between the neck and the furthest shoulder
        temp = (furthest_shoulder[1] - neck[1]) / (furthest_shoulder[0] - neck[0])
        alpha = np.arctan([temp])
        alpha = alpha[0]
        return alpha

    def rotate(self, points, alpha):  # rotate the points around the given angle alpha
        rotated_points = points
        for point in rotated_points:
            x = point[0]
            y = point[1]
            point[0] = x * np.cos(alpha) - y * np.sin(alpha)
            point[1] = x * np.sin(alpha) + y * np.cos(alpha)
        return rotated_points

    def normalize(self, points, neck, furthest_shoulder):  # normalize the points by the distance from the neck to the furthest shoulder
        normalized_points = points
        for point in normalized_points:
            point[0] = point[0] / np.linalg.norm(neck - furthest_shoulder)
            point[1] = point[1] / np.linalg.norm(neck - furthest_shoulder)
        return normalized_points

    def findPose(self, img, draw=True):  # draw the skeleton on the image
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert from BRG to RGB
        self.results = self.pose.process(imgRGB)  # get the landmarks
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                # the POSE_CONNECTIONS is for drawing the edges between the landmarks
        return img

    def getPosition(self, img, draw=True):  # return the landmarks of the skeleton
        lmList = np.empty((0, 2))  # initialize empty array of 2 values in each cell, this will be the list containing the landmarks
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):  # iterate through every landmark
                if id in self.LANDMARKS:  # take only the desired landmarks
                    h, w, c = img.shape  # h and w are the height and width of the image
                    cx, cy = int(lm.x * w), int(lm.y * h)  # the x and y coordinates normalized by the height and width of the image respectively
                    lmList = np.append(lmList, [[cx, cy]], axis=0)  # add the last calculated landmark to the list
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)  # draw the landmarks

            nose = lmList[0]
            left_shoulder = lmList[1]
            right_shoulder = lmList[2]
            neck = self.calculate_neck(nose, left_shoulder, right_shoulder)
            lmList = self.center(lmList, neck)  # centralize before the other calculations, to avoid being affected by the initial coordinate system
            furthest_shoulder = self.calculate_furthest_shoulder(neck, left_shoulder, right_shoulder)
            alpha = self.calculate_angle(neck, furthest_shoulder)
            lmList = self.rotate(lmList, alpha)
            lmList = self.normalize(lmList, neck, furthest_shoulder)


        return lmList