import cv2
import mediapipe as mp
import time


class PoseDetector():
    def __init__(self, mode=False):
        self.mode = mode

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        #self.pose = self.mpPose.Pose(self.mode)
        self.pose = self.mpPose.Pose(self.mode, model_complexity=0)#model_complexity decides the balance between how quick the analysis of each frame is, and how accurate it is. model_complexity = 0 (which is what we currently have) means it is the least accurate but the fastest. model_complexity = 2 means the most accurate but slow. Default value is 1
        #current code assumes that frames are a part of video and not just individual frames


    def findPose(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        #print(results.pose_landmarks)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)

        return img

    def getPosition(self, img, draw=True):
        lmList= []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                #if id not in [1,2,3,4,5,6,7,8,9,10]:
                h, w, c = img.shape
                #print(id, lm)
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)


        """
        cx_head = 0
        cy_head = 0

        for i in range(1,10):
            cx_head += lmList[i][1]
            cy_head += lmList[i][2]


        lmList.append([0, cx_head, cy_head])
        """

        """
        for i in range(1,10):
            del lmList[i]
            #lmList.pop(i)
        """
        #del lmList[0]




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