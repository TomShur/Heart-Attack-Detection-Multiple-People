from yolo_get_people_class import PersonDetection
from HeartAttackDetection import HearAttackModel
from PoseModuleTom import PoseDetector #Probably just change to PoseModule
import cv2
import numpy as np

def main():
    #Maybe create different main function for if we assume that there is only one person
    #Maybe these constants should be parameters or maybe they should be const of the file
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    CONF = 0.6
    EXTRA_MARGIN = 60
    HEART_ATTACK_DETECTION_PATH = "model/09_model_1D.h5"
    skeleton_detector = PoseDetector()#Maybe should be per skeleton and person, who knows
    person_detection = PersonDetection(CONF,EXTRA_MARGIN)
    heart_attack_detection = HearAttackModel(HEART_ATTACK_DETECTION_PATH)
    cap = cv2.VideoCapture(0)
    while True: #Probably shouldn't be while True, probably want to be able to stop it
        #Maybe should be numpy array
        predictions = []        #Maybe should be numpy array
        emotions = []         #Maybe should be numpy array
        skeletons = [] #Maybe should be numpy array
        rep,frame = cap.read() #Maybe add check here if rep is false, close program
        frame,people = person_detection.get_people(frame)
        # skeleton_detector.extract_head(person) # Most likely don't need it as passing 1 person at a time will work fine
        # skeletons maybe #Maybe list of skeletons so we can interact with them and pass emotions above shoulders

        for person in people:
            curr_person = skeleton_detector.findPose(person)
            skeleton = skeleton_detector.getPosition(person)
            #img = detector.findPose(frame)
            #lmList = detector.getPosition(img)
            #skeletons.append(skeleton)
            print(np.array([skeleton]).shape,np.array([skeleton]))
            predictions.append(heart_attack_detection.predict(np.array([skeleton])))
            #Also use threading for prediction and emotion recognition as well
            """emotions.append(emotion_detection(person))#Maybe should pass it
            if (emotion == "anger" or emotion == "sadness") and predicton == positive:
                draw_on_frame("Heart attack")"""
        print(predictions)
        cv2.imshow("Image", frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


main()
#Add code here to close cap