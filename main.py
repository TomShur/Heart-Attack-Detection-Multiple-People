from yolo_get_people_class import PersonDetection
from HeartAttackDetection import HearAttackModel
from PoseModule import PoseDetector #Probably just change to PoseModule
from EmotionModule import EmotionDetector
import cv2
import numpy as np
#import ...

def combine_preds(emotion_prediction, pose_prediction):
    if pose_prediction == None:
        return 0
    return 0.75 * pose_prediction + 0.25 * emotion_prediction


def multi_person(model_complexity=0, with_emotion=True):
    #Maybe create different main function for if we assume that there is only one person
    #Maybe these constants should be parameters or maybe they should be const of the file
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480 #Maybe python can calculate these values somehow
    CONF = 0.6
    EXTRA_MARGIN = 60
    HEART_ATTACK_DETECTION_PATH = "model_1D/09_model_1D.h5"
    skeleton_detector = PoseDetector(model_complexity=model_complexity)#Maybe should be per skeleton and person, who knows
    person_detection = PersonDetection(FRAME_WIDTH,FRAME_HEIGHT,EXTRA_MARGIN,CONF)
    heart_attack_detection = HearAttackModel(HEART_ATTACK_DETECTION_PATH)
    if with_emotion:
        emotion_detector = EmotionDetector()

    #pain_detection = PainDetection()

    cap = cv2.VideoCapture(0)
    while True: #Probably shouldn't be while True, probably want to be able to stop it
        #Maybe should be numpy array
        predictions = []        #Maybe should be numpy array
        emotions = []         #Maybe should be numpy array
        skeletons = [] #Maybe should be numpy array
        rep,frame = cap.read() #Maybe add check here if rep is false, close program
        #frame,people = person_detection.get_people(frame)

        results = person_detection.predict(frame)
        frame, boxes = person_detection.plot_bboxes(results, frame)
        people = person_detection.get_people_from_boxes(frame, boxes)

        # skeleton_detector.extract_head(person) # Most likely don't need it as passing 1 person at a time will work fine
        # skeletons maybe #Maybe list of skeletons so we can interact with them and pass emotions above shoulders
        #in_pain_lst = []
        for i in range(len(people)):

            if with_emotion:
                emotion_prediction = emotion_detector.isPain(people[i])


            curr_person = skeleton_detector.findPose(people[i], draw=False)
            skeleton = skeleton_detector.getPosition(people[i])
            #img = detector.findPose(frame)
            #lmList = detector.getPosition(img)
            #skeletons.append(skeleton)
            print(np.array([skeleton]).shape,np.array([skeleton]))
            #in_pain_lst.append(pain_detection(person))






            pred = heart_attack_detection.predict(np.array([skeleton]))



            if with_emotion:
                print(f'emotion_prediction = {emotion_prediction}')

            print(f'pose_prediction = {pred}')

            if with_emotion:
                updated_pred = combine_preds(emotion_prediction, pred)
                updated_pred = round(updated_pred, 2)
            else:
                updated_pred = pred
                if updated_pred != None:
                    updated_pred = round(updated_pred, 2)


            print(f'updated_pred = {updated_pred}')



            predictions.append((updated_pred, heart_attack_detection.confidence_level(updated_pred)))

            cv2.putText(frame, str(predictions[i]), (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            #Also use threading for prediction and emotion recognition as well
            """emotions.append(emotion_detection(person))#Maybe should pass it
            if (emotion == "anger" or emotion == "sadness") and predicton == positive:
                draw_on_frame("Heart attack")"""


        print(predictions)#,heart_attack_detection.confidence_level(predictions[0]))
        #print(in_pain_lst)


        #cv2.putText(frame, str(predictions), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 3)

        cv2.imshow("Image", frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()

def one_person(model_complexity=0, with_emotion=True):
    HEART_ATTACK_DETECTION_PATH = "model_1D/09_model_1D.h5"
    skeleton_detector = PoseDetector(model_complexity=model_complexity)
    heart_attack_detection = HearAttackModel(HEART_ATTACK_DETECTION_PATH)

    if with_emotion:
        emotion_detector = EmotionDetector()


    cap = cv2.VideoCapture(0)
    while True:
        rep, frame = cap.read()
        if with_emotion:
            emotion_prediction = emotion_detector.isPain(frame)

        person = skeleton_detector.findPose(frame, draw=False)
        skeleton = skeleton_detector.getPosition(person)

        pred = heart_attack_detection.predict(np.array([skeleton]))

        if with_emotion:
            print(f'emotion_prediction = {emotion_prediction}')

        print(f'pose_prediction = {pred}')
        if with_emotion:
            updated_pred = combine_preds(emotion_prediction, pred)
            updated_pred = round(updated_pred, 2)
        else:
            updated_pred = pred
            if updated_pred != None:
                updated_pred = round(updated_pred, 2)

        print(f'updated_pred = {updated_pred}')

        clasification = heart_attack_detection.confidence_level(updated_pred)


        print(f'{updated_pred}, {clasification}')

        cv2.putText(frame, str(updated_pred) + ", " + str(clasification), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 3)

        cv2.imshow("Image", frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()

def main(multi=True, model_complexity=0, with_emotion=True):
    if multi:
        multi_person(model_complexity=model_complexity, with_emotion=with_emotion)
    else:
        one_person(model_complexity=model_complexity, with_emotion=with_emotion)

#Add code here to close cap

main(multi=True, with_emotion=False, model_complexity=2)
