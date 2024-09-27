from yolo_get_people_class import PersonDetection
from HeartAttackDetection import HearAttackModel
from PoseModule import PoseDetector #Probably just change to PoseModule
from EmotionModule import EmotionDetector
import cv2
import numpy as np
#import ...


def combine_preds(emotion_prediction, pose_prediction):
    #Gets the prediction of both the model we trained (to recognize hand on chest) as well as the emotion model that returns whether a person displays pain
    #Check before hand that there is a person in the picture (the prediction isn't None)
    if pose_prediction == None:
        return 0
    #Return an adjusted prediction where the multipliers are based on trial and error from our experience
    return 0.75 * pose_prediction + 0.25 * emotion_prediction


def multi_person(model_complexity=0, with_emotion=True):
    #Get the model complexity and with_emotion parameters:
    # model complexity - how fast we want our model to be (between 0-2). 0 is the fastest and 2 is the slowest. Also 0 is the least accurate and 2 is the most accurate,
    # with_emotion - do you want to adjust the prediction of whether the person is displaying pain into the mix

    FRAME_WIDTH = 640#This constant was hard coded by us based on the cameras on our computers
    FRAME_HEIGHT = 480 #This constant was hard coded by us based on the cameras on our computers
    CONF = 0.6 #How much confidence we want the yolo model we run to have in it's assesment that something is a human before we treat it as such
    EXTRA_MARGIN = 60 #How much extra margin we want the yolo model to take when it creates the boundries around people (the boxes)
    HEART_ATTACK_DETECTION_PATH = "model_1D/09_model_1D.h5" #Path to the model we trained for checking hands on chest

    #Initialize the different models
    #Skeleton detector analyses the skeleton of the people given in the picture
    skeleton_detector = PoseDetector(model_complexity=model_complexity)
    #Person detection extracts the people from the frame
    person_detection = PersonDetection(FRAME_WIDTH,FRAME_HEIGHT,EXTRA_MARGIN,CONF)
    #Initialize the hand on chest model based on the path
    heart_attack_detection = HearAttackModel(HEART_ATTACK_DETECTION_PATH)
    #If we want emotions to be a part of the calculation, initialize the variable as well
    if with_emotion:
        emotion_detector = EmotionDetector()

    #Start capturing the pictures from the camera
    cap = cv2.VideoCapture(0)
    while True: #Run as long as we don't stop it
        #Initialize list for predictions
        predictions = []
        emotions = []
        skeletons = []
        #Read frame from the camera
        rep,frame = cap.read()

        results = person_detection.predict(frame) #get the prediction from yolo
        frame, boxes = person_detection.plot_bboxes(results, frame) #Extract the boxes and updated frame from the frame based on the result
        people = person_detection.get_people_from_boxes(frame, boxes) #Get people from the boxes

        for i in range(len(people)):
            #Go over the people and for each person get the final prediction
            if with_emotion:
                #If we do want to use the emotions, get the result of whether the person is in pain
                emotion_prediction = emotion_detector.isPain(people[i])

            #Get the skeleton based on the pic of the person. Use draw=false (draw connects all the dots on the person's body but this obscures the face so we decided not to use it since you can't see emotion the person is displaying)
            curr_person = skeleton_detector.findPose(people[i], draw=False)
            skeleton = skeleton_detector.getPosition(people[i])

            print(np.array([skeleton]).shape,np.array([skeleton]))




            #Predict using the hands on chest model based on the skeleton. It is passed inside more arrays because the model expects an array of 9x2 coordinates so we adjust the scale
            pred = heart_attack_detection.predict(np.array([skeleton]))


            #If with emotion print the results
            if with_emotion:
                print(f'emotion_prediction = {emotion_prediction}')

            print(f'pose_prediction = {pred}')
            #Create updated prediction based on whether we want emotion or not
            if with_emotion:
                updated_pred = combine_preds(emotion_prediction, pred)
                updated_pred = round(updated_pred, 2) #Initially the prediction includes multiple numbers after the dot. Stops it at 2
            else:
                updated_pred = pred
                if updated_pred != None: #Make sure there is an actual prediction
                    updated_pred = round(updated_pred, 2) #Initially the prediction includes multiple numbers after the dot. Stops it at 2


            print(f'updated_pred = {updated_pred}')

            #Add prediction to list with the corresponding message
            predictions.append((updated_pred, heart_attack_detection.confidence_level(updated_pred)))

            #Write these predictions into the box of the person in the image
            cv2.putText(frame, str(predictions[i]), (int(boxes[i][0]), int(boxes[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)




        print(predictions)
        #Display updated image for a frame
        cv2.imshow("Image", frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()


def one_person(model_complexity=0, with_emotion=True):
    # Get the model complexity and with_emotion parameters:
    # model complexity - how fast we want our model to be (between 0-2). 0 is the fastest and 2 is the slowest. Also 0 is the least accurate and 2 is the most accurate,
    # with_emotion - do you want to adjust the prediction of whether the person is displaying pain into the mix

    HEART_ATTACK_DETECTION_PATH = "model_1D/09_model_1D.h5"#Path to the model we trained for checking hands on chest
    # Initialize the different models
    # Skeleton detector analyses the skeleton of the people given in the picture
    skeleton_detector = PoseDetector(model_complexity=model_complexity)
    #Initialize the hand on chest model based on the path
    heart_attack_detection = HearAttackModel(HEART_ATTACK_DETECTION_PATH)
    #If we want emotions to be a part of the calculation, initialize the variable as well
    if with_emotion:
        emotion_detector = EmotionDetector()

    #Start capturing the pictures from the camera
    cap = cv2.VideoCapture(0)
    while True:#Run as long as we don't stop it
        #Read frame from the camera
        rep, frame = cap.read()
        if with_emotion:
            # If we do want to use the emotions, get the result of whether the person is in pain

            emotion_prediction = emotion_detector.isPain(frame)
        #Get the skeleton based on the pic of the person. Use draw=false (draw connects all the dots on the person's body but this obscures the face so we decided not to use it since you can't see emotion the person is displaying)
        person = skeleton_detector.findPose(frame, draw=False)
        skeleton = skeleton_detector.getPosition(person)
        # Predict using the hands on chest model based on the skeleton. It is passed inside more arrays because the model expects an array of 9x2 coordinates so we adjust the scale

        pred = heart_attack_detection.predict(np.array([skeleton]))
        # If with emotion print the results
        if with_emotion:
            print(f'emotion_prediction = {emotion_prediction}')

        print(f'pose_prediction = {pred}')
        # Create updated prediction based on whether we want emotion or not

        if with_emotion:
            updated_pred = combine_preds(emotion_prediction, pred)
            updated_pred = round(updated_pred, 2)
            # Initially the prediction includes multiple numbers after the dot. Stops it at 2

        else:
            updated_pred = pred
            if updated_pred != None:#Make sure there is an actual prediction
                updated_pred = round(updated_pred, 2)
                # Initially the prediction includes multiple numbers after the dot. Stops it at 2

        print(f'updated_pred = {updated_pred}')
        #Get the correspoding message based on the final confidence level
        clasification = heart_attack_detection.confidence_level(updated_pred)


        print(f'{updated_pred}, {clasification}')
        # Write the prediction on the image and display the image for 1 frame

        cv2.putText(frame, str(updated_pred) + ", " + str(clasification), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 3)

        cv2.imshow("Image", frame)
        cv2.waitKey(1)
    cap.release()
    cv2.destroyAllWindows()

def main(multi=True, model_complexity=0, with_emotion=True):
    #Main function operates the rest of the segments.
    #Parameters:
    # multi- whether we assume there can be multiple people in the frame,
    # model complexity - how fast we want our model to be (between 0-2). 0 is the fastest and 2 is the slowest. Also 0 is the least accurate and 2 is the most accurate,
    # with_emotion - do you want to adjust the prediction of whether the person is displaying pain into the mix
    if multi:
        #If you assume there can be multiple people at once run the code for multiple people
        multi_person(model_complexity=model_complexity, with_emotion=with_emotion)
    else:
        #If you assume there can be only one person at once run the code for a single person
        one_person(model_complexity=model_complexity, with_emotion=with_emotion)


main(multi=True, with_emotion=True, model_complexity=0)