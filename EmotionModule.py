from fer import FER

#Class for detecing pain in an image based on the FER package which returns a dictionary with the distribution of the emotion
#Recognizes these 7 emotions:surprise,anger,fear,neutral,happiness,disgust,sadness


class EmotionDetector:
    #These constants represent the thresholds for these emotions which we determined best represent pain
    SURPRISE_THRESHOLD = 0.4
    ANGRY_THRESHOLD = 0.25
    FEAR_THRESHOLD = 0.3
    NEUTRAL_THRESHOLD = 0.5

    def __init__(self):
        #Initialize the emotion detector
        self.emotion_detector = FER()

    def isPain(self, img):
        #Function gets an image of a person and returns whether it deems the face the person is making to be pain
        result = self.emotion_detector.detect_emotions(img) #Get the result of the model

        if (len(result) != 0): #As long as the model does recognize a person
            emotions = result[0]['emotions'] #Get the emotional distribution of the picture
            if (emotions["surprise"] >= self.SURPRISE_THRESHOLD or emotions["angry"] >= self.ANGRY_THRESHOLD or emotions["fear"] >= self.FEAR_THRESHOLD) and emotions['neutral'] <= self.NEUTRAL_THRESHOLD:
                #This if statement was decided by us as a result of trial and error to represent whether a person is displaying pain or not
                return True
        return False