from fer import FER


class EmotionDetector:
    SURPRISE_THRESHOLD = 0.2 #0.35
    ANGRY_THRESHOLD = 0.12   #0.3
    FEAR_THRESHOLD = 0.3     #0.35
    NEUTRAL_THRESHOLD = 0.5  #0.5  # lower threshold

    def __init__(self):
        self.emotion_detector = FER()

    def isPain(self, img):
        result = self.emotion_detector.detect_emotions(img)

        if (len(result) != 0):
            emotions = result[0]['emotions']
            if (emotions["surprise"] >= self.SURPRISE_THRESHOLD or emotions["angry"] >= self.ANGRY_THRESHOLD or emotions["fear"] >= self.FEAR_THRESHOLD) and emotions['neutral'] <= self.NEUTRAL_THRESHOLD:
                return True
        return False
