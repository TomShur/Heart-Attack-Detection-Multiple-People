import numpy as np
import os
from fer import FER
import cv2
from sklearn.utils import shuffle
import random

def emotions_to_array(pred):
    return np.array([pred['angry'],pred['disgust'],pred['fear'],pred['happy'],pred['sad'],pred['surprise'],pred['neutral']])

def prepare_data():
    emotion_detector = FER()
    NUM_EMOTIONS = 7

    path_pain = "images_pain"
    paths = os.listdir(path_pain)
    pain_preds = np.empty(NUM_EMOTIONS)
    for img_path in paths:
        img = cv2.imread(img_path)
        cur_preds = emotion_detector.detect_emotions(path_pain + "/" + img_path)
        if len(cur_preds) != 0:
            cur_preds = cur_preds[0]['emotions']
            cur_preds = emotions_to_array(cur_preds)
            pain_preds = np.append(pain_preds, cur_preds, axis=0)

    print("done pain preds")

    path_neutral = "images_neutral"
    paths = os.listdir(path_neutral)
    neutral_preds = np.empty(NUM_EMOTIONS)
    for img_path in paths:
        img = cv2.imread(img_path)
        cur_preds = emotion_detector.detect_emotions(path_neutral + "/" + img_path)
        if len(cur_preds) != 0:
            cur_preds = cur_preds[0]['emotions']
            cur_preds = emotions_to_array(cur_preds)
            neutral_preds = np.append(neutral_preds, cur_preds, axis=0)

    print("done neutral preds")


    preds = np.concatenate((neutral_preds, pain_preds))
    labels = np.concatenate((np.zeros(neutral_preds.shape[0]), (np.ones(pain_preds.shape[0])))) # watch that 0 for no pain and 1 for pain
    preds, labels = shuffle(preds, labels, random_state=20)
    print(f"preds.shape={preds.shape}")
    print(f"labels.shape={labels.shape}")

    return preds, labels


def choose_thresholds():
    """search_space = {
        'anger': (0.0, 1.0),
        'disgust': (0.0, 1.0),
        'fear': (0.0, 1.0),
        'happy': (0.0, 1.0),
        'sad': (0.0, 1.0),
        'surprise': (0.0, 1.0),
        'neutral': (0.0, 1.0)
    }"""

    NUM_EMOTIONS = 7
    NUM_ITERATION = 1000000
    cur_threshold = np.empty(NUM_EMOTIONS)
    thresholds = np.empty((0, NUM_EMOTIONS))
    """cur_threshold = {
        'anger': 0.0,
        'disgust': 0.0,
        'fear': 0.0,
        'happy': 0.0,
        'sad': 0.0,
        'surprise': 0.0,
        'neutral': 0.0
    }"""

    for i in range(NUM_ITERATION):
        #sum = 0
        cur_threshold[0] = round(random.uniform(0.0, 1.0),1) # anger
        cur_threshold[1] = 0.0 # disgust
        cur_threshold[2] = round(random.uniform(0.0, cur_threshold[0]),1) # fear (skip disgust)
        cur_threshold[3] = round(random.uniform(0.0, cur_threshold[2]),1) # happy
        cur_threshold[4] = round(random.uniform(0.0, cur_threshold[3]),1) # sad
        cur_threshold[5] = round(random.uniform(0.0, cur_threshold[4]),1) # surprise
        cur_threshold[6] = round(random.uniform(0.0, cur_threshold[5]),1) # neutral
        #cur_threshold[6] = 1 - (cur_threshold[0]+cur_threshold[2]+cur_threshold[3]+cur_threshold[4]+cur_threshold[5])

        """cur_threshold[0] = round(random.uniform(0.0, 1.0), 2)  # anger
        cur_threshold[1] = 0.0  # disgust
        cur_threshold[2] = round(random.uniform(0.0, 1.0), 2)  # fear (skip disgust)
        cur_threshold[3] = round(random.uniform(0.0, 1.0), 2)  # happy
        cur_threshold[4] = round(random.uniform(0.0, 1.0), 2)  # sad
        cur_threshold[5] = round(random.uniform(0.0, 1.0), 2)  # surprise"""


        thresholds = np.append(thresholds, [cur_threshold], axis=0)
        print(f'chosen threshold {i} = {cur_threshold}')

    return thresholds

        #for emotion in ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprise']:  # no 'neutral'
         #   cur_threshold[emotion] = round(random.uniform(0.0, 1.0), 2)




    #return {emotion: np.random.uniform(low, high) for emotion, (low, high) in search_space.items()}

def is_pain_by_threshold(threshold, prediction):
    if (prediction > threshold).all():
        return 1
    return 0

def calculate_accuracy(threshold, preds, labels):
    count_correct = 0
    for i in range(len(preds)):
        if is_pain_by_threshold(threshold, preds[i]) == labels[i]:
            count_correct += 1
    return count_correct / len(preds)


def choose_best_threshold(thresholds, preds, labels):
    best_threshold = thresholds[0]
    best_acc = 0
    cur_acc = 0
    for threshold in thresholds:
        cur_acc = calculate_accuracy(threshold, preds, labels)
        print(f'threshold {threshold} with acc = {cur_acc}')
        if(cur_acc > best_acc):
            best_threshold = threshold
            best_acc = cur_acc
            print(f"current best acc = {best_acc} with best_threshold = {best_threshold}")

    return best_threshold, best_acc


preds, labels = prepare_data()
"""thresholds = choose_thresholds()
best_threshold, best_acc = choose_best_threshold(thresholds, preds, labels)"""



"""print(f'best_threshold = {best_threshold}')
print(f'best_acc = {best_acc}')"""







"""
import numpy as np
from sklearn.metrics import f1_score

# Example predictions for a validation set (list of arrays)
validation_predictions = [
    np.array([0.1, 0.05, 0.3, 0.1, 0.4, 0.05, 0.0]),
    np.array([0.2, 0.1, 0.4, 0.2, 0.3, 0.1, 0.1]),
    # Add more validation samples
]

# Corresponding labels for the validation set (1 for pain, 0 for no pain)
validation_labels = [1, 0,  # Add more labels
                     ]

# Define the search space for each threshold
search_space = {
    'anger': (0.0, 1.0),
    'disgust': (0.0, 0.15),
    'fear': (0.0, 1.0),
    'happy': (0.0, 1.0),
    'sad': (0.0, 1.0),
    'surprise': (0.0, 1.0),
    'neutral': (0.0, 1.0)
}


# Function to generate random thresholds
def generate_random_thresholds(search_space):
    return {emotion: np.random.uniform(low, high) for emotion, (low, high) in search_space.items()}


# Function to detect pain based on thresholds
def detect_pain(predictions, thresholds):
    pain_detected = False
    for i, emotion in enumerate(thresholds.keys()):
        if predictions[i] > thresholds[emotion]:
            pain_detected = True
            break
    return pain_detected


# Random search
best_f1_score = 0
best_thresholds = None
num_iterations = 100  # Number of random samples

for _ in range(num_iterations):
    thresholds = generate_random_thresholds(search_space)
    predictions = [detect_pain(pred, thresholds) for pred in validation_predictions]
    score = f1_score(validation_labels, predictions)

    if score > best_f1_score:
        best_f1_score = score
        best_thresholds = thresholds

print("Best F1 Score:", best_f1_score)
print("Best Thresholds:", best_thresholds)
"""