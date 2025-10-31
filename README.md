# Visual Heart Attack Detection Project
This project is an application of Deep Learning and Computer Vision techniques to **detect a potential heart attack in real-time video streams**.


## Methodology
The model outputs a real value between 0 to 1, representing the probability that the person in the frame is having a heart attack.
The final prediction is a weighted average between two computed values, each relies on a different visual characteristic:
### 1) Facial Expression - 25%
We use the **FER** librery to detect emotions that should be noticeable during a heart attack (e.g. surprise, anger, fear), and emotions that indicate the opposite (e.g. neutrality).
A probability is computed for each such emotion, and with predetermined hyper parameters, we threshold each probability to compute an emotion-based prediction.
### 2) Body Posture - 75%
We trained a CNN model that outputs a prediction based on the body position of the person in the image.
