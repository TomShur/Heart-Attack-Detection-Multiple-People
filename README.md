# 💓Visual Heart Attack Detection Project
This project is an application of Deep Learning and Computer Vision techniques to **detect a potential heart attack in real-time video streams**.


## Methodology Overview
### For Single Person
The model outputs a real value between 0 to 1, representing the probability that the person in the frame is having a heart attack.
The final prediction is a weighted average between two computed values, each relies on a different visual characteristic:
#### 1) Facial Expression - 25%
We use the **FER** librery to detect emotions that should be noticeable during a heart attack (e.g. surprise, anger, fear), and emotions that indicate the opposite (e.g. neutrality). <br>
A probability is computed for each such emotion, and with predetermined tuned hyper parameters, we threshold each probability to compute an emotion-based prediction.
#### 2) Body Posture - 75%
We trained a CNN model that outputs a prediction based on the body position of the person in the image. <br>
Instead of the whole image, the model gets the **skeleton** of the person in the image as input. <br>
The "skeleton" of the person is a set of specific 2d points identified using a pose detection model from the **Mediapipe** library. <br>

<img width="482" height="856" alt="image" src="https://github.com/user-attachments/assets/1d920989-f384-4c5f-b024-9a7e439a5b3a" />

This approach helps reduce dimensionality ,which is crucial both to deal with the curse of dimensionality and to speedup the computation. <br>
It also may reduce potential bias ,for example the model will depend less on the background of the image ,the skin color of the person etc. <br>
### Generalizing for Multi-Person
We use a **YOLO** model to detect each person in the frame. More precisely, a bounding box is computed for each person and cropped from the whole image. <br>
Then we activate the single-person model on each cropped image.

## Results
The final model achieved a train accuracy of 90%, and test accuracy of 84%. <br>
The model works in an average of time 5 fps for single person, and 2 fps for multi-person (this is when running on i7 11th gen 2.8gh CPU).

demo for single-person : https://youtu.be/YbF29tdP6BQ <br>
demo for multi-person : https://youtu.be/X9KXEo-PqRI
