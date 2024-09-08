"""import cv2
from keras.models import load_model
import numpy as np
import PoseModuleTom as pm

import numpy as np

# Load the saved model

#model = load_model('model/05_model.h5')
model = load_model('model/05_model.h5')


# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    detector = pm.PoseDetector()
    img = detector.findPose(frame)
    lmList = detector.getPosition(img)
    print(lmList)

    lmList = lmList[:,1:]
    #lmList2 = np.array([lmList])
    #print(f'shape={lmList2.shape}')
    #print(lmList2)


    lmList = lmList.reshape((9,2,1))
    lm2 = np.array(lmList)
    print(f'shape={lm2.shape}')
    print(lm2)
    # Make predictions
    prediction = model.predict(lm2)

    # Display the resulting frame
    cv2.putText(frame, f'Prediction: {prediction[0][0]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Live Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()"""










#This works but may be slow
"""import cv2
from keras.models import load_model
import numpy as np
import PoseModuleTom as pm
#Remember that the updated PoseModule is currently called PoseModuleTom
import numpy as np

# Load the saved model

#model = load_model('model/05_model.h5')
model = load_model('model/09_model_1D.h5')


# Initialize video capture
cap = cv2.VideoCapture(0)  # 0 for default webcam

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    detector = pm.PoseDetector() #Maybe should be outside the main loop
    img = detector.findPose(frame)
    lmList = detector.getPosition(img)

    lmList = lmList[:,1:]
    lmList2 = np.array([lmList])
    print(f'shape={lmList2.shape}')
    print(lmList2)


    #lmList = lmList.reshape((9,2,1))

    # Make predictions
    prediction = model.predict(lmList2)

    # Display the resulting frame
    cv2.putText(frame, f'Prediction: {prediction[0][0]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Live Video', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()"""


"""import cv2
import numpy as np
import PoseModuleTom as pm
from keras.models import load_model






model = load_model('model_1D_Z/09_model_1D_Z.h5')

cap = cv2.VideoCapture(0)
detector = pm.PoseDetector()

while True:

    ret, frame = cap.read()



    img = detector.findPose(frame)
    lmList = detector.getPosition(img)

    # notice if the lmList contain id's or not


    print(f'shape={lmList.shape}')
    print(lmList)

    lmList2 = np.array([lmList])
    print(f'shape2={lmList2.shape}')
    print(lmList2)

    if len(lmList2[0]) != 0:
        print("hey")
        prediction = model.predict(lmList2)
        cv2.putText(frame, f'Prediction: {prediction[0][0]:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #cv2.imshow('Live Video', frame)
    cv2.imshow('Live Video', img)
    cv2.waitKey(1)




cap.release()
cv2.destroyAllWindows()"""




from keras.models import load_model





#model = load_model('model_1D/09_model_1D.h5')

#model = load_model('model_1D_Z/09_model_1D_Z.h5')


class HearAttackModel:

    def __init__(self,model_path):
        self.model_path = model_path
        self.model = load_model(self.model_path)

    def predict(self,skeleton):
        #Maybe do threading and predict only once every few frames or seconds
        if len(skeleton[0]) != 0:
            prediction = self.model.predict(skeleton)
            #Check what prediction is without the [0][0]
            return prediction[0][0]


"""from PoseModuleTom import PoseDetector
detector = PoseDetector()
heart_attack_model = HearAttackModel(r"model\09_model_1D.h5")

import cv2
import numpy as np


#model = load_model('model_1D_Z/07_model_1D_Z.h5')


cap = cv2.VideoCapture(0)
while True:

    ret, frame = cap.read()



    img = detector.findPose(frame)
    lmList = detector.getPosition(img)

    lmList = lmList[:, 1:] # this is if we keep the id's
    print(f'shape={lmList.shape}')
    print(lmList)

    lmList2 = np.array([lmList])
    print(f'shape2={lmList2.shape}')
    print(lmList2)

    if len(lmList2[0]) != 0:
        print("hey")
        prediction = heart_attack_model.predict(lmList2)
        cv2.putText(frame, f'Prediction: {prediction:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    #cv2.imshow('Live Video', frame)
    cv2.imshow('Live Video', img)
    cv2.waitKey(1)




cap.release()
cv2.destroyAllWindows()"""