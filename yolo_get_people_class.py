
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision import Detections, BoxAnnotator,LabelAnnotator

#CONF=0.6
#CLASSES=[0]
#FRAME_WIDTH=640
#FRAME_HEIGHT=480
YOLO_PATH = "yolov8n.pt" #Maybe should be a parameter of the class or a constant of the class
#EXTRA_MARGIN = 30 #extra space of pixels we take for the boxes


class PersonDetection:
    # Maybe YOLO_PATH should be a parameter of the class or a constant of the class
    CLASSES = [0]
    def __init__(self,frame_width,frame_height,extra_margin,conf):#__init__(self, capture_index,conf,classes,extra_margin):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.conf=conf
        #self.CLASSES=classes #Should be const instead
        self.extra_margin = extra_margin #extra space of pixels we take for the boxes

        #self.capture_index = capture_index #Most likely don't need it if we won't get the picture here which is what should happen


        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        #self.box_annotator = BoxAnnotator()#(ColorPalette, thickness=3) #Line was earlier BoxAnnotator(color = ColorPalette, thickness=3) maybe return to it
        self.bounding_box_annotator = BoxAnnotator()
        self.label_annotator = LabelAnnotator()

    def load_model(self):

        model = YOLO(YOLO_PATH)  # load a pretrained YOLOv8n model
        model.fuse()

        return model

    def predict(self, frame):

        results = self.model(frame, conf=self.conf, classes=self.CLASSES)#elf.model(frame)

        return results

    def plot_bboxes(self, results, frame): #Maybe change name

        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for person class
        for result in results:
            #print(result)
            boxes = result.boxes.cpu().numpy()
            #print("Boxes type = ",type(boxes),"boxes = ", boxes,"size = ",np.size(boxes.cls))
            if np.size(boxes.cls)> 0:
                """class_id = boxes.cls[0]
                conf = boxes.conf[0]
                xyxy = boxes.xyxy[0]
                print(class_id,type(class_id),class_id == 0)
                if class_id == 0:
                    xyxys.append(result.boxes.xyxy.cpu().numpy())
                    confidences.append(result.boxes.conf.cpu().numpy())
                    class_ids.append(result.boxes.cls.cpu().numpy().astype(int))"""
                """xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))"""
                xyxys = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)


        # Setup detections for visualization
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        # Format custom labels
        #print(len(detections),type(detections),detections[0]== detections,detections.xyxy)
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for class_id,confidence in zip(detections.class_id,detections.confidence)]
        #print(self.labels)
        print(type(xyxys),xyxys)

        # Annotate and display frame
        annotated_frame = frame.copy()

        annotated_frame = self.bounding_box_annotator.annotate(
            scene=annotated_frame,
            detections=detections
        )
        annotated_frame = self.label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=self.labels
        )
        #Obviously need to change to work with multiple people. Also probably return a list of pictures
        for box in xyxys:#may collapse if it is empty

            cv2.circle(annotated_frame,(max(int(box[0]) - self.extra_margin,0),max(int(box[1])- self.extra_margin,0)),20,color=(255, 0, 0) )
            cv2.circle(annotated_frame,(min(int(box[2])+self.extra_margin,self.frame_width),min(int(box[3])+self.extra_margin,self.frame_height)),20,color=(255, 0, 0))
        #cv2.circle(annotated_frame, (int(xyxys[0][0]), int(xyxys[0][1])), 20, color=(255, 0, 0))
        #cv2.circle(annotated_frame, (int(xyxys[0][2]), int(xyxys[0][3])), 20, color=(255, 0, 0))

        #print(type(frame),type(annotated_frame))
        #frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels) #Maybe do something like this https://github.com/roboflow/supervision/discussions/374
        #return xyxys[0]. may collapse if it is empty

        return annotated_frame,xyxys

    def crop_image(self,frame,x1,y1,x2,y2):
        return frame[max(int(y1) - self.extra_margin, 0):min(int(y2) + self.extra_margin, self.frame_height), max(int(x1) - self.extra_margin, 0):min(int(x2) + self.extra_margin, self.frame_width)]
        #return frame[max(int(y1) - self.extra_margin, 0):max(int(y2) + self.extra_margin, 0), max(int(x1) - self.extra_margin, 0):max(int(x2) + self.extra_margin, 0)]

    def get_people_from_boxes(self,frame,boxes):
        people = []
        for box in boxes:
            people.append(self.crop_image(frame, int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        return people

    def get_people(self,frame):
        #Return updated frame. Maybe shouldn't write fps
        start_time = time()
        results = self.predict(frame)
        frame, boxes = self.plot_bboxes(results, frame)
        # print(type(frame))
        # print(type(boxes[0][0]),boxes[0][0])
        print(boxes)

        end_time = time()
        fps = 1 / np.round(end_time - start_time, 2)

        #cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        # frame = self.crop_image(frame,int(boxes[0][0]),int(boxes[0][1]),int(boxes[0][2]),int(boxes[0][3])) #Code needs while loop to work for multiple people.
        return frame,self.get_people_from_boxes(frame,boxes)





#detector = PersonDetection(conf=0.6,classes=[0],extra_margin=30)#PersonDetection(capture_index=0,conf=0.6,classes=[0],extra_margin=30)
#detector()

#Code crashes when there isn't a person to detect
#Code crashes when there isn't a person to detect+ slow + not entirely accurate may need to change it and filter people by how confident it is. Also some parts of the code are depreceated
#Maybe add somehow something like confidence like this https://github.com/DAVIDNYARKO123/yolov8-silva/blob/main/yolov8_n_opencv.py
#https://iopscience.iop.org/article/10.1088/1742-6596/2466/1/012034/pdf
#https://github.com/ultralytics/ultralytics/issues/4601
#Maybe add multithreading


"""cap = cv2.VideoCapture(0)
assert cap.isOpened()
#cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(3, 640)
cap.set(4, 480)
person_detection = PersonDetection(0.6,30)
while True:
    start_time = time()
    frames = []
    ret, frame = cap.read()
    #print(type(frame))
    assert ret #Maybe get this out of the loop

    frame,people = person_detection.get_people(frame)

    cv2.imshow('YOLOv8 Detection', frame)
    #cv2.imshow('YOLOv8 Detection v2', frame)


    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()"""