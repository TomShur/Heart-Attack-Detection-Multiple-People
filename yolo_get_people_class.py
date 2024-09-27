
import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision import Detections, BoxAnnotator,LabelAnnotator


YOLO_PATH = "yolov8n.pt"


class PersonDetection:
    CLASSES = [0] #This constant tells the YOLO model to only identify people
    def __init__(self,frame_width,frame_height,extra_margin,conf):
        #Initialize instance where the fram width,frame height are known
        #Extra margin refers to the amount of extra pixels
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.conf=conf
        self.extra_margin = extra_margin #extra space of pixels we take for the boxes of people to adjust for the errors the yolo program makes when creating them



        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model() #The yolo model we use

        self.CLASS_NAMES_DICT = self.model.model.names #List of all the possibly classification the model has

        #self.box_annotator = BoxAnnotator()#(ColorPalette, thickness=3) #Line was earlier BoxAnnotator(color = ColorPalette, thickness=3) maybe return to it
        self.bounding_box_annotator = BoxAnnotator()#Instance to help draw boxes around the people
        self.label_annotator = LabelAnnotator()#Instance to help write the labels around the people

    def load_model(self):

        model = YOLO(YOLO_PATH)  # load a pretrained YOLOv8n model
        model.fuse()

        return model

    def predict(self, frame):
        #Get a frame and return the prediction of the YOLO model

        results = self.model(frame, conf=self.conf, classes=self.CLASSES)

        return results

    def plot_bboxes(self, results, frame):
        #Get frame and the results of the predict method
        #Return the updated frame (with the boxes, classification displayed)
        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for person class
        for result in results:
            boxes = result.boxes.cpu().numpy()
            if np.size(boxes.cls)> 0:

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
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for class_id,confidence in zip(detections.class_id,detections.confidence)]
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
        #For each person draw 2 circles - one for each side of the rectangle with the extra margin
        for box in xyxys:

            cv2.circle(annotated_frame,(max(int(box[0]) - self.extra_margin,0),max(int(box[1])- self.extra_margin,0)),20,color=(255, 0, 0) )
            cv2.circle(annotated_frame,(min(int(box[2])+self.extra_margin,self.frame_width),min(int(box[3])+self.extra_margin,self.frame_height)),20,color=(255, 0, 0))


        return annotated_frame,xyxys

    def crop_image(self,frame,x1,y1,x2,y2):
        #Given an image return it cropped between the 2 coordinates. Important to know that frames are numpy array so the leftmost part of the image is the (0,0)
        return frame[max(int(y1) - self.extra_margin, 0):min(int(y2) + self.extra_margin, self.frame_height), max(int(x1) - self.extra_margin, 0):min(int(x2) + self.extra_margin, self.frame_width)]

    def get_people_from_boxes(self,frame,boxes):
        #Get the frame and the coordinates of the rectangles of the people and extract each person and return the list
        people = []
        for box in boxes:
            people.append(self.crop_image(frame, int(box[0]), int(box[1]), int(box[2]), int(box[3])))
        return people

    def get_people(self,frame):
        #Get a frame, and return the updated frame and the people extracted
        start_time = time()
        results = self.predict(frame)
        frame, boxes = self.plot_bboxes(results, frame)

        print(boxes)

        end_time = time()
        fps = 1 / np.round(end_time - start_time, 2)

        return frame,self.get_people_from_boxes(frame,boxes)





#detector = PersonDetection(conf=0.6,classes=[0],extra_margin=30)#PersonDetection(capture_index=0,conf=0.6,classes=[0],extra_margin=30)


