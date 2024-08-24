

import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO

from supervision.draw.color import ColorPalette
from supervision import Detections, BoundingBoxAnnotator,LabelAnnotator


class ObjectDetection:

    def __init__(self, capture_index):

        self.capture_index = capture_index

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)

        self.model = self.load_model()

        self.CLASS_NAMES_DICT = self.model.model.names

        #self.box_annotator = BoxAnnotator()#(ColorPalette, thickness=3) #Line was earlier BoxAnnotator(color = ColorPalette, thickness=3) maybe return to it
        self.bounding_box_annotator =BoundingBoxAnnotator()
        self.label_annotator = LabelAnnotator()

    def load_model(self):

        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
        model.fuse()

        return model

    def predict(self, frame):

        results = self.model(frame)

        return results

    def plot_bboxes(self, results, frame):

        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for person class
        for result in results:
            boxes = result.boxes.cpu().numpy()
            class_id = boxes.cls[0]
            conf = boxes.conf[0]
            xyxy = boxes.xyxy[0]

            if class_id == 0.0:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(result.boxes.conf.cpu().numpy())
                class_ids.append(result.boxes.cls.cpu().numpy().astype(int))

        # Setup detections for visualization
        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        # Format custom labels
        #print(len(detections),type(detections),detections[0]== detections,detections.xyxy)
        self.labels = [f"{self.CLASS_NAMES_DICT[class_id]} {confidence:0.2f}" for class_id,confidence in zip(detections.class_id,detections.confidence)]
        print(self.labels)

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
        #frame = self.box_annotator.annotate(scene=frame, detections=detections, labels=self.labels) #Maybe do something like this https://github.com/roboflow/supervision/discussions/374

        return annotated_frame

    def __call__(self):

        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        #cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        #cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(3, 640)
        cap.set(4, 480)

        while True:

            start_time = time()

            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()
        cv2.destroyAllWindows()


detector = ObjectDetection(capture_index=0)
detector()

#Code crashes when there isn't a person to detect
#Code crashes when there isn't a person to detect+ slow + not entirely accurate may need to change it and filter people by how confident it is. Also some parts of the code are depreceated