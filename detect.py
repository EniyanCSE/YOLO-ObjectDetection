# Code to check with video

import cv2
import supervision as sv
from ultralytics import YOLO

video = cv2.VideoCapture("video.mp4")

model = YOLO("yolov8s.pt")
bbox_annotator = sv.BoxAnnotator()



while video.isOpened():
    ret, frame = video.read()
    if ret == True:
        result = model(frame)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = detections[detections.confidence > 0.5]
        labels = [
            result.names[class_id]
            for class_id
            in detections.class_id
        ]
        frame = bbox_annotator.annotate(scene=frame,
                                        detections=detections,
                                        labels=labels)
        
        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break 
    else:
        break

video.release()
cv2.destroyAllWindows()


# Code to check with image


# import cv2
# import supervision as sv
# from ultralytics import YOLO

# # Load the image
# image = cv2.imread("2.jpg")

# # Initialize YOLO model
# model = YOLO("yolov8s.pt")

# # Initialize bbox_annotator
# bbox_annotator = sv.BoxAnnotator()

# # Perform inference on the image
# result = model(image)[0]

# # Process detections
# detections = sv.Detections.from_ultralytics(result)
# detections = detections[detections.confidence > 0.5]
# labels = [result.names[class_id] for class_id in detections.class_id]

# # Annotate the image
# annotated_image = bbox_annotator.annotate(scene=image, detections=detections, labels=labels)

# # Display the annotated image
# cv2.imshow("Annotated Image", annotated_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
