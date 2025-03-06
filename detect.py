import cv2
import numpy as np
from Const import (
    padding, faceProto, faceModel, ageProto, ageModel, 
    genderProto, genderModel, MODEL_MEAN_VALUES, ageList, genderList
)

def highlight_face(net, frame, conf_threshold=0.7):
    """Detect faces in the frame and return the processed frame and face boxes."""
    frame_dnn = frame.copy()
    height, width = frame.shape[:2]
    
    blob = cv2.dnn.blobFromImage(frame_dnn, 1.0, (300, 300), [104, 117, 123], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    face_boxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1, y1, x2, y2 = (detections[0, 0, i, 3:7] * np.array([width, height, width, height])).astype(int)
            face_boxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_dnn, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return frame_dnn, face_boxes


face_net = cv2.dnn.readNet(faceModel, faceProto)
age_net = cv2.dnn.readNet(ageModel, ageProto)
gender_net = cv2.dnn.readNet(genderModel, genderProto)


video = cv2.VideoCapture(0)

while cv2.waitKey(1) < 0:
    has_frame, frame = video.read()
    if not has_frame:
        break
    
    result_img, face_boxes = highlight_face(face_net, frame)
    
    if not face_boxes:
        print("No face detected")
        continue
    
    for x1, y1, x2, y2 in face_boxes:
        face = frame[max(0, y1 - padding): min(y2 + padding, frame.shape[0] - 1),
                     max(0, x1 - padding): min(x2 + padding, frame.shape[1] - 1)]
        
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        
       
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = genderList[gender_preds[0].argmax()]
        
       
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = ageList[age_preds[0].argmax()]
        
        print(f'Gender: {gender}, Age: {age[1:-1]} years')
        
        cv2.putText(result_img, f'{gender}, {age}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    cv2.imshow("Age and Gender Detection", result_img)

video.release()
cv2.destroyAllWindows()