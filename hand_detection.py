from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-pose.pt")  

cap = cv2.VideoCapture(0)  

def check_hand_direction(shoulder, wrist):
    if wrist[1] < shoulder[1]:
        return "HAND UP"
    elif abs(wrist[1] - shoulder[1]) < 80:
        return "HAND SIDE"
    else:
        return "HAND DOWN"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, conf=0.3, save=False, show=False, stream=True)
    for r in results:
        keypoints = r.keypoints.xy  
        if len(keypoints):
            person = keypoints[0]  

            left_shoulder = person[5]
            left_wrist = person[9]
            left_status = check_hand_direction(left_shoulder, left_wrist)

            right_shoulder = person[6]
            right_wrist = person[10]
            right_status = check_hand_direction(right_shoulder, right_wrist)

            
            cv2.putText(frame, f"Left Hand: {left_status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
            cv2.putText(frame, f"Right Hand: {right_status}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("Pose Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


