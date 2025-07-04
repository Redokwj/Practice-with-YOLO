from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-pose.pt")  

cap = cv2.VideoCapture(0)  

Is_camera = True

image = "test_images/down.png"

def check_hand_status(shoulder, wrist):
    if wrist[1] < shoulder[1]:
        return "HAND UP"
    elif abs(wrist[1] - shoulder[1]) < 80:
        return "HAND SIDE"
    else:
        return "HAND DOWN"

def detect_keypoints(results,frame=None):
   for r in results:
        keypoints = r.keypoints.xy  
        if len(keypoints):
            person = keypoints[0]  

            if Is_camera:
                draw_skeleton(frame, person)

            left_shoulder = person[5]
            left_wrist = person[9]
            left_status = check_hand_status(left_shoulder, left_wrist)

            right_shoulder = person[6]
            right_wrist = person[10]
            right_status = check_hand_status(right_shoulder, right_wrist)
            
            return left_status, right_status
        
        else:
            return "No hand detected", "No hand detected"
        
def draw_skeleton(frame, keypoints):
    skeleton = [
        (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 12), (5, 11), (6, 12),
        (11, 13), (13, 15), (12, 14), (14, 16)
    ]

    for i, point in enumerate(keypoints):
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), 4, (0, 255, 255), -1) 

    for connection in skeleton:
        pt1 = keypoints[connection[0]]
        pt2 = keypoints[connection[1]]
        if pt1 is not None and pt2 is not None:
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  


if Is_camera:
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, conf=0.3, save=False, show=False, stream=True)
        left_status, right_status = detect_keypoints(results, frame=frame)    

        cv2.putText(frame, f"Left Hand: {left_status}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
        cv2.putText(frame, f"Right Hand: {right_status}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Pose Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

else:       
    results = model.predict(source=image, conf=0.3, save=False, show=False)
    left_status, right_status = detect_keypoints(results)    

    print(f"Left Hand: {left_status}")
    print(f"Right Hand: {right_status}")



