import os
import torch
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, 'yolov5', 'runs', 'train', 'Taras', 'weights', 'best.pt')

face_model = torch.hub.load('yolov5', 'custom', path=model_path, source='local')
coco_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
face_model.to(device)
coco_model.to(device)

cap = cv2.VideoCapture(0)
print("Go")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results_face = face_model(frame)
    results_coco = coco_model(frame)

    df_face = results_face.pandas().xyxy[0]
    df_coco = results_coco.pandas().xyxy[0]

    persons = df_coco[df_coco['name'] == 'person']

    for _, row_f in df_face.iterrows():
        fx1, fy1, fx2, fy2 = map(int, [row_f['xmin'], row_f['ymin'], row_f['xmax'], row_f['ymax']])
        face_box_drawn = False

        for _, row_p in persons.iterrows():
            px1, py1, px2, py2 = map(int, [row_p['xmin'], row_p['ymin'], row_p['xmax'], row_p['ymax']])
            if fx1 > px1 and fy1 > py1 and fx2 < px2 and fy2 < py2:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
                cv2.putText(frame, "person+Taras", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                face_box_drawn = True
                break

        if not face_box_drawn:
            cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), (255, 0, 255), 2)
            cv2.putText(frame, "Taras", (fx1, fy1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

    for _, row_p in persons.iterrows():
        px1, py1, px2, py2 = map(int, [row_p['xmin'], row_p['ymin'], row_p['xmax'], row_p['ymax']])
        person_contains_face = any(
            (int(row_f['xmin']) > px1 and int(row_f['ymin']) > py1 and
             int(row_f['xmax']) < px2 and int(row_f['ymax']) < py2)
            for _, row_f in df_face.iterrows()
        )
        if not person_contains_face:
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
            cv2.putText(frame, "person", (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.imshow("YOLOv5 Combined", frame)

    key = cv2.waitKey(10)
    if key == ord('q') or cv2.getWindowProperty("YOLOv5 Combined", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
