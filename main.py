import torch
import cv2

model_path = r'E:\Repository\Yolo-v5-example\yolov5\runs\train\Taras\weights\best.pt'

# Завантаження кастомної моделі
model = torch.hub.load(
    'yolov5',
    'custom',
    path=model_path,
    source='local' 
)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

cap = cv2.VideoCapture(0)

print("Go")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Не вдалося зчитати кадр.")
        break

    results = model(frame)

    annotated_frame = results.render()[0]

    cv2.imshow("YOLOv5s", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
