import cv2

cap = cv2.VideoCapture(0)
i = 0
while i < 50:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Capture", frame)
    key = cv2.waitKey(1)
    if key == ord('c'):
        cv2.imwrite(f'myface_dataset/images/train/face_{i}.jpg', frame)
        print(f"Captured face_{i}.jpg")
        i += 1
    elif key == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()