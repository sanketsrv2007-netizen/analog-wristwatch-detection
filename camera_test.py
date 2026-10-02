from ultralytics import YOLO
import cv2

model = YOLO(
    r"E:\Study\SEM-3\Robotics\Analog_wristwatch_detection\runs\detect\runs\watch_yolov8\weights\best.pt"
)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Camera error")
        break

    results = model(frame, imgsz=320, conf=0.01, verbose=False)

    best_conf = 0.0

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            print(f"{class_name}: {confidence:.3f}")

            if confidence > best_conf:
                best_conf = confidence

    print(f"Best confidence: {best_conf:.3f}")

    annotated_frame = results[0].plot()

    cv2.imshow("Analog Watch Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()