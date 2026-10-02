import cv2
from ultralytics import YOLO

# Load our trained analog wristwatch model
model = YOLO(r"E:\Study\SEM-3\Robotics\Analog_wristwatch_detection\models\best.pt")

# Open laptop camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open laptop camera.")
    exit()

print("Laptop camera started.")
print("Show an analog wristwatch to the camera.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Failed to read camera frame.")
        break

    # Run detection
    results = model.predict(
        source=frame,
        imgsz=320,
        conf=0.5,
        verbose=False
    )

    # Draw detections
    annotated_frame = results[0].plot()

    # Display
    cv2.imshow("Analog Wristwatch Detection", annotated_frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()