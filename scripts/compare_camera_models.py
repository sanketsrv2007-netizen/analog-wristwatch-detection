import cv2
from ultralytics import YOLO

# -----------------------------
# Load both models
# -----------------------------
pt_model = YOLO(
    r"E:\Study\SEM-3\Robotics\Analog_wristwatch_detection\models\best.pt"
)

# -----------------------------
# Open laptop camera
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open laptop camera.")
    exit()

print("Camera opened.")
print()
print("Press SPACE to capture one frame.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read frame.")
        break

    cv2.imshow("Camera - Press SPACE", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 32:  # SPACE

        # Save the exact frame
        image_path = (
            r"E:\Study\SEM-3\Robotics\Analog_wristwatch_detection"
            r"\results\camera_comparison.jpg"
        )

        cv2.imwrite(image_path, frame)

        print()
        print("Captured:", image_path)

        # ---------------------------------
        # Run Ultralytics best.pt
        # ---------------------------------
        results = pt_model.predict(
            source=frame,
            imgsz=320,
            conf=0.5,
            verbose=False
        )

        annotated = results[0].plot()

        # Print detections
        boxes = results[0].boxes

        print()
        print("===== best.pt / Ultralytics =====")

        if boxes is None or len(boxes) == 0:
            print("No detections.")

        else:
            for i in range(len(boxes)):

                xyxy = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())

                print(
                    f"Detection {i + 1}: "
                    f"x1={xyxy[0]:.1f}, "
                    f"y1={xyxy[1]:.1f}, "
                    f"x2={xyxy[2]:.1f}, "
                    f"y2={xyxy[3]:.1f}, "
                    f"conf={conf:.3f}"
                )

        # Show result
        cv2.imshow(
            "best.pt - Ultralytics",
            annotated
        )

        print()
        print("Frame saved.")
        print("Close the result window or press Q.")

        while True:

            k = cv2.waitKey(1) & 0xFF

            if k == ord("q"):
                cap.release()
                cv2.destroyAllWindows()
                exit()

            # SPACE captures another frame
            if k == 32:
                break

cap.release()
cv2.destroyAllWindows()