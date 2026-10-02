import cv2
import os
import time

SAVE_DIR = "webcam_watch_images"
os.makedirs(SAVE_DIR, exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Could not open laptop camera.")
    input("Press Enter to exit...")
    raise SystemExit

count = 0
last_save = 0

print("Camera started.")
print("Press SPACE to capture an image.")
print("Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("ERROR: Could not read camera frame.")
        break

    cv2.putText(
        frame,
        "SPACE = capture | Q = quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    cv2.imshow("Watch Dataset Capture", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord(' '):
        filename = os.path.join(
            SAVE_DIR,
            f"watch_{count:04d}.jpg"
        )

        cv2.imwrite(filename, frame)

        print("Saved:", filename)

        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print()
print(f"Captured {count} images.")
print("Images saved in:", SAVE_DIR)

input("Press Enter to exit...")