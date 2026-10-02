import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"E:\Study\SEM-3\Robotics\Analog_wristwatch_detection\models\best.onnx"

INPUT_SIZE = 320
CONF_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.45


# ============================================================
# LOAD ONNX MODEL
# ============================================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("ONNX model loaded.")
print("Input:", session.get_inputs()[0].shape)
print("Outputs:")

for output in session.get_outputs():
    print(" ", output.name, output.shape)


# ============================================================
# SIGMOID
# ============================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ============================================================
# DFL DECODING
# ============================================================

def decode_dfl(box_output):
    """
    Convert 64 DFL values into 4 distances:
    left, top, right, bottom

    64 = 4 coordinates × 16 bins
    """

    # Shape:
    # (1, 64, H, W)

    _, _, h, w = box_output.shape

    x = box_output.reshape(1, 4, 16, h, w)

    # Softmax over the 16 bins
    x = np.exp(x - np.max(x, axis=2, keepdims=True))
    x = x / np.sum(x, axis=2, keepdims=True)

    bins = np.arange(16, dtype=np.float32)

    distances = np.sum(x * bins.reshape(1, 1, 16, 1, 1), axis=2)

    # Shape:
    # (1, 4, H, W)

    return distances[0].transpose(1, 2, 0).reshape(-1, 4)


# ============================================================
# ANCHORS
# ============================================================

def make_anchors(h, w, stride):
    """
    YOLO anchor points:
    x = column + 0.5
    y = row + 0.5
    """

    x = np.arange(w, dtype=np.float32) + 0.5
    y = np.arange(h, dtype=np.float32) + 0.5

    yy, xx = np.meshgrid(y, x)

    anchors = np.stack(
        [xx.reshape(-1), yy.reshape(-1)],
        axis=1
    )

    return anchors


# ============================================================
# DECODE ONE SCALE
# ============================================================

def decode_scale(box_output, score_output, stride):

    _, _, h, w = box_output.shape

    # Decode DFL
    distances = decode_dfl(box_output)

    # Anchors
    anchors = make_anchors(h, w, stride)

    # Convert distances to x1,y1,x2,y2
    x1 = (anchors[:, 0] - distances[:, 0]) * stride
    y1 = (anchors[:, 1] - distances[:, 1]) * stride

    x2 = (anchors[:, 0] + distances[:, 2]) * stride
    y2 = (anchors[:, 1] + distances[:, 3]) * stride

    boxes = np.stack(
        [x1, y1, x2, y2],
        axis=1
    )

    # Class score
    scores = sigmoid(
        score_output.reshape(-1)
    )

    return boxes, scores


# ============================================================
# IOU
# ============================================================

def calculate_iou(box, boxes):

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])

    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection_w = np.maximum(0, x2 - x1)
    intersection_h = np.maximum(0, y2 - y1)

    intersection = intersection_w * intersection_h

    area_box = (
        (box[2] - box[0]) *
        (box[3] - box[1])
    )

    area_boxes = (
        (boxes[:, 2] - boxes[:, 0]) *
        (boxes[:, 3] - boxes[:, 1])
    )

    union = area_box + area_boxes - intersection

    return intersection / np.maximum(union, 1e-6)


# ============================================================
# NMS
# ============================================================

def nms(boxes, scores, iou_threshold):

    order = np.argsort(scores)[::-1]

    keep = []

    while len(order) > 0:

        i = order[0]

        keep.append(i)

        if len(order) == 1:
            break

        ious = calculate_iou(
            boxes[i],
            boxes[order[1:]]
        )

        remaining = np.where(
            ious <= iou_threshold
        )[0]

        order = order[remaining + 1]

    return keep


# ============================================================
# DETECTION
# ============================================================

def detect(frame):

    original_h, original_w = frame.shape[:2]

    # Resize exactly like our previous decoder test
    resized = cv2.resize(
        frame,
        (INPUT_SIZE, INPUT_SIZE)
    )

    # BGR -> RGB
    image = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    )

    # uint8 -> float32
    image = image.astype(np.float32) / 255.0

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # ONNX inference
    outputs = session.run(
        None,
        {input_name: image}
    )

    # --------------------------------------------------------
    # Outputs:
    #
    # 0 = box0  -> 40x40, stride 8
    # 1 = score0
    # 2 = box1  -> 20x20, stride 16
    # 3 = score1
    # 4 = box2  -> 10x10, stride 32
    # 5 = score2
    # --------------------------------------------------------

    all_boxes = []
    all_scores = []

    scales = [
        (outputs[0], outputs[1], 8),
        (outputs[2], outputs[3], 16),
        (outputs[4], outputs[5], 32)
    ]

    for box_output, score_output, stride in scales:

        boxes, scores = decode_scale(
            box_output,
            score_output,
            stride
        )

        # Confidence filtering
        mask = scores >= CONF_THRESHOLD

        if np.any(mask):

            all_boxes.append(
                boxes[mask]
            )

            all_scores.append(
                scores[mask]
            )

    # No detections
    if len(all_boxes) == 0:
        return []

    boxes = np.concatenate(
        all_boxes,
        axis=0
    )

    scores = np.concatenate(
        all_scores,
        axis=0
    )

    # NMS
    keep = nms(
        boxes,
        scores,
        NMS_IOU_THRESHOLD
    )

    detections = []

    # Convert 320x320 coordinates
    # back to camera frame coordinates
    scale_x = original_w / INPUT_SIZE
    scale_y = original_h / INPUT_SIZE

    for i in keep:

        x1, y1, x2, y2 = boxes[i]

        x1 *= scale_x
        x2 *= scale_x

        y1 *= scale_y
        y2 *= scale_y

        # Clip to camera frame
        x1 = max(0, min(original_w - 1, x1))
        y1 = max(0, min(original_h - 1, y1))

        x2 = max(0, min(original_w - 1, x2))
        y2 = max(0, min(original_h - 1, y2))

        detections.append(
            (
                int(x1),
                int(y1),
                int(x2),
                int(y2),
                float(scores[i])
            )
        )

    return detections


# ============================================================
# LAPTOP CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open laptop camera.")

    exit()


print()
print("Laptop camera started.")
print("Show an analog wristwatch.")
print("Press Q to quit.")
print()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("ERROR: Could not read camera frame.")

        break

    detections = detect(frame)

    for x1, y1, x2, y2, confidence in detections:

        # Bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            2
        )

        # Label
        label = f"analog_watch {confidence:.2f}"

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

    cv2.imshow(
        "ESP-DL ONNX - Analog Wristwatch Detection",
        frame
    )

    # Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Camera test finished.")