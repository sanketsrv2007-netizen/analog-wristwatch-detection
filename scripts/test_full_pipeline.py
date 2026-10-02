import os
import sys
import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# SETTINGS
# ============================================================

MODEL_PATH = r"models\best.onnx"
OUTPUT_DIR = r"results\full_pipeline"

INPUT_SIZE = 320
CONFIDENCE_THRESHOLD = 0.50
NMS_IOU_THRESHOLD = 0.45

STRIDES = [8, 16, 32]


# ============================================================
# RGB565 -> RGB888
# ============================================================

def rgb565_to_rgb888(rgb565):
    r = ((rgb565 >> 11) & 0x1F) << 3
    g = ((rgb565 >> 5) & 0x3F) << 2
    b = (rgb565 & 0x1F) << 3

    return np.stack([r, g, b], axis=-1).astype(np.uint8)


# ============================================================
# DFL DECODER
# ============================================================

def decode_dfl(box_output):
    """
    Convert 64 DFL values into 4 distances:
    left, top, right, bottom
    """

    # Shape:
    # (1, 64, N) -> (N, 4, 16)

    x = box_output[0]
    x = x.reshape(4, 16, -1)
    x = np.transpose(x, (2, 0, 1))

    # Stable softmax
    x = x - np.max(x, axis=2, keepdims=True)

    exp_x = np.exp(x)
    probabilities = exp_x / np.sum(
        exp_x,
        axis=2,
        keepdims=True
    )

    bins = np.arange(16, dtype=np.float32)

    distances = np.sum(
        probabilities * bins,
        axis=2
    )

    return distances


# ============================================================
# ANCHORS
# ============================================================

def make_anchors(height, width, stride):

    y, x = np.meshgrid(
        np.arange(height, dtype=np.float32),
        np.arange(width, dtype=np.float32),
        indexing="ij"
    )

    anchors = np.stack(
        [
            x.reshape(-1) + 0.5,
            y.reshape(-1) + 0.5
        ],
        axis=1
    )

    return anchors


# ============================================================
# DECODE ONE SCALE
# ============================================================

def decode_scale(box_output, score_output, stride):

    _, _, height, width = box_output.shape

    distances = decode_dfl(box_output)

    anchors = make_anchors(
        height,
        width,
        stride
    )

    scores = score_output[0, 0].reshape(-1)

    # Sigmoid
    scores = 1.0 / (1.0 + np.exp(-scores))

    # Convert distances to pixel coordinates
    x1 = (anchors[:, 0] - distances[:, 0]) * stride
    y1 = (anchors[:, 1] - distances[:, 1]) * stride

    x2 = (anchors[:, 0] + distances[:, 2]) * stride
    y2 = (anchors[:, 1] + distances[:, 3]) * stride

    boxes = np.stack(
        [x1, y1, x2, y2],
        axis=1
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

    intersection = np.maximum(
        0,
        x2 - x1
    ) * np.maximum(
        0,
        y2 - y1
    )

    area_box = (
        (box[2] - box[0]) *
        (box[3] - box[1])
    )

    area_boxes = (
        (boxes[:, 2] - boxes[:, 0]) *
        (boxes[:, 3] - boxes[:, 1])
    )

    union = area_box + area_boxes - intersection

    return intersection / (union + 1e-6)


# ============================================================
# NMS
# ============================================================

def nms(boxes, scores, iou_threshold):

    order = scores.argsort()[::-1]

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

        order = order[
            remaining + 1
        ]

    return keep


# ============================================================
# FULL YOLO DECODER
# ============================================================

def decode_outputs(outputs):

    all_boxes = []
    all_scores = []

    for i, stride in enumerate(STRIDES):

        # Output order:
        #
        # box0
        # score0
        # box1
        # score1
        # box2
        # score2

        box_output = outputs[i * 2]
        score_output = outputs[i * 2 + 1]

        boxes, scores = decode_scale(
            box_output,
            score_output,
            stride
        )

        all_boxes.append(boxes)
        all_scores.append(scores)

    boxes = np.concatenate(
        all_boxes,
        axis=0
    )

    scores = np.concatenate(
        all_scores,
        axis=0
    )

    # Confidence filtering
    mask = scores >= CONFIDENCE_THRESHOLD

    boxes = boxes[mask]
    scores = scores[mask]

    print(
        "Detections before NMS:",
        len(boxes)
    )

    if len(boxes) == 0:
        return [], []

    # NMS
    keep = nms(
        boxes,
        scores,
        NMS_IOU_THRESHOLD
    )

    boxes = boxes[keep]
    scores = scores[keep]

    return boxes, scores


# ============================================================
# MAIN
# ============================================================

if len(sys.argv) < 2:

    print(
        "Usage:"
    )

    print(
        "python scripts\\test_full_pipeline.py "
        "<image_path>"
    )

    sys.exit(1)


image_path = sys.argv[1]

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# Load original image
# ------------------------------------------------------------

original = cv2.imread(
    image_path
)

if original is None:

    print(
        "ERROR: Could not read image:"
    )

    print(image_path)

    sys.exit(1)


print(
    "Original image:",
    original.shape
)


# ------------------------------------------------------------
# Simulate ESP32-CAM 240x240 frame
#
# The real camera gives RGB565.
#
# Here we:
# BGR image
# -> RGB
# -> resize to 240x240
# -> RGB888 -> RGB565
# -> RGB565 -> RGB888
#
# This lets us test the complete camera preprocessing
# without having the ESP32-CAM connected.
# ------------------------------------------------------------

rgb_original = cv2.cvtColor(
    original,
    cv2.COLOR_BGR2RGB
)

camera_rgb = cv2.resize(
    rgb_original,
    (240, 240),
    interpolation=cv2.INTER_LINEAR
)


# RGB888 -> RGB565

r = camera_rgb[:, :, 0].astype(np.uint16)
g = camera_rgb[:, :, 1].astype(np.uint16)
b = camera_rgb[:, :, 2].astype(np.uint16)

rgb565 = (
    ((r >> 3) << 11)
    |
    ((g >> 2) << 5)
    |
    (b >> 3)
)


# RGB565 -> RGB888

camera_rgb_converted = rgb565_to_rgb888(
    rgb565
)


# ------------------------------------------------------------
# Resize camera frame -> model input
# ------------------------------------------------------------

model_image = cv2.resize(
    camera_rgb_converted,
    (INPUT_SIZE, INPUT_SIZE),
    interpolation=cv2.INTER_LINEAR
)


# ------------------------------------------------------------
# Prepare ONNX input
# ------------------------------------------------------------

input_tensor = (
    model_image.astype(np.float32)
    / 255.0
)

input_tensor = np.transpose(
    input_tensor,
    (2, 0, 1)
)

input_tensor = np.expand_dims(
    input_tensor,
    axis=0
)


print(
    "Model input:",
    input_tensor.shape
)

print(
    "Model input dtype:",
    input_tensor.dtype
)


# ============================================================
# LOAD MODEL
# ============================================================

print()
print("Loading ESP-DL-compatible ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print(
    "Input name:",
    input_name
)


# ============================================================
# RUN MODEL
# ============================================================

print()
print("Running YOLO11n...")

outputs = session.run(
    None,
    {
        input_name: input_tensor
    }
)


print(
    "Number of outputs:",
    len(outputs)
)

for i, output in enumerate(outputs):

    print(
        f"Output {i}:",
        output.shape
    )


# ============================================================
# DECODE
# ============================================================

print()
print("Decoding detections...")

boxes, scores = decode_outputs(
    outputs
)


print(
    "Final detections:",
    len(boxes)
)


# ============================================================
# DRAW RESULTS
# ============================================================

result = cv2.cvtColor(
    model_image,
    cv2.COLOR_RGB2BGR
)

for box, score in zip(
    boxes,
    scores
):

    x1, y1, x2, y2 = box

    x1 = max(
        0,
        min(INPUT_SIZE - 1, int(x1))
    )

    y1 = max(
        0,
        min(INPUT_SIZE - 1, int(y1))
    )

    x2 = max(
        0,
        min(INPUT_SIZE - 1, int(x2))
    )

    y2 = max(
        0,
        min(INPUT_SIZE - 1, int(y2))
    )

    cv2.rectangle(
        result,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    label = f"analog_watch {score:.3f}"

    cv2.putText(
        result,
        label,
        (x1, max(15, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1
    )

    print(
        f"Box: "
        f"{x1}, {y1}, {x2}, {y2} "
        f"Confidence: {score:.3f}"
    )


# ============================================================
# SAVE RESULT
# ============================================================

output_path = os.path.join(
    OUTPUT_DIR,
    "full_pipeline_result.jpg"
)

cv2.imwrite(
    output_path,
    result
)

print()
print(
    "Saved:",
    output_path
)