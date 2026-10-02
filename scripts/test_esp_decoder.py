import onnxruntime as ort
import numpy as np
import cv2
import sys


# =========================================================
# SETTINGS
# =========================================================

MODEL_PATH = r"models\best.onnx"

IMG_SIZE = 320
CONF_THRESHOLD = 0.50

STRIDES = [8, 16, 32]
REG_MAX = 16

NMS_IOU_THRESHOLD = 0.45


# =========================================================
# IMAGE PATH
# =========================================================

IMAGE_PATH = (
    sys.argv[1]
    if len(sys.argv) > 1
    else r"wrist-watch.v4i.yolov8\test\images\03ad9685d1353f119100437524c078b2_jpg.rf.4612c8739c58717e800a8858e7b805ae.jpg"
)


# =========================================================
# SIGMOID
# =========================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# =========================================================
# DFL DECODING
# =========================================================

def decode_dfl(box):
    """
    Decode one 64-value YOLO DFL prediction.

    64 values = 4 coordinates × 16 bins

    Output:
        left, top, right, bottom
    """

    box = box.reshape(4, REG_MAX)

    # Softmax over the 16 bins
    box_max = np.max(
        box,
        axis=1,
        keepdims=True
    )

    exp_box = np.exp(box - box_max)

    probabilities = exp_box / np.sum(
        exp_box,
        axis=1,
        keepdims=True
    )

    # Bins: 0 ... 15
    bins = np.arange(
        REG_MAX,
        dtype=np.float32
    )

    # Expected value
    distances = np.sum(
        probabilities * bins,
        axis=1
    )

    return distances


# =========================================================
# GENERATE YOLO ANCHOR POINTS
# =========================================================

def make_anchors(height, width):
    """
    Generate YOLO anchor points using
    a 0.5 grid-cell offset.
    """

    y, x = np.meshgrid(
        np.arange(height, dtype=np.float32) + 0.5,
        np.arange(width, dtype=np.float32) + 0.5,
        indexing="ij"
    )

    anchors = np.stack(
        [x, y],
        axis=-1
    )

    return anchors.reshape(-1, 2)


# =========================================================
# DECODE ONE SCALE
# =========================================================

def decode_scale(box_output, score_output, stride):
    """
    Decode one YOLO detection scale.

    box_output:
        (1, 64, H, W)

    score_output:
        (1, 1, H, W)
    """

    _, _, height, width = box_output.shape

    # (1, 64, H, W)
    #        ↓
    # (H*W, 64)
    boxes = box_output[0].transpose(1, 2, 0)
    boxes = boxes.reshape(-1, 64)

    # (1, 1, H, W)
    #        ↓
    # (H*W,)
    scores = score_output[0, 0].reshape(-1)

    # Raw logits -> probability
    scores = sigmoid(scores)

    # Generate anchors
    anchors = make_anchors(
        height,
        width
    )

    detections = []

    for i in range(height * width):

        confidence = scores[i]

        if confidence < CONF_THRESHOLD:
            continue

        # Decode DFL
        left, top, right, bottom = decode_dfl(
            boxes[i]
        )

        # Convert grid coordinates to 320x320 pixels
        x1 = (
            anchors[i, 0] - left
        ) * stride

        y1 = (
            anchors[i, 1] - top
        ) * stride

        x2 = (
            anchors[i, 0] + right
        ) * stride

        y2 = (
            anchors[i, 1] + bottom
        ) * stride

        # Keep coordinates inside 320x320
        x1 = np.clip(x1, 0, IMG_SIZE)
        y1 = np.clip(y1, 0, IMG_SIZE)
        x2 = np.clip(x2, 0, IMG_SIZE)
        y2 = np.clip(y2, 0, IMG_SIZE)

        detections.append(
            [
                x1,
                y1,
                x2,
                y2,
                confidence
            ]
        )

    return detections


# =========================================================
# IOU
# =========================================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a[:4]
    bx1, by1, bx2, by2 = box_b[:4]

    intersection_x1 = max(ax1, bx1)
    intersection_y1 = max(ay1, by1)

    intersection_x2 = min(ax2, bx2)
    intersection_y2 = min(ay2, by2)

    intersection_width = max(
        0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0,
        intersection_y2 - intersection_y1
    )

    intersection_area = (
        intersection_width *
        intersection_height
    )

    area_a = (
        max(0, ax2 - ax1) *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1) *
        max(0, by2 - by1)
    )

    union_area = (
        area_a +
        area_b -
        intersection_area
    )

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


# =========================================================
# NMS
# =========================================================

def nms(detections, iou_threshold=0.45):

    if not detections:
        return []

    # Highest confidence first
    detections = sorted(
        detections,
        key=lambda x: x[4],
        reverse=True
    )

    kept = []

    while detections:

        best = detections.pop(0)

        kept.append(best)

        remaining = []

        for detection in detections:

            iou = calculate_iou(
                best,
                detection
            )

            if iou < iou_threshold:
                remaining.append(detection)

        detections = remaining

    return kept


# =========================================================
# LOAD ONNX MODEL
# =========================================================

session = ort.InferenceSession(
    MODEL_PATH,
    providers=["CPUExecutionProvider"]
)

print("Model loaded successfully.")


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:

    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

print(
    "Original image shape:",
    image.shape
)

original_h, original_w = image.shape[:2]


# =========================================================
# ULTRALYTICS-STYLE LETTERBOX
# =========================================================

# Calculate scale while preserving aspect ratio
scale = min(
    IMG_SIZE / original_w,
    IMG_SIZE / original_h
)

# New dimensions after scaling
new_w = int(
    round(original_w * scale)
)

new_h = int(
    round(original_h * scale)
)

# Resize while preserving aspect ratio
resized = cv2.resize(
    image,
    (new_w, new_h),
    interpolation=cv2.INTER_LINEAR
)

# Calculate padding
pad_x = (IMG_SIZE - new_w) // 2
pad_y = (IMG_SIZE - new_h) // 2

# Create 320x320 image filled with 114
letterboxed = np.full(
    (IMG_SIZE, IMG_SIZE, 3),
    114,
    dtype=np.uint8
)

# Put resized image in the center
letterboxed[
    pad_y:pad_y + new_h,
    pad_x:pad_x + new_w
] = resized


# =========================================================
# BGR -> RGB
# =========================================================

image_rgb = cv2.cvtColor(
    letterboxed,
    cv2.COLOR_BGR2RGB
)


# =========================================================
# NORMALIZE
# =========================================================

image_input = image_rgb.astype(
    np.float32
) / 255.0


# =========================================================
# HWC -> CHW
# =========================================================

image_input = np.transpose(
    image_input,
    (2, 0, 1)
)


# =========================================================
# ADD BATCH DIMENSION
# =========================================================

model_input = np.expand_dims(
    image_input,
    axis=0
)


# =========================================================
# RUN ONNX INFERENCE
# =========================================================

input_name = session.get_inputs()[0].name

outputs = session.run(
    None,
    {
        input_name: model_input
    }
)


# =========================================================
# MAP OUTPUTS
# =========================================================

box0 = outputs[0]
score0 = outputs[1]

box1 = outputs[2]
score1 = outputs[3]

box2 = outputs[4]
score2 = outputs[5]


# =========================================================
# DECODE ALL THREE SCALES
# =========================================================

detections = []

detections.extend(
    decode_scale(
        box0,
        score0,
        STRIDES[0]
    )
)

detections.extend(
    decode_scale(
        box1,
        score1,
        STRIDES[1]
    )
)

detections.extend(
    decode_scale(
        box2,
        score2,
        STRIDES[2]
    )
)


print()
print(
    "Detections before NMS:",
    len(detections)
)


# =========================================================
# NMS
# =========================================================

final_detections = nms(
    detections,
    iou_threshold=NMS_IOU_THRESHOLD
)


# =========================================================
# CONVERT BOXES FROM 320x320
# BACK TO ORIGINAL IMAGE
# =========================================================

converted_detections = []

for detection in final_detections:

    x1, y1, x2, y2, confidence = detection

    # Remove letterbox padding
    x1 = (x1 - pad_x) / scale
    y1 = (y1 - pad_y) / scale

    x2 = (x2 - pad_x) / scale
    y2 = (y2 - pad_y) / scale

    # Clip to original image
    x1 = np.clip(
        x1,
        0,
        original_w - 1
    )

    y1 = np.clip(
        y1,
        0,
        original_h - 1
    )

    x2 = np.clip(
        x2,
        0,
        original_w - 1
    )

    y2 = np.clip(
        y2,
        0,
        original_h - 1
    )

    converted_detections.append(
        [
            x1,
            y1,
            x2,
            y2,
            confidence
        ]
    )


# =========================================================
# PRINT FINAL DETECTIONS
# =========================================================

print()
print(
    "Final detections:",
    len(converted_detections)
)

for i, detection in enumerate(
    converted_detections
):

    x1, y1, x2, y2, confidence = detection

    print(
        f"Detection {i + 1}: "
        f"x1={x1:.1f}, "
        f"y1={y1:.1f}, "
        f"x2={x2:.1f}, "
        f"y2={y2:.1f}, "
        f"confidence={confidence:.3f}"
    )


# =========================================================
# DRAW RESULTS
# =========================================================

output_image = image.copy()

for detection in converted_detections:

    x1, y1, x2, y2, confidence = detection

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    # Draw bounding box
    cv2.rectangle(
        output_image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2
    )

    # Label
    label = (
        f"analog_watch "
        f"{confidence:.2f}"
    )

    cv2.putText(
        output_image,
        label,
        (x1, max(y1 - 10, 20)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2
    )


# =========================================================
# SAVE RESULT
# =========================================================

OUTPUT_PATH = r"results\esp_decoder_test.jpg"

cv2.imwrite(
    OUTPUT_PATH,
    output_image
)

print()
print("Saved result to:")
print(OUTPUT_PATH)