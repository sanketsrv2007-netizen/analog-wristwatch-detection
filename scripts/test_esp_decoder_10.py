import os
import cv2
import numpy as np
import onnxruntime as ort

MODEL_PATH = r"models\best.onnx"

IMAGE_DIR = r"wrist-watch.v4i.yolov8\test\images"

OUTPUT_DIR = r"results\decoder_10"

IMG_SIZE = 320
CONF_THRESHOLD = 0.50
IOU_THRESHOLD = 0.45

STRIDES = [8, 16, 32]
REG_MAX = 16


# ---------------------------------------------------------
# Sigmoid
# ---------------------------------------------------------

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------
# DFL decoding
# ---------------------------------------------------------

def decode_dfl(box):

    box = box.reshape(4, REG_MAX)

    box_max = np.max(
        box,
        axis=1,
        keepdims=True
    )

    exp_box = np.exp(box - box_max)

    probabilities = (
        exp_box /
        np.sum(
            exp_box,
            axis=1,
            keepdims=True
        )
    )

    bins = np.arange(
        REG_MAX,
        dtype=np.float32
    )

    distances = np.sum(
        probabilities * bins,
        axis=1
    )

    return distances


# ---------------------------------------------------------
# Anchor generation
# ---------------------------------------------------------

def make_anchors(height, width):

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


# ---------------------------------------------------------
# Decode one scale
# ---------------------------------------------------------

def decode_scale(
    box_output,
    score_output,
    stride
):

    _, _, height, width = box_output.shape

    boxes = box_output[0].transpose(
        1, 2, 0
    )

    boxes = boxes.reshape(
        -1,
        64
    )

    scores = score_output[0, 0].reshape(-1)

    scores = sigmoid(scores)

    anchors = make_anchors(
        height,
        width
    )

    detections = []

    for i in range(height * width):

        confidence = scores[i]

        if confidence < CONF_THRESHOLD:
            continue

        left, top, right, bottom = decode_dfl(
            boxes[i]
        )

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

        x1 = np.clip(
            x1,
            0,
            IMG_SIZE
        )

        y1 = np.clip(
            y1,
            0,
            IMG_SIZE
        )

        x2 = np.clip(
            x2,
            0,
            IMG_SIZE
        )

        y2 = np.clip(
            y2,
            0,
            IMG_SIZE
        )

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


# ---------------------------------------------------------
# IoU
# ---------------------------------------------------------

def calculate_iou(
    box_a,
    box_b
):

    ax1, ay1, ax2, ay2 = box_a[:4]

    bx1, by1, bx2, by2 = box_b[:4]

    intersection_x1 = max(
        ax1,
        bx1
    )

    intersection_y1 = max(
        ay1,
        by1
    )

    intersection_x2 = min(
        ax2,
        bx2
    )

    intersection_y2 = min(
        ay2,
        by2
    )

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


# ---------------------------------------------------------
# NMS
# ---------------------------------------------------------

def nms(
    detections,
    iou_threshold
):

    if not detections:
        return []

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
                remaining.append(
                    detection
                )

        detections = remaining

    return kept


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("Loading ONNX model...")

session = ort.InferenceSession(
    MODEL_PATH,
    providers=[
        "CPUExecutionProvider"
    ]
)

input_name = session.get_inputs()[0].name

print("Model loaded successfully.")
print()


# ---------------------------------------------------------
# Create output directory
# ---------------------------------------------------------

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ---------------------------------------------------------
# Get first 10 images
# ---------------------------------------------------------

image_files = sorted(
    [
        f
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png"
            )
        )
    ]
)

image_files = image_files[:10]

print(
    f"Testing {len(image_files)} images..."
)

print()


# ---------------------------------------------------------
# Process images
# ---------------------------------------------------------

for index, filename in enumerate(
    image_files,
    start=1
):

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"[{index}/10] "
            f"FAILED: {filename}"
        )

        continue

    original_height, original_width = (
        image.shape[:2]
    )

    # BGR → RGB
    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    resized = cv2.resize(
        image_rgb,
        (IMG_SIZE, IMG_SIZE)
    )

    # Normalize
    model_input = (
        resized.astype(
            np.float32
        ) / 255.0
    )

    # HWC → CHW
    model_input = np.transpose(
        model_input,
        (2, 0, 1)
    )

    # Add batch
    model_input = np.expand_dims(
        model_input,
        axis=0
    )

    # -----------------------------------------------------
    # Inference
    # -----------------------------------------------------

    outputs = session.run(
        None,
        {
            input_name:
            model_input
        }
    )

    box0, score0 = outputs[0], outputs[1]
    box1, score1 = outputs[2], outputs[3]
    box2, score2 = outputs[4], outputs[5]

    # -----------------------------------------------------
    # Decode
    # -----------------------------------------------------

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

    before_nms = len(
        detections
    )

    final_detections = nms(
        detections,
        IOU_THRESHOLD
    )

    # -----------------------------------------------------
    # Draw detections
    # -----------------------------------------------------

    output_image = resized.copy()

    for detection in final_detections:

        x1, y1, x2, y2, confidence = (
            detection
        )

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        cv2.rectangle(
            output_image,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = (
            f"analog_watch "
            f"{confidence:.2f}"
        )

        cv2.putText(
            output_image,
            label,
            (
                x1,
                max(y1 - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    # RGB → BGR
    output_image = cv2.cvtColor(
        output_image,
        cv2.COLOR_RGB2BGR
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{index:02d}_{filename}"
    )

    cv2.imwrite(
        output_path,
        output_image
    )

    # -----------------------------------------------------
    # Print result
    # -----------------------------------------------------

    if final_detections:

        best_confidence = max(
            d[4]
            for d in final_detections
        )

        print(
            f"[{index}/10] "
            f"{filename}"
        )

        print(
            f"    Before NMS : {before_nms}"
        )

        print(
            f"    Final boxes : "
            f"{len(final_detections)}"
        )

        print(
            f"    Best conf   : "
            f"{best_confidence:.3f}"
        )

    else:

        print(
            f"[{index}/10] "
            f"{filename}"
        )

        print(
            f"    No detection"
        )

    print()


print("===================================")
print("10-image decoder test completed.")
print("Results saved to:")
print(OUTPUT_DIR)
print("===================================")