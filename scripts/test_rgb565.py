import numpy as np
import cv2


def rgb565_to_rgb888(rgb565):
    """
    Convert RGB565 image data to RGB888.
    Input:  H x W uint16 array
    Output: H x W x 3 uint8 RGB image
    """

    r = ((rgb565 >> 11) & 0x1F) << 3
    g = ((rgb565 >> 5) & 0x3F) << 2
    b = (rgb565 & 0x1F) << 3

    rgb = np.stack([r, g, b], axis=-1)

    return rgb.astype(np.uint8)


# -------------------------------------------------
# Simulate one 240x240 RGB565 camera frame
# -------------------------------------------------

width = 240
height = 240

# Create a test RGB image
test_rgb = np.zeros((height, width, 3), dtype=np.uint8)

# Put a simple colored pattern in it
test_rgb[:, :, 0] = 255
test_rgb[:, :, 1] = 100
test_rgb[:, :, 2] = 50


# -------------------------------------------------
# Convert RGB888 -> RGB565
# This simulates what the ESP32 camera gives us.
# -------------------------------------------------

r = test_rgb[:, :, 0].astype(np.uint16)
g = test_rgb[:, :, 1].astype(np.uint16)
b = test_rgb[:, :, 2].astype(np.uint16)

rgb565 = (
    ((r >> 3) << 11)
    | ((g >> 2) << 5)
    | (b >> 3)
)


# -------------------------------------------------
# Convert RGB565 -> RGB888
# -------------------------------------------------

converted_rgb = rgb565_to_rgb888(rgb565)


# -------------------------------------------------
# Resize 240x240 -> 320x320
# -------------------------------------------------

model_input = cv2.resize(
    converted_rgb,
    (320, 320),
    interpolation=cv2.INTER_LINEAR
)


# -------------------------------------------------
# Convert to YOLO input format
# RGB uint8
#      ↓
# float32
#      ↓
# /255
#      ↓
# HWC -> CHW
#      ↓
# batch dimension
# -------------------------------------------------

input_tensor = model_input.astype(np.float32) / 255.0

input_tensor = np.transpose(
    input_tensor,
    (2, 0, 1)
)

input_tensor = np.expand_dims(
    input_tensor,
    axis=0
)


# -------------------------------------------------
# Print results
# -------------------------------------------------

print("RGB565 shape:", rgb565.shape)
print("RGB888 shape:", converted_rgb.shape)
print("Resized image shape:", model_input.shape)
print("Model input shape:", input_tensor.shape)
print("Model input dtype:", input_tensor.dtype)
print("Model input range:",
      input_tensor.min(),
      "to",
      input_tensor.max())


# -------------------------------------------------
# Save images so we can visually inspect them
# -------------------------------------------------

# OpenCV expects BGR when writing
cv2.imwrite(
    "results/rgb565_converted.jpg",
    cv2.cvtColor(converted_rgb, cv2.COLOR_RGB2BGR)
)

cv2.imwrite(
    "results/rgb565_320.jpg",
    cv2.cvtColor(model_input, cv2.COLOR_RGB2BGR)
)

print()
print("Saved:")
print("results/rgb565_converted.jpg")
print("results/rgb565_320.jpg")