import tensorflow as tf
import numpy as np
from PIL import Image

MODEL = "runs/detect/runs/micro_watch_medium/weights/best_int8.tflite"

# Change this only if this image does not exist
IMAGE = "wrist-watch.v4i.yolov8/test/images"

import os
files = [
    os.path.join(IMAGE, f)
    for f in os.listdir(IMAGE)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Pick the first test image
image_path = files[0]

print("IMAGE:", image_path)

interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

img = Image.open(image_path).convert("RGB")
img = img.resize((160, 160))

x = np.asarray(img).astype(np.float32) / 255.0

# HWC -> NCHW
x = np.transpose(x, (2, 0, 1))
x = np.expand_dims(x, axis=0)

interpreter.set_tensor(input_index, x)
interpreter.invoke()

output = interpreter.get_tensor(output_index)

print("\nOUTPUT SHAPE:", output.shape)

confidence = output[0, 4, :]

print("\nCONFIDENCE MIN:", confidence.min())
print("CONFIDENCE MAX:", confidence.max())
print("CONFIDENCE ARGMAX:", confidence.argmax())

idx = confidence.argmax()

print("\nBEST RAW PREDICTION:")
print("X:", output[0, 0, idx])
print("Y:", output[0, 1, idx])
print("W:", output[0, 2, idx])
print("H:", output[0, 3, idx])
print("CONF:", output[0, 4, idx])

print("\nTOP 10 CONFIDENCES:")

top = np.argsort(confidence)[-10:][::-1]

for i in top:
    print(
        "index =", i,
        "x =", output[0, 0, i],
        "y =", output[0, 1, i],
        "w =", output[0, 2, i],
        "h =", output[0, 3, i],
        "conf =", output[0, 4, i]
    )
