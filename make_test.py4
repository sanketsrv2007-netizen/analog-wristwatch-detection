import numpy as np
from PIL import Image
import tensorflow as tf

DIR = r"E:\Study\SEM-3\Robotics\Analog_wristwatch_detection"
IMG = DIR + r"\wrist-watch.v4i.yolov8\test\images\03ad9685d1353f119100437524c078b2_jpg.rf.4612c8739c58717e800a8858e7b805ae.jpg"
OUT = DIR + r"\esp32\camera\watch_camera\watch_camera\test_input.bin"
MODEL = DIR + r"\esp32\camera\watch_camera\watch_camera\best_int8.tflite"

im = Image.open(IMG).convert("RGB").resize((160, 160), Image.BILINEAR)
arr = (np.asarray(im, dtype=np.float32) / 255.0).transpose(2, 0, 1)[None]
arr.tofile(OUT)
print("saved", arr.shape, arr.nbytes, "bytes")

it = tf.lite.Interpreter(model_path=MODEL)
it.allocate_tensors()
it.set_tensor(it.get_input_details()[0]["index"], arr)
it.invoke()
out = it.get_tensor(it.get_output_details()[0]["index"])
conf = out[0, 4]
i = int(conf.argmax())
print("PC RESULT idx=%d conf=%.4f x=%.3f y=%.3f w=%.3f h=%.3f" %
      (i, conf[i], out[0, 0, i], out[0, 1, i], out[0, 2, i], out[0, 3, i]))