import tensorflow as tf
import numpy as np

MODEL = "runs/detect/runs/micro_watch_medium/weights/best_int8.tflite"

interpreter = tf.lite.Interpreter(model_path=MODEL)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("INPUT:")
print(input_details)

print("\nOUTPUT:")
print(output_details)

input_index = input_details[0]["index"]
output_index = output_details[0]["index"]

# Random input for testing
x = np.random.rand(1, 3, 160, 160).astype(np.float32)

interpreter.set_tensor(input_index, x)
interpreter.invoke()

output = interpreter.get_tensor(output_index)

print("\nOUTPUT SHAPE:", output.shape)
print("OUTPUT DTYPE:", output.dtype)

print("\nFirst 10 predictions:")
print(output[0, :, :10])

confidence = output[0, 4, :]

print("\nCONFIDENCE MIN:", confidence.min())
print("CONFIDENCE MAX:", confidence.max())
print("CONFIDENCE ARGMAX:", confidence.argmax())
print("CONFIDENCE AT 500:", confidence[500])

print("\nConfidence values:")
for i in range(20):
    print(i, confidence[i])
