import time

import cv2
import os
import numpy as np

from tensorflow.contrib.lite.python import interpreter as interpreter_wrapper

from definitions import ROOT_DIR
from predict import preprocess_image

model_path = os.path.join(ROOT_DIR, 'model', 'yolov3.tflite')

interpreter = interpreter_wrapper.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(input_details)
print(output_details)

# check the type of the input tensor
if input_details[0]['dtype'] == np.float32:
    floating_model = True

orig = cv2.imread('data/dog-cycle-car.png')

height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

image, image_data = preprocess_image(orig, (height, width))

start = time.time()
interpreter.set_tensor(input_details[0]['index'], image_data)
interpreter.invoke()
end = time.time()

output_data = interpreter.get_tensor(output_details[0]['index'])

print(output_data)

print("Inference time: {:.2f}s".format((end - start)))
