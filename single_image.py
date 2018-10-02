import time
import cv2

from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model as plot

from darknet import darknet_base
from predict import predict

inputs = Input(shape=(None, None, 3))
outputs, config = darknet_base(inputs)

model = Model(inputs, outputs)
model.summary()

plot(model, to_file='utils/model.png', show_shapes=True)

orig = cv2.imread('data/dog-cycle-car.png')

start = time.time()
predict(model, orig, config, confidence=0.5, iou_threshold=0.4)
end = time.time()

print("Inference time: {:.2f}s".format(end - start))
