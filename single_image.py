import time

from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model as plot
import cv2
import numpy as np

from darknet import darknet_base
from predict import handle_predictions, draw_boxes

inputs = Input(shape=(None, None, 3))
outputs, config = darknet_base(inputs)

model = Model(inputs, outputs)
model.summary()

plot(model, to_file='utils/model.png', show_shapes=True)

# Feed in one image

# orig = cv2.imread('data/dog-cycle-car.png')
orig = cv2.imread('data/keong_saik.jpg')
img = cv2.resize(orig, (config['width'], config['height']))

img = img.astype(np.float32)
img = img[:, :, ::-1]  # BGR -> RGB
img /= 255.0
img = np.expand_dims(img, axis=0)


start = time.time()
boxes, classes, scores = handle_predictions(model.predict([img]))
end = time.time()
print("Inference time: {:.2f}s".format(end - start))

# NOTE: Notice that we are passing in the _original_ image here, not `img`
# NOTE: that has been transformed and have its axis expanded (for batch size)
draw_boxes(orig, boxes, classes, scores, config)
