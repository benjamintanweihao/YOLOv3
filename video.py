import cv2
import numpy as np

from tensorflow.keras import Input, Model
from darknet import darknet_base
from predict import handle_predictions, draw_boxes
from utils.preprocessor import preprocess_image

inputs = Input(shape=(None, None, 3))
outputs, config = darknet_base(inputs)
model = Model(inputs, outputs)

vidcap = cv2.VideoCapture('data/demo.mp4')
success, image = vidcap.read()

size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, size)


def predict(model, frame):
    image, image_data = preprocess_image(frame, model_image_size=(config['width'], config['height']))

    boxes, classes, scores = handle_predictions(model.predict([image_data]))

    draw_boxes(image, boxes, classes, scores, config)

    return np.array(image)



while success:
    output_image = predict(model, image)
    out.write(output_image)
    success, image = vidcap.read()

vidcap.release()
out.release()
