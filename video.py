import cv2

from tensorflow.keras import Input, Model
from darknet import darknet_base
from predict import predict

inputs = Input(shape=(None, None, 3))
outputs, config = darknet_base(inputs)
model = Model(inputs, outputs)

vidcap = cv2.VideoCapture('data/demo.mp4')
success, image = vidcap.read()

size = (int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, size)


while success:
    output_image = predict(model, image, config)
    out.write(output_image)
    success, image = vidcap.read()

vidcap.release()
out.release()
