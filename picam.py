import time
import cv2
import picamera

from tensorflow.keras import Input, Model
from darknet import darknet_base
from predict import predict, predict_with_yolo_head


INCLUDE_YOLO_HEAD = False

inputs = Input(shape=(None, None, 3))
outputs, config = darknet_base(inputs, include_yolo_head=INCLUDE_YOLO_HEAD)
model = Model(inputs, outputs)

import time
import picamera
import numpy as np
import cv2

width, height = 640, 480

with picamera.PiCamera() as camera:
    camera.resolution = (width, height)
    camera.framerate = 2 # we don't need it to be s ofast
    time.sleep(2)

    while True:

        image = np.empty((height * width * 3,), dtype=np.uint8)
        camera.capture(image, 'bgr')
        image = image.reshape((height, width, 3))
        # Capture frame-by-frame
        start = time.time()

        if INCLUDE_YOLO_HEAD:
            output_image = predict(model, image, config, confidence=0.2, iou_threshold=0.3)
        else:
            output_image = predict_with_yolo_head(model, image, config, confidence=0.1, iou_threshold=0.2)

        # output_image = frame
        end = time.time()
        print("Inference time: {:.2f}s".format(end - start))

        # Display the resulting frame
        cv2.imshow('', output_image)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cv2.destroyAllWindows()
