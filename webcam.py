import time
import cv2
import numpy as np

from tensorflow.keras import Input, Model
from darknet import darknet_base
from predict import handle_predictions, draw_boxes
from utils.preprocessor import preprocess_image

stream = cv2.VideoCapture(0)

inputs = Input(shape=(None, None, 3))
outputs, config = darknet_base(inputs)
model = Model(inputs, outputs)


def predict(model, frame):
    image, image_data = preprocess_image(frame, model_image_size=(config['width'], config['height']))

    boxes, classes, scores = handle_predictions(model.predict([image_data]))

    draw_boxes(image, boxes, classes, scores, config)

    return np.array(image)


while True:
    # Capture frame-by-frame
    grabbed, frame = stream.read()
    if not grabbed:
        break

    # Run detection
    start = time.time()
    output_image = predict(model, frame)
    end = time.time()
    print("Inference time: {:.2f}s".format(end - start))

    # Display the resulting frame
    cv2.imshow('', output_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
stream.release()
cv2.destroyAllWindows()
