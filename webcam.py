import time
import cv2
import numpy as np
from tensorflow.python.keras import Input, Model

from darknet import darknet_base
from predict import handle_predictions, draw_boxes

stream = cv2.VideoCapture(0)


inputs = Input(shape=(None, None, 3))
outputs, config = darknet_base(inputs)
model = Model(inputs, outputs)


def preprocess_webcam_image(img_arr, model_image_size):
    image = img_arr.astype('uint8')
    resized_image = cv2.resize(image, tuple(reversed(model_image_size)), cv2.INTER_AREA)
    image_data = resized_image.astype('float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data


def predict(model, frame):
    image, image_data = preprocess_webcam_image(frame, model_image_size=(config['width'], config['height']))

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
