import cv2
import numpy as np


def preprocess_image(img_arr, model_image_size):
    image = img_arr.astype('uint8')
    resized_image = cv2.resize(image, tuple(reversed(model_image_size)), cv2.INTER_AREA)
    image_data = resized_image.astype('float32')
    image_data /= 255.
    image_data = np.expand_dims(image_data, 0)  # Add batch dimension.
    return image, image_data
