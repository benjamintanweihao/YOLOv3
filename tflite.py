# NOTE: Because of a bug in TensorFlow, this should be run in the console
# NOTE: python tflite.py
import os
import tensorflow as tf

from tensorflow.contrib.lite.python import lite
from tensorflow.keras import Input, Model

from darknet import darknet_base
from definitions import ROOT_DIR

inputs = Input(shape=(None, None, 3))
# NOTE: Here, we do not include the YOLO head because TFLite does not
# NOTE: support custom layers yet. Therefore, we'll need to implement
# NOTE: the YOLO head ourselves.
outputs, config = darknet_base(inputs, include_yolo_head=False)

model = Model(inputs, outputs)
model_path = os.path.join(ROOT_DIR, 'model', 'yolov3.h5')

tf.keras.models.save_model(model, model_path, overwrite=True)

# Sanity check to see if model loads properly
# NOTE: See https://github.com/keras-team/keras/issues/4609#issuecomment-329292173
# on why we have to pass in `tf: tf` in `custom_objects`
model = tf.keras.models.load_model(model_path,
                                   custom_objects={'tf': tf})

converter = lite.TocoConverter.from_keras_model_file(model_path,
                                                     input_shapes={'input_1': [1, config['width'], config['height'], 3]})
converter.post_training_quantize = True
tflite_model = converter.convert()

open("model/yolov3.tflite", "wb").write(tflite_model)
