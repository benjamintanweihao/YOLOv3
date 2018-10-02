import time
import cv2
import tensorflow as tf

from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model as plot
from tensorflow.keras import backend as K

from darknet import darknet_base
from predict import predict
from utils.freeze_graph import freeze_session

inputs = Input(shape=(None, None, 3))
outputs, config = darknet_base(inputs)

model = Model(inputs, outputs)
model.summary()

frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

tf.train.write_graph(frozen_graph, "model", "yolov3.pb", as_text=False)

plot(model, to_file='utils/model.png', show_shapes=True)

orig = cv2.imread('data/dog-cycle-car.png')

start = time.time()
predict(model, orig, config, confidence=0.5, iou_threshold=0.4)
end = time.time()

print("Inference time: {:.2f}s".format(end - start))
