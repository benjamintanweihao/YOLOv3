import tensorflow as tf


class YOLOLayer(tf.keras.layers.Layer):

    def __init__(self, classes, anchors, **kwargs):
        self.classes = classes
        self.anchors = anchors
        super(YOLOLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(YOLOLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        outputs = tf.reshape(inputs, shape=(-1, -1, 3, self.classes + 5))

        return outputs

    def compute_output_shape(self, input_shape):
        input_shape = tf.reshape(input_shape,
                                 shape=(input_shape[0], -1, 3, self.classes + 5))

        return input_shape.get_shape()

    def get_config(self):
        return super(YOLOLayer, self).get_config()

    @classmethod
    def from_config(cls, config):
        return cls(**config)
