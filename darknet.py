import os
import tensorflow as tf
from tensorflow.python.keras import Input
from tensorflow.python.keras.layers import BatchNormalization, Conv2D, UpSampling2D, Concatenate


def _build_conv_layer(x, block):
    stride = int(block['stride'])

    # TODO: Not too sure how to change the input to add padding
    pad = int(block['pad'])

    filters = int(block['filters'])
    kernel_size = int(block['size'])
    padding = 'valid' if stride == 2 else 'same'

    activation = None
    if block['activation'] == 'leaky':
        activation = tf.nn.leaky_relu

    assert block['activation'] in ['linear', 'leaky'], 'activation = {}'.format(block['activation'])

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=(stride, stride),
               padding=padding,
               activation=activation)(x)

    if 'batch_normalize' in block:
        x = BatchNormalization()(x)

    return x


def _build_upsample_layer(x, block):
    stride = int(block['stride'])

    return UpSampling2D(size=(stride, stride))(x)


def _build_route_layer(x, block):
    layers = [int(l) for l in block['layers'].split(',')]

    if len(layers) == 1:
        # TODO: How to read this layer?
        # layers[0]
        pass

    elif len(layers) == 2:
        # layer_1 = model.layers[layers[0]]
        # layer_2 = model.layers[layers[1]]  # TODO: Is this off by 1?

        # return Concatenate(axis=3)([layer_1, layer_2])(x)
        pass

    else:
        raise ValueError('Invalid number of layers: {}'.format(layers))

    return x


def build_model(path=os.path.join(os.getcwd(), 'cfg', 'yolov3.cfg')):
    """
    Builds Darknet53 by reading the YOLO configuration file

    :param path: Path to YOLO configuration file
    :return: Darknet53 model
    """

    blocks = parse_cfg(path)

    x = Input(shape=(10000, 10000, 3))

    for block in blocks:
        block_type = block['type']

        if block_type == 'net':
            pass

        elif block_type == 'convolutional':
            x = _build_conv_layer(x, block)

        elif block_type == 'shortcut':
            pass

        elif block_type == 'yolo':
            pass

        elif block_type == 'route':
            x = _build_route_layer(x, block)

        elif block_type == 'upsample':
            x = _build_upsample_layer(x, block)

        else:
            raise ValueError('{} not recognized as block type'.format(block_type))

    return x


def parse_cfg(path=os.path.join(os.getcwd(), 'cfg', 'yolov3.cfg')):
    with open(path) as cfg:
        lines = [line.rstrip() for line in cfg if line.rstrip()]
        lines = [line for line in lines if not line.startswith('#')]

        block = {}
        blocks = []

        for line in lines:
            if line.startswith('['):
                type = line[1:-1]
                if len(block) > 0:
                    blocks.append(block)
                block = {'type': type}
            else:
                key, value = [token.strip() for token in line.split('=')]
                block[key] = value

        blocks.append(block)

        return blocks


# NOTE: Still doesn't work yet.
model = build_model()
print(model.summary())
