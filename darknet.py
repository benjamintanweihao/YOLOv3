import os
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Conv2D, UpSampling2D, Concatenate


def build_model(path=os.path.join(os.getcwd(), 'cfg', 'yolov3.cfg')):
    """
    Builds Darknet53 by reading the YOLO configuration file

    :param path: Path to YOLO configuration file
    :return: Darknet53 model
    """

    blocks = parse_cfg(path)
    layers = []

    x = input_layer = Input(shape=(None, None, 3))

    for block in blocks:
        block_type = block['type']

        if block_type == 'net':
            pass

        elif block_type == 'convolutional':
            x, layers = _build_conv_layer(x, block, layers)

        elif block_type == 'shortcut':
            x, layers = _build_shortcut_layer(x, block, layers)

        elif block_type == 'yolo':
            pass

        elif block_type == 'route':
            x, layers = _build_route_layer(x, block, layers)

        elif block_type == 'upsample':
            x, layers = _build_upsample_layer(x, block, layers)

        else:
            raise ValueError('{} not recognized as block type'.format(block_type))

    return Model(inputs=input_layer, outputs=layers[1:])


def _build_conv_layer(x, block, layers):
    stride = int(block['stride'])

    # TODO: Not too sure how to change the input to add padding
    pad = int(block['pad'])

    filters = int(block['filters'])
    kernel_size = int(block['size'])
    padding = 'valid' if stride == 2 else 'same'

    assert block['activation'] in ['linear', 'leaky'], 'Invalid activation: {}'.format(block['activation'])

    activation = None
    if block['activation'] == 'leaky':
        activation = tf.nn.leaky_relu

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=(stride, stride),
               padding=padding,
               activation=activation)(x)

    if 'batch_normalize' in block:
        x = BatchNormalization()(x)

    layers.append(x)
    return x, layers


def _build_upsample_layer(x, block, layers):
    stride = int(block['stride'])

    x = UpSampling2D(size=(stride, stride))(x)
    layers.append(x)

    return x, layers


def _build_route_layer(x, block, layers):
    layer_ids = [int(l) for l in block['layers'].split(',')]

    if len(layer_ids) == 1:
        x = layers[0]
        layers.append(x)

        return x, layers

    elif len(layer_ids) == 2:
        layer_1 = layers[layer_ids[0]]
        layer_2 = layers[layer_ids[1]]

        x = Concatenate(axis=3)([layer_1, layer_2])
        layers.append(x)

        return x, layers

    else:
        raise ValueError('Invalid number of layers: {}'.format(layer_ids))


def _build_shortcut_layer(x, block, layers):
    assert block['activation'] == 'linear', 'Invalid activation: {}'.format(block['activation'])

    from_layer = layers[int(block['from'])]
    x = Add()([from_layer, x])
    layers.append(x)

    return x, layers


def parse_cfg(path=os.path.join(os.getcwd(), 'cfg', 'yolov3.cfg')):
    with open(path) as cfg:
        lines = [line.rstrip() for line in cfg if line.rstrip()]
        lines = [line for line in lines if not line.startswith('#')]

        block = {}
        blocks = []

        for line in lines:
            if line.startswith('['):
                block_type = line[1:-1]
                if len(block) > 0:
                    blocks.append(block)
                block = {'type': block_type}
            else:
                key, value = [token.strip() for token in line.split('=')]
                block[key] = value

        blocks.append(block)

        return blocks


model = build_model()
print(model.summary())
