import os
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, GlobalAveragePooling2D, \
    LeakyReLU, UpSampling2D
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers import Dense, ZeroPadding2D


def darknet_base(inputs):
    """
    Builds Darknet53 by reading the YOLO configuration file

    :param path: Path to YOLO configuration file
    :return: Darknet53 model
    """
    path = os.path.join(os.getcwd(), 'cfg', 'yolov3.cfg')
    blocks = parse_cfg(path)
    x, layers = inputs, []

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

    return x


def _build_conv_layer(x, block, layers):
    stride = int(block['stride'])

    # TODO: Not too sure how to change the input to add padding
    pad = int(block['pad'])

    filters = int(block['filters'])
    kernel_size = int(block['size'])
    padding = 'same' if pad == 1 and stride == 1 else 'valid'

    if stride > 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=(stride, stride),
               padding=padding,
               activation='linear',
               kernel_regularizer=l2(5e-4))(x)

    if 'batch_normalize' in block:
        x = BatchNormalization()(x)

    assert block['activation'] in ['linear', 'leaky'], 'Invalid activation: {}'.format(block['activation'])

    if block['activation'] == 'leaky':
        x = LeakyReLU(alpha=0.1)(x)

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
        x = layers[layer_ids[0]]
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
    from_layer = layers[int(block['from'])]
    x = Add()([from_layer, x])

    assert block['activation'] == 'linear', 'Invalid activation: {}'.format(block['activation'])
    x = Activation('linear')(x)
    layers.append(x)

    return x, layers


def parse_cfg(path):
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


inputs = Input(shape=(416, 416, 3))
x = darknet_base(inputs)
x = GlobalAveragePooling2D()(x)
x = Dense(1000, activation='softmax')(x)

model = Model(inputs, x)

print(model.summary())
