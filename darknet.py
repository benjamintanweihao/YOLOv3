import os
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, GlobalAveragePooling2D, \
    LeakyReLU, UpSampling2D
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers import Dense, ZeroPadding2D
from tensorflow.keras.utils import plot_model as plot


def darknet_base(inputs):
    """
    Builds Darknet53 by reading the YOLO configuration file

    :param inputs: Input tensor
    :return: A list of output layers
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
            x, layers = _build_yolo_layer(x, block, layers)

        elif block_type == 'route':
            x, layers = _build_route_layer(x, block, layers)

        elif block_type == 'upsample':
            x, layers = _build_upsample_layer(x, block, layers)

        else:
            raise ValueError('{} not recognized as block type'.format(block_type))

    # NOTE: All the indices with NONE are YOLO layers. Therefore, the layer right
    # NOTE: before the YOLO layer is an output layer, which we are interested in.
    outputs = [layers[i - 1] for i in range(len(layers)) if layers[i] is None]

    return outputs


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

    x = UpSampling2D(size=stride)(x)
    layers.append(x)

    return x, layers


def _build_route_layer(x, block, layers):
    selected_layers = [layers[int(l)] for l in block['layers'].split(',')]

    if len(selected_layers) == 1:
        x = selected_layers[0]
        layers.append(x)

        return x, layers

    elif len(selected_layers) == 2:
        x = Concatenate(axis=3)(selected_layers)
        layers.append(x)

        return x, layers

    else:
        raise ValueError('Invalid number of layers: {}'.format(len(selected_layers)))


def _build_shortcut_layer(x, block, layers):
    from_layer = layers[int(block['from'])]
    x = Add()([from_layer, x])

    assert block['activation'] == 'linear', 'Invalid activation: {}'.format(block['activation'])
    layers.append(x)

    return x, layers


def _build_yolo_layer(x, block, layers):
    # NOTE: Here we append None to specify that the preceding layer is a output layer
    layers.append(None)

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


inputs = Input(shape=(None, None, 3))
outputs = darknet_base(inputs)

# NOTE: Attach the Average Pooling layers to the last output layer
# last = outputs[-1]
# last = GlobalAveragePooling2D()(last)
# last = Dense(1000, activation='softmax')(last)
# outputs[-1] = last

model = Model(inputs, outputs)

print(model.summary())

plot(model, to_file='utils/model.png', show_shapes=True)
