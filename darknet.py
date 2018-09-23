import os
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, \
    LeakyReLU, UpSampling2D
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.keras.utils import plot_model as plot
import tensorflow.keras.backend as K

# NOTE: The original Darknet parser is at
# NOTE: https://github.com/pjreddie/darknet/blob/master/src/parser.c
weights_file = open('cfg/yolov3.weights', 'rb')
major, minor, revision = np.ndarray(
    shape=(3,), dtype='int32', buffer=weights_file.read(12))
if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000:
    seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
else:
    seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
print('Weights Header: ', major, minor, revision, seen)


def darknet_base(inputs):
    """
    Builds Darknet53 by reading the YOLO configuration file

    :param inputs: Input tensor
    :return: A list of output layers and a ptr to the weights file
    """
    path = os.path.join(os.getcwd(), 'cfg', 'yolov3.cfg')
    blocks = parse_cfg(path)
    x, layers = inputs, []
    ptr = 0

    for block in blocks:
        block_type = block['type']

        if block_type == 'net':
            pass

        elif block_type == 'convolutional':
            x, layers, ptr = _build_conv_layer(x, block, layers, ptr)

        elif block_type == 'shortcut':
            x, layers, ptr = _build_shortcut_layer(x, block, layers, ptr)

        elif block_type == 'yolo':
            x, layers, ptr = _build_yolo_layer(x, block, layers, ptr)

        elif block_type == 'route':
            x, layers, ptr = _build_route_layer(x, block, layers, ptr)

        elif block_type == 'upsample':
            x, layers, ptr = _build_upsample_layer(x, block, layers, ptr)

        else:
            raise ValueError('{} not recognized as block type'.format(block_type))

    # NOTE: All the indices with NONE are YOLO layers. Therefore, the layer right
    # NOTE: before the YOLO layer is an output layer, which we are interested in.
    outputs = [layers[i - 1] for i in range(len(layers)) if layers[i] is None]

    return outputs, ptr


def _build_conv_layer(x, block, layers, ptr):
    stride = int(block['stride'])

    # TODO: Not too sure how to change the input to add padding
    pad = int(block['pad'])

    filters = int(block['filters'])
    kernel_size = int(block['size'])
    padding = 'same' if pad == 1 and stride == 1 else 'valid'
    batch_normalize = 'batch_normalize' in block

    # Darknet serializes convolutional weights as:
    # [bias/beta, [gamma, mean, variance], conv_weights]

    prev_layer_shape = K.int_shape(x)
    weights_shape = (kernel_size, kernel_size, prev_layer_shape[-1], filters)
    darknet_w_shape = (filters, weights_shape[2], kernel_size, kernel_size)
    weights_size = np.product(weights_shape)

    # number of filters * 4 bytes
    conv_bias = np.ndarray(
        shape=(filters,),
        dtype='float32',
        buffer=weights_file.read(filters * 4)
    )
    ptr += filters

    bn_weights_list = []
    if batch_normalize:
        # [gamma, mean, variance] * filters * 4 bytes
        bn_weights = np.ndarray(
            shape=(3, filters),
            dtype='float32',
            buffer=weights_file.read(3 * filters * 4)
        )
        ptr += 3 * filters

        bn_weights_list = [
            bn_weights[0],  # scale gamma
            conv_bias,      # shift beta
            bn_weights[1],  # running mean
            bn_weights[2]   # running var
        ]

    conv_weights = np.ndarray(
        shape=darknet_w_shape,
        dtype='float32',
        buffer=weights_file.read(weights_size * 4))
    ptr += weights_size

    # DarkNet conv_weights are serialized Caffe-style:
    # (out_dim, in_dim, height, width)
    # We would like to set these to Tensorflow order:
    # (height, width, in_dim, out_dim)
    conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
    if batch_normalize:
        conv_weights = [conv_weights]
    else:
        conv_weights = [conv_weights, conv_bias]

    if stride > 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=(stride, stride),
               padding=padding,
               use_bias=not batch_normalize,
               activation='linear',
               kernel_regularizer=l2(5e-4),
               weights=conv_weights)(x)

    if batch_normalize:
        x = BatchNormalization(weights=bn_weights_list)(x)

    assert block['activation'] in ['linear', 'leaky'], 'Invalid activation: {}'.format(block['activation'])

    if block['activation'] == 'leaky':
        x = LeakyReLU(alpha=0.1)(x)

    layers.append(x)

    return x, layers, ptr


def _build_upsample_layer(x, block, layers, ptr):
    stride = int(block['stride'])

    x = UpSampling2D(size=stride)(x)
    layers.append(x)

    return x, layers, ptr


def _build_route_layer(_x, block, layers, ptr):
    selected_layers = [layers[int(l)] for l in block['layers'].split(',')]

    if len(selected_layers) == 1:
        x = selected_layers[0]
        layers.append(x)

        return x, layers, ptr

    elif len(selected_layers) == 2:
        x = Concatenate(axis=3)(selected_layers)
        layers.append(x)

        return x, layers, ptr

    else:
        raise ValueError('Invalid number of layers: {}'.format(len(selected_layers)))


def _build_shortcut_layer(x, block, layers, ptr):
    from_layer = layers[int(block['from'])]
    x = Add()([from_layer, x])

    assert block['activation'] == 'linear', 'Invalid activation: {}'.format(block['activation'])
    layers.append(x)

    return x, layers, ptr


def _build_yolo_layer(x, _block, layers, ptr):
    # NOTE: Here we append None to specify that the preceding layer is a output layer
    layers.append(None)

    return x, layers, ptr


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
outputs, weights_ptr = darknet_base(inputs)

model = Model(inputs, outputs)
model.summary()

plot(model, to_file='utils/model.png', show_shapes=True)

# Check that the weights file has been completely consumed.

remaining_weights = len(weights_file.read()) // 4
weights_file.close()
percentage = int((weights_ptr / (weights_ptr + remaining_weights)) * 100)
print('Read {}% from Darknet weights.'.format(percentage))
if remaining_weights > 0:
    print('Warning: {} unused weights'.format(remaining_weights))
else:
    print('Weights loaded successfully!')

