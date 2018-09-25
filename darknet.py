import os
import numpy as np
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Add, BatchNormalization, Concatenate, Conv2D, \
    LeakyReLU, UpSampling2D
from tensorflow.keras.regularizers import l2
from tensorflow.python.keras.layers import ZeroPadding2D
from tensorflow.keras.utils import plot_model as plot
import tensorflow as tf
import tensorflow.keras.backend as K
import cv2

from utils.parser import Parser
from yolo_layer import YOLOLayer

# TODO: It seems like turning on eager execution always gives different values during
# TODO: inference. Weirdly, it also loads the network very fast compared to non-eager.
# TODO: It could be that in eager mode, the weights are not loaded. Need to verify
# TODO: this.
tf.enable_eager_execution()

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
    :return: A list of output (YOLO) layers and a dict containing a ptr to the weights file and network config
    """
    path = os.path.join(os.getcwd(), 'cfg', 'yolov3.cfg')
    blocks = Parser.parse_cfg(path)
    x, layers, yolo_layers = inputs, [], []
    ptr = 0
    config = {}

    for block in blocks:
        block_type = block['type']

        if block_type == 'net':
            config = _read_net_config(block)

        elif block_type == 'convolutional':
            x, layers, yolo_layers, ptr = _build_conv_layer(x, block, layers, yolo_layers, ptr)

        elif block_type == 'shortcut':
            x, layers, yolo_layers, ptr = _build_shortcut_layer(x, block, layers, yolo_layers, ptr)

        elif block_type == 'yolo':
            x, layers, yolo_layers, ptr = _build_yolo_layer(x, block, layers, yolo_layers, ptr, config)

        elif block_type == 'route':
            x, layers, yolo_layers, ptr = _build_route_layer(x, block, layers, yolo_layers, ptr)

        elif block_type == 'upsample':
            x, layers, yolo_layers, ptr = _build_upsample_layer(x, block, layers, yolo_layers, ptr)

        else:
            raise ValueError('{} not recognized as block type'.format(block_type))

    return tf.keras.layers.Concatenate(axis=1)(yolo_layers), {'ptr': ptr, 'config': config}


def _read_net_config(block):
    width = int(block['width'])
    height = int(block['height'])
    channels = int(block['channels'])

    return {
        'width': width,
        'height': height,
        'channels': channels}


def _build_conv_layer(x, block, layers, outputs, ptr):
    stride = int(block['stride'])
    filters = int(block['filters'])
    kernel_size = int(block['size'])
    pad = int(block['pad'])
    padding = 'same' if pad == 1 and stride == 1 else 'valid'
    use_batch_normalization = 'batch_normalize' in block

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
    if use_batch_normalization:
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
    if use_batch_normalization:
        conv_weights = [conv_weights]
    else:
        conv_weights = [conv_weights, conv_bias]

    if stride > 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)

    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=(stride, stride),
               padding=padding,
               use_bias=not use_batch_normalization,
               activation='linear',
               kernel_regularizer=l2(5e-4),
               weights=conv_weights)(x)

    if use_batch_normalization:
        x = BatchNormalization(weights=bn_weights_list)(x)

    assert block['activation'] in ['linear', 'leaky'], 'Invalid activation: {}'.format(block['activation'])

    if block['activation'] == 'leaky':
        x = LeakyReLU(alpha=0.1)(x)

    layers.append(x)

    return x, layers, outputs, ptr


def _build_upsample_layer(x, block, layers, outputs, ptr):
    stride = int(block['stride'])

    x = UpSampling2D(size=stride)(x)
    layers.append(x)

    return x, layers, outputs, ptr


def _build_route_layer(_x, block, layers, outputs, ptr):
    selected_layers = [layers[int(l)] for l in block['layers'].split(',')]

    if len(selected_layers) == 1:
        x = selected_layers[0]
        layers.append(x)

        return x, layers, outputs, ptr

    elif len(selected_layers) == 2:
        x = Concatenate(axis=3)(selected_layers)
        layers.append(x)

        return x, layers, outputs, ptr

    else:
        raise ValueError('Invalid number of layers: {}'.format(len(selected_layers)))


def _build_shortcut_layer(x, block, layers, outputs, ptr):
    from_layer = layers[int(block['from'])]
    x = Add()([from_layer, x])

    assert block['activation'] == 'linear', 'Invalid activation: {}'.format(block['activation'])
    layers.append(x)

    return x, layers, outputs, ptr


def _build_yolo_layer(x, block, layers, outputs, ptr, config):
    # Read indices of masks
    masks = [int(m) for m in block['mask'].split(',')]
    # Anchors used based on mask indices
    anchors = [a for a in block['anchors'].split(',  ')]
    anchors = [anchors[i] for i in range(len(anchors)) if i in masks]
    anchors = [tuple([int(a) for a in anchor.split(',')]) for anchor in anchors]
    classes = int(block['classes'])

    x = YOLOLayer(num_classes=classes, anchors=anchors, input_dims=(config['width'], config['height']))(x)
    outputs.append(x)
    # NOTE: Here we append None to specify that the preceding layer is a output layer
    layers.append(None)

    return x, layers, outputs, ptr


def verify_weights_completed_consumed(aux):
    remaining_weights = len(weights_file.read()) // 4
    weights_file.close()
    percentage = int((aux['ptr'] / (aux['ptr'] + remaining_weights)) * 100)
    print('Read {}% from Darknet weights.'.format(percentage))
    if remaining_weights > 0:
        print('Warning: {} unused weights'.format(remaining_weights))
    else:
        print('Weights loaded successfully!')


inputs = Input(shape=(None, None, 3))
outputs, aux = darknet_base(inputs)

verify_weights_completed_consumed(aux)

model = Model(inputs, outputs)
model.summary()

plot(model, to_file='utils/model.png', show_shapes=True)

# Feed in one image

orig = cv2.imread('utils/dog-cycle-car.png')
orig = cv2.resize(orig, (aux['config']['width'], aux['config']['height']))

img = orig.astype(np.float32)
img = img[:, :, ::-1]  # BGR -> RGB
img /= 255.0
img = np.expand_dims(img, axis=0)

predictions = model.predict([img])

# (1, 17328, 85)
# (batch, number of bounding boxes, scores)
(batches, bboxes, scores) = predictions.shape
for batch in range(batches):
    for bbox in range(bboxes):
        pred = predictions[batch][bbox]
        box_xy = pred[0:2]
        box_wh = pred[2:4]
        objectness = pred[4]
        class_scores = pred[5:]

        if objectness > 0.54:
            print(objectness)
            x1, y1 = box_xy
            w, h = box_wh

            x2 = x1 + w
            y2 = y1 + h

            cv2.rectangle(orig, (x1, y1), (x2, y2), (255, 0, 0), 1)

    cv2.imwrite("out.png", orig)
