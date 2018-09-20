import os
from pprint import pprint
import tensorflow as tf
from tensorflow import keras


def build_model(path=os.path.join(os.getcwd(), 'cfg', 'yolov3.cfg')):
    """
    Builds Darknet53 by reading the YOLO configuration file

    :param path: Path to YOLO configuration file
    :return: Darknet53 model
    """
    blocks = parse_cfg(path)

    model = keras.Sequential()

    for block in blocks:
        block_type = block['type']

        if block_type == 'net':
            pass

        elif block_type == 'convolutional':
            pass

        elif block_type == 'shortcut':
            pass

        elif block_type == 'yolo':
            pass

        elif block_type == 'route':
            pass

        elif block_type == 'upsample':
            pass

        else:
            raise ValueError('{} not recognized as block type'.format(block_type))

    return model


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


build_model()
