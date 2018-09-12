import os
from pprint import pprint


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


pprint(parse_cfg())
