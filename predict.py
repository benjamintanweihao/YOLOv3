from tensorflow.keras import Input, Model
from tensorflow.keras.utils import plot_model as plot
from darknet import darknet_base

import cv2
import numpy as np

from data.COCOLabels import COCOLabels


def handle_predictions(predictions, confidence=0.6, iou_threshold=0.5):
    boxes = predictions[:, :, :4]
    box_confidences = np.expand_dims(predictions[:, :, 4], -1)
    box_class_probs = predictions[:, :, 5:]

    box_scores = box_confidences * box_class_probs
    box_classes = np.argmax(box_scores, axis=-1)
    box_class_scores = np.max(box_scores, axis=-1)
    pos = np.where(box_class_scores >= confidence)

    boxes = boxes[pos]
    classes = box_classes[pos]
    scores = box_class_scores[pos]

    nboxes, nclasses, nscores = [], [], []

    # TODO: Check if scaled properly
    for c in set(classes):
        inds = np.where(classes == c)
        b = boxes[inds]
        c = classes[inds]
        s = scores[inds]

        x = b[:, 0]
        y = b[:, 1]
        w = b[:, 2]
        h = b[:, 3]

        areas = w * h
        order = s.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x[i], x[order[1:]])
            yy1 = np.maximum(y[i], y[order[1:]])
            xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
            yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

            w1 = np.maximum(0.0, xx2 - xx1 + 1)
            h1 = np.maximum(0.0, yy2 - yy1 + 1)

            inter = w1 * h1
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        keep = np.array(keep)

        nboxes.append(b[keep])
        nclasses.append(c[keep])
        nscores.append(s[keep])

    if nboxes:
        boxes = np.concatenate(nboxes)
        classes = np.concatenate(nclasses)
        scores = np.concatenate(nscores)

        return boxes, classes, scores

    else:
        return None, None, None


def draw_boxes(boxes, classes, scores):
    if classes is None or len(classes) == 0:
        return

    labels = COCOLabels.all()

    for b, c, s in zip(boxes, classes, scores):
        x1, y1, w, h = b
        x2 = int(x1 + w)
        y2 = int(y1 + h)

        print("{}: {}".format(labels[c], s))
        cv2.rectangle(orig, (x1, y1), (x2, y2), (255, 0, 0), 1)

    cv2.imwrite("out.png", orig)


inputs = Input(shape=(None, None, 3))
outputs, config = darknet_base(inputs)

model = Model(inputs, outputs)
model.summary()

plot(model, to_file='utils/model.png', show_shapes=True)

# Feed in one image

orig = cv2.imread('utils/dog-cycle-car.png')
orig = cv2.resize(orig, (config['width'], config['height']))

img = orig.astype(np.float32)
img = img[:, :, ::-1]  # BGR -> RGB
img /= 255.0
img = np.expand_dims(img, axis=0)

b, c, s = handle_predictions(model.predict([img]))
draw_boxes(b, c, s)
