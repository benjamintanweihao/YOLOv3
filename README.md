# :unicorn: YOLOv3 Implementation in TensorFlow 1.1x + Keras :unicorn:

# How it Looks Like

![](https://i.imgur.com/Phozn0T.png)

# Installation and Setup

Download weights into the `cfg` directory:

```
cd cfg
wget https://pjreddie.com/media/files/yolov3.weights
```

Install OpenCV:

```
pip install opencv-python
```

- [X] YOLO configuration parser
- [X] Build YOLO model
- [X] Check architecture against a well-known implementation
- [X] Load YOLO pre-trained weights
- [X] Handle YOLO layer (Detection Layer)
- [X] Non-Maximal Supression
- [X] Colorful boxes with labels and scores
- [X] Test out on a Webcam
- [ ] Check dependencies

# TODO

- [ ] Read `decay` and `momentum` from `net`

# Credits

* [Series: YOLO Object Detector in PyTorch by Ayoosh Kathuria](https://blog.paperspace.com/tag/series-yolo/)
* [Implementing YOLO v3 in Tensorflow (TF-Slim) by Paweł Kapica](https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe)
* [`xiaochus/YOLOv3`](https://github.com/xiaochus/YOLOv3)
* [`qqwweee/keras-yolo3`](https://github.com/qqwweee/keras-yolo3)
* [`kevinwuhoo/randomcolor`](https://github.com/kevinwuhoo/randomcolor-py)

