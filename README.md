# :unicorn: YOLOv3 Implementation in TensorFlow 1.1x + Keras :unicorn:

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
- [X] Check against a well-known implementation
- [X] Load YOLO pre-trained weights
- [X] Handle YOLO layer (Detection Layer)
- [X] Non-Maximal Supression
- [X] Colorful boxes with labels and scores
- [X] Test out on a Webcam
- [ ] Check dependencies

# TODO

- [ ] Read `decay` and `momentum` from `net`
