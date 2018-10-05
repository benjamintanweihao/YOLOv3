# :unicorn: YOLOv3 Implementation in TensorFlow 1.1x + Keras :unicorn:

## How it Looks Like

[![](https://img.youtube.com/vi/D_WXxjgQzKk/0.jpg)](https://youtu.be/D_WXxjgQzKk)

[Watch the demo](https://youtu.be/D_WXxjgQzKk)

## Quick Start

Create `conda` environment depending on whether you have a supported GPU or not:

```
conda env create -f environment-[c|g]pu.yml
source activate yolov3-[c|g]pu
```

Download weights into the `cfg` directory:

```
cd cfg
wget https://pjreddie.com/media/files/yolov3.weights
```

### Demo on Single Image:

```
python single_image.py
```

The output is stored on `out.png` in the root folder.

### Demo on Web Cam:


To see it live on your Web Cam:

```
python webcam.py
```

## Progress

- [X] YOLO configuration parser
- [X] Build YOLO model
- [X] Check architecture against a well-known implementation
- [X] Load YOLO pre-trained weights
- [X] Handle YOLO layer (Detection Layer)
- [X] Non-Maximal Suppression
- [X] Colorful boxes with labels and scores
- [X] Test out on a Web Cam
- [X] Check dependencies
- [X] Dependencies for CPU and GPU
- [X] Instructions for running the project
- [X] Use original scale of input image
- [X] YOLO head in a function
- [ ] Figure out Eager Execution + Loading Weights
- [X] Support Tiny YOLOv3
- [ ] Allow passing in options to use Tiny YOLOv3
- [ ] Investigate Quantization / Smaller Weights
- [ ] Try this out on a Raspi3
- [ ] Tensorflow.js (¯\\_(ツ)_/¯)

## Credits

* [Series: YOLO Object Detector in PyTorch by Ayoosh Kathuria](https://blog.paperspace.com/tag/series-yolo/)
* [Implementing YOLO v3 in Tensorflow (TF-Slim) by Paweł Kapica](https://itnext.io/implementing-yolo-v3-in-tensorflow-tf-slim-c3c55ff59dbe)
* [`xiaochus/YOLOv3`](https://github.com/xiaochus/YOLOv3)
* [`qqwweee/keras-yolo3`](https://github.com/qqwweee/keras-yolo3)
* [`kevinwuhoo/randomcolor`](https://github.com/kevinwuhoo/randomcolor-py)

