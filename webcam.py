import time
import cv2

from tensorflow.keras import Input, Model
from darknet import darknet_base
from predict import predict

stream = cv2.VideoCapture(0)

inputs = Input(shape=(None, None, 3))
outputs, config = darknet_base(inputs)
model = Model(inputs, outputs)


while True:
    # Capture frame-by-frame
    grabbed, frame = stream.read()
    if not grabbed:
        break

    # Run detection
    start = time.time()
    output_image = predict(model, frame, config)
    end = time.time()
    print("Inference time: {:.2f}s".format(end - start))

    # Display the resulting frame
    cv2.imshow('', output_image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
stream.release()
cv2.destroyAllWindows()
