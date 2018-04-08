#!/usr/bin/env python
import sys
import argparse
import tensorflow as tf
import cv2
import numpy as np
import glob
import time
import socket

from model import OpenNsfwModel, InputType
from flask import Flask, request, Response, jsonify

import numpy as np
import json

app = Flask(__name__)
sess = None
model = None

def is_used_port(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex(('127.0.0.1', port))

    if result == 0:
       return True
    else:
        return False

def detect_port(port):
    if not is_used_port(port):
        return port
    return detect_port(port + 1)

@app.route('/')
def index():
    return Response(open('./static/getImage.html').read(), mimetype="text/html")


@app.route('/image', methods=['POST'])
def image():
    i = request.files['image']
    data = np.fromstring(i.stream.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    network_data = read_image(img)
    global sess
    global model
    predictions = \
        sess.run(model.predictions,
                 feed_dict={model.input: network_data})
    #print("Predictions: nsfw ")
    #print(predictions)
    print(type(predictions[0][0].item()))
    result = {"sfw": predictions[0]
              [0].item(), "nsfw": predictions[0][1].item()}
    return jsonify(result)


def read_image(image1):
    H, W, _ = image1.shape
    if(W > H):
        x_off = (W-H)//2
        image1 = image1[:, x_off:x_off+H, :]
    elif(H < W):
        y_off = (H-W)//2
        image1 = image1[y_off:y_off+W, :, :]

    image1 = cv2.resize(image1, (256, 256))

    H, W, _ = image1.shape
    h, w = (224, 224)

    h_off = max((H - h) // 2, 0)
    w_off = max((W - w) // 2, 0)
    image = image1[h_off:h_off + h, w_off:w_off + w, :]

    image = image.astype(np.float32, copy=False)

    VGG_MEAN = [104, 117, 123]
    image -= np.array(VGG_MEAN, dtype=np.float32)

    image = np.expand_dims(image, axis=0)
    return image


def main(argv):
    global sess
    global model
    parser = argparse.ArgumentParser()

    parser.add_argument("-m", "--model_weights", required=True,
                        help="Path to trained model weights file")

    parser.add_argument("-t", "--input_type",
                        default=InputType.TENSOR.name.lower(),
                        help="input type",
                        choices=[InputType.TENSOR.name.lower(),
                                 InputType.BASE64_JPEG.name.lower()])

    parser.add_argument("-p", "--port", default=6000, help="port number")

    args = parser.parse_args()

    model = OpenNsfwModel()
    gpuConfig = tf.ConfigProto(
        gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.3),
        device_count={'GPU': 1}
    )
    sess = tf.Session(config=gpuConfig)
    if not(sess):
        exit(1)

    input_type = InputType[args.input_type.upper()]
    model.build(weights_path=args.model_weights, input_type=input_type)
    sess.run(tf.global_variables_initializer())
    print("Session  initialized. Running flask")
    port = detect_port(int(args.port))
    app.run(debug=False, host='0.0.0.0', port=port)


if __name__ == "__main__":
    main(sys.argv)
