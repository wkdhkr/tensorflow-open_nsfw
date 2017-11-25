#!/usr/bin/env python
import sys
import argparse
import tensorflow as tf
import cv2
import numpy as np
import glob
import time

from model import OpenNsfwModel, InputType
from image_utils import create_tensorflow_image_loader
from image_utils import create_yahoo_image_loader

import numpy as np


IMAGE_LOADER_TENSORFLOW = "tensorflow"
IMAGE_LOADER_YAHOO = "yahoo"

def read_image(image1):
        H,W, _ = image1.shape
        if(W>H):
            x_off = (W-H)//2
            image1 = image1[:,x_off:x_off+H,:]
        elif(H<W):
            y_off = (H-W)//2
            image1 = image1[y_off:y_off+W,:,:]
        
        image1 = cv2.resize(image1,(256,256))

        H, W, _ = image1.shape
        h, w = (224, 224)

        h_off = max((H - h) // 2, 0)
        w_off = max((W - w) // 2, 0)
        image = image1[h_off:h_off + h, w_off:w_off + w, :]

        image = image.astype(np.float32, copy=False)

        VGG_MEAN = [104, 117, 123]
        image -= np.array(VGG_MEAN, dtype=np.float32)

        image = np.expand_dims(image,axis=0)
        return image

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument("input_file", help="Path to the input image.\
                        Only jpeg images are supported.")
    parser.add_argument("-m", "--model_weights", required=True,
                        help="Path to trained model weights file")

    parser.add_argument("-l", "--image_loader",
                        default=IMAGE_LOADER_YAHOO,
                        help="image loading mechanism",
                        choices=[IMAGE_LOADER_YAHOO, IMAGE_LOADER_TENSORFLOW])

    parser.add_argument("-t", "--input_type",
                        default=InputType.TENSOR.name.lower(),
                        help="input type",
                        choices=[InputType.TENSOR.name.lower(),
                                 InputType.BASE64_JPEG.name.lower()])

    args = parser.parse_args()

    model = OpenNsfwModel()

    with tf.Session() as sess:

        input_type = InputType[args.input_type.upper()]
        model.build(weights_path=args.model_weights, input_type=input_type)

        fn_load_image = None
        for myfile in glob.glob(args.input_file+'/*.jpg'):
            start_time = time.time()
            image1 = cv2.imread(myfile)

            image = read_image(image1)

            sess.run(tf.global_variables_initializer())



            predictions = \
                sess.run(model.predictions,
                         feed_dict={model.input: image})
            end_time =time.time()

            print("Results for '{}'".format(myfile))
            print("\tSFW score:\t{}\n\tNSFW score:\t{}".format(*predictions[0]))
            print("Run time {}".format(end_time-start_time))
            print("")

if __name__ == "__main__":
    main(sys.argv)
