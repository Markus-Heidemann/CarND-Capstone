from styx_msgs.msg import TrafficLight

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm
import cv2 as cv
import h5py

from keras.models import load_model
from keras import __version__ as keras_version


class TLClassifierSimple(object):
    def __init__(self):

        # load the model for the traffic light bounding box detection
        SSD_GRAPH_FILE = './models/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
        detection_graph = self.load_graph(SSD_GRAPH_FILE)

        # The input placeholder for the image.
        # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        # Each box represents a part of the image where a particular object was detected.
        self.detection_boxes = detection_graph.get_tensor_by_name(
            'detection_boxes:0')

        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        self.detection_scores = detection_graph.get_tensor_by_name(
            'detection_scores:0')

        # The classification of the object (integer id).
        self.detection_classes = detection_graph.get_tensor_by_name(
            'detection_classes:0')

        self.sess = tf.Session(graph=detection_graph)


        # Load the model for the traffic light state classification
        global keras_version

        TL_CNN_H5 = './models/tl_state_classifier/model.h5'
        f = h5py.File(TL_CNN_H5, mode='r')
        model_version = f.attrs.get('keras_version')
        keras_version = str(keras_version).encode('utf8')

        # if model_version != keras_version:
        #     print('You are using Keras version ', keras_version,
        #         ', but the model was built using ', model_version)

        global tl_state_model
        tl_state_model = load_model(TL_CNN_H5)
        global tl_state_graph
        tl_state_graph = tf.get_default_graph()


    def filter_boxes(self, min_score, boxes, scores, classes):
        """Return boxes with a confidence >= `min_score`"""
        n = len(classes)
        idxs = []
        for i in range(n):
            if scores[i] >= min_score and classes[i] == 10:
                idxs.append(i)

        filtered_boxes = boxes[idxs, ...]
        filtered_scores = scores[idxs, ...]
        filtered_classes = classes[idxs, ...]
        return filtered_boxes, filtered_scores, filtered_classes


    def to_image_coords(self, box, height, width):
        """
        The original box coordinate output is normalized, i.e [0, 1].

        This converts it back to the original coordinate based on the image
        size.
        """
        box_coords = np.zeros_like(box)
        box_coords[0] = box[0] * height
        box_coords[1] = box[1] * width
        box_coords[2] = box[2] * height
        box_coords[3] = box[3] * width

        return box_coords


    def load_graph(self, graph_file):
        """Loads a frozen inference graph"""
        graph = tf.Graph()
        with graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(graph_file, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return graph


    def crop_box(self, image, box):
        bottom, left, top, right = box[...]
        return image[int(bottom):int(top), int(left):int(right), :]


    def detect_tl_circles(self, img):
        height = img.shape[0]
        img = img[:, :, ::-1].copy()  # convert to BGR
        gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gimg = cv.medianBlur(gimg, 5)
        circles = cv.HoughCircles(gimg, cv.HOUGH_GRADIENT, 1, int(height * 0.25),
                                param1=50, param2=30, minRadius=0, maxRadius=0)
        if circles is not None:
            circles = np.uint16(np.around(circles))[0]
        return circles


    def sort_circles_by_y(self, circles):
        if circles is not None:
            if len(circles) == 3:
                # determine top, middle and bottom circle w.r.t. y-coord
                top_idx = np.argmin(circles[:, 1])
                top_circ = circles[top_idx]
                circles = np.delete(circles, top_idx, axis=0)

                mid_idx = np.argmin(circles[:, 1])
                mid_circ = circles[mid_idx]
                circles = np.delete(circles, mid_idx, axis=0)

                bot_circ = circles[0]

                return top_circ, mid_circ, bot_circ
        return None, None, None



    def apply_color_threshold(self, img):
        img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        img = cv.medianBlur(img, 5)

        # RED
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([30, 255, 255])
        mask = cv.inRange(img, lower_red, upper_red)
        res = cv.bitwise_and(img, img, mask=mask)

        # since the H value is circular and red les between 160 and 30,
        # we have to deal with this here
        lower_red_1 = np.array([160, 50, 50])
        upper_red_1 = np.array([180, 255, 255])
        mask = cv.inRange(img, lower_red_1, upper_red_1)
        res_1 = cv.bitwise_and(img, img, mask=mask)
        res_red = cv.bitwise_or(res, res_1)

        # YELLOW
        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([30, 255, 255])
        mask = cv.inRange(img, lower_yellow, upper_yellow)
        res_yellow = cv.bitwise_and(img, img, mask=mask)

        # GREEN
        lower_green = np.array([60, 50, 50])
        upper_green = np.array([80, 255, 255])
        mask = cv.inRange(img, lower_green, upper_green)
        res_green = cv.bitwise_and(img, img, mask=mask)

        # combine results for red, yellow and green
        res = cv.bitwise_or(res_red, res_green)
        res = cv.bitwise_or(res, res_yellow)
        res = cv.cvtColor(res, cv.COLOR_RGB2GRAY)
        res[res > 0] = 255

        return res


    def determine_active_light(self, thresh_img, red_circ, yellow_circ, green_circ):
        # create binary circle mask
        circle_image_red = np.zeros(
            (thresh_img.shape[0], thresh_img.shape[1]), np.uint8)
        circle_image_yellow = np.zeros(
            (thresh_img.shape[0], thresh_img.shape[1]), np.uint8)
        circle_image_green = np.zeros(
            (thresh_img.shape[0], thresh_img.shape[1]), np.uint8)

        cv.circle(circle_image_red,
                (red_circ[0], red_circ[1]), red_circ[2], 255, -1)
        cv.circle(circle_image_yellow,
                (yellow_circ[0], yellow_circ[1]), yellow_circ[2], 255, -1)
        cv.circle(circle_image_green,
                (green_circ[0], green_circ[1]), green_circ[2], 255, -1)

        sum_red_pix = sum(sum(circle_image_red == 255))
        sum_yellow_pix = sum(sum(circle_image_yellow == 255))
        sum_green_pix = sum(sum(circle_image_green == 255))

        red_overlap = cv.bitwise_and(thresh_img, circle_image_red)
        yellow_overlap = cv.bitwise_and(thresh_img, circle_image_yellow)
        green_overlap = cv.bitwise_and(thresh_img, circle_image_green)

        sum_red_overlap = sum(sum(red_overlap == 255))
        sum_yellow_overlap = sum(sum(yellow_overlap == 255))
        sum_green_overlap = sum(sum(green_overlap == 255))

        state_red = False
        state_yellow = False
        state_green = False

        if float(sum_red_overlap) / float(sum_red_pix) > 0.7:
            state_red = True
        if float(sum_yellow_overlap) / float(sum_yellow_pix) > 0.7:
            state_yellow = True
        if float(sum_green_overlap) / float(sum_green_pix) > 0.7:
            state_green = True

        return state_red, state_yellow, state_green


    def apply_box_detector(self, image):
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)
        (boxes, scores, classes) = self.sess.run([self.detection_boxes, self.detection_scores, self.detection_classes],
                                                 feed_dict={self.image_tensor: image_np})
        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        return boxes, scores, classes

    def classify_tl_with_cnn(self, img):
        """Classifies a 16x16x3 image by using a CNN model

        Args:
            img (cv::Mat): 16x16x3 image containing a cropped traffic light

        Return:
            vector<int> with size (3,1), which contains the softmax output of the
                traffic light state classifier [red, yellow, green]
        """
        global tl_state_model
        global tl_state_graph

        # Resize to input size of CNN
        img = cv.resize(img, (16, 16))

        # The model needs the R and B channel swapped
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img = np.expand_dims(np.asarray(img, dtype=np.uint8), 0)

        # res = self.tl_state_model.predict(img, batch_size=1)
        # return res

        preds = [0, 0, 0]

        with tl_state_graph.as_default():
	        preds = tl_state_model.predict(img, batch_size=1)
        return preds


    def classifiy_tl_with_hough(self, img):
        # Detect traffic light countours with Hough transform
        circles = self.detect_tl_circles(img)
        # Distinguish the red, yellow and green light by sorting the w.r.t. their y coords
        red_circ, yellow_circ, green_circ = self.sort_circles_by_y(circles)

        red = yellow = green = False

        if red_circ is not None and yellow_circ is not None and green_circ is not None:
            # Apply color thresholds, to determine, which light is active
            thresh_image = self.apply_color_threshold(img)
            red, yellow, green = self.determine_active_light(thresh_image, red_circ, yellow_circ, green_circ)

        return [float(red), float(yellow), float(green)]


    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        boxes, scores, classes = self.apply_box_detector(image)

        confidence_cutoff = 0.5
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = self.filter_boxes(
            confidence_cutoff, boxes, scores, classes)

        if boxes.size > 0:
            # Get the box with the highest probability
            box = boxes[np.argmax(scores)]

            # The current box coordinates are normalized to a range between 0 and 1.
            # This converts the coordinates actual location on the image.
            height, width, _ = image.shape
            box_coords = self.to_image_coords(box, height, width)

            image = self.crop_box(image, box_coords)

            tl_state_probs = self.classify_tl_with_cnn(image)

            # check, if there is only one highest probability
            if len(np.where(tl_state_probs == np.max(tl_state_probs))[0]) == 1:
                tl_state_idx = np.argmax(tl_state_probs)

                if tl_state_idx == 0:
                    # print("RED")
                    return TrafficLight.RED
                elif tl_state_idx == 1:
                    # print("YELLOW")
                    return TrafficLight.YELLOW
                elif tl_state_idx == 2:
                    # print("GREEN")
                    return TrafficLight.GREEN
                else:
                    # print("UNKNOWN")
                    return TrafficLight.UNKNOWN

        # print("UNKNOWN - NO BOXES")
        return TrafficLight.UNKNOWN