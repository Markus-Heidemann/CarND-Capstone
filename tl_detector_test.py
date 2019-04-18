import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageDraw
from PIL import ImageColor
import time
from scipy.stats import norm
import cv2 as cv

#%matplotlib inline
plt.style.use('ggplot')

SSD_GRAPH_FILE_V1 = '/home/markus/Udacity/models/ssd_mobilenet_v1_coco_2018_01_28/frozen_inference_graph.pb'
SSD_GRAPH_FILE = '/home/markus/Udacity/models/ssd_mobilenet_v1_coco_11_06_2017/frozen_inference_graph.pb'
RFCN_GRAPH_FILE = '/home/markus/Udacity/models/rfcn_resnet101_coco_11_06_2017/frozen_inference_graph.pb'
FASTER_RCNN_GRAPH_FILE = '/home/markus/Udacity/models/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017/frozen_inference_graph.pb'

# Colors (one for each class)
cmap = ImageColor.colormap
#print("Number of colors =", len(cmap))
COLOR_LIST = sorted([c for c in cmap.keys()])

#
# Utility funcs
#

def filter_boxes(min_score, boxes, scores, classes):
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

def to_image_coords(boxes, height, width):
    """
    The original box coordinate output is normalized, i.e [0, 1].

    This converts it back to the original coordinate based on the image
    size.
    """
    box_coords = np.zeros_like(boxes)
    box_coords[:, 0] = boxes[:, 0] * height
    box_coords[:, 1] = boxes[:, 1] * width
    box_coords[:, 2] = boxes[:, 2] * height
    box_coords[:, 3] = boxes[:, 3] * width

    return box_coords

def draw_boxes(image, boxes, classes, thickness=4):
    """Draw bounding boxes on the image"""
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)

def draw_boxes_np(image, boxes, classes, thickness=4):
    # image = np.squeeze(image, axis=0)
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)
    for i in range(len(boxes)):
        bot, left, top, right = boxes[i, ...]
        class_id = int(classes[i])
        color = COLOR_LIST[class_id]
        draw.line([(left, top), (left, bot), (right, bot), (right, top), (left, top)], width=thickness, fill=color)
    # plt.figure(figsize=(12, 8))
    # plt.imshow(image)
    # image = np.asarray(image)
    # image = np.expand_dims(image, 0)
    return image

def crop_box(image, box):
    bottom, left, top, right = box[...]
    return image.crop((int(left), int(bottom), int(right), int(top)))

def draw_circles(img):
    img = np.asarray(img)
    height = img.shape[0]
    img = img[:, :, ::-1].copy() # convert to BGR
    gimg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gimg = cv.medianBlur(gimg, 5)
    circles = cv.HoughCircles(gimg, cv.HOUGH_GRADIENT, 1, int(height * 0.25),
                            param1=50, param2=30, minRadius=0, maxRadius=0)
    if circles is not None:
        circles = np.uint16(np.around(circles))[0]
        assert(len(circles) == 3)
        print(circles)

        top_idx = np.argmin(circles[:,1])
        top_circ = circles[top_idx]
        circles = np.delete(circles, top_idx, axis=0)

        mid_idx = np.argmin(circles[:,1])
        mid_circ = circles[mid_idx]
        circles = np.delete(circles, mid_idx, axis=0)

        bot_circ = circles[0]

        # for i in circles[0,:]:
        #     # draw the outer circle
        #     cv.circle(img, (i[0], i[1]), i[2], (0,255,0), 2)
        #     # draw the center of the circle
        #     # cv.circle(img,(i[0],i[1]),2,(0,0,255),3)

        cv.circle(img, (top_circ[0], top_circ[1]), top_circ[2], (0,0,255), 2)
        cv.circle(img, (mid_circ[0], mid_circ[1]), mid_circ[2], (255,255,0), 2)
        cv.circle(img, (bot_circ[0], bot_circ[1]), bot_circ[2], (0,255,0), 2)

        print(top_circ)
        print(mid_circ)
        print(bot_circ)

        print()

        print(img[top_circ[1], top_circ[0], :])
        print(img[mid_circ[1], mid_circ[0], :])
        print(img[bot_circ[1], bot_circ[0], :])

    img = img[:, :, ::-1].copy() # convert back to RGB
    img = Image.fromarray(img)

    return img

def load_graph(graph_file):
    """Loads a frozen inference graph"""
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return graph

def main():
    detection_graph = load_graph(SSD_GRAPH_FILE)
    # detection_graph = load_graph(RFCN_GRAPH_FILE)
    # detection_graph = load_graph(FASTER_RCNN_GRAPH_FILE)

    # The input placeholder for the image.
    # `get_tensor_by_name` returns the Tensor with the associated name in the Graph.
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

    # The classification of the object (integer id).
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

    # Load a sample image.
    # img_path = '/home/markus/workspace/CarND-Object-Detection-Lab/assets/Traffic-Signals.jpg'
    img_path = '/home/markus/.ros/frame0000.jpg'
    image = Image.open(img_path)
    image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

    with tf.Session(graph=detection_graph) as sess:
        # Actual detection.
        (boxes, scores, classes) = sess.run([detection_boxes, detection_scores, detection_classes],
                                            feed_dict={image_tensor: image_np})

        # Remove unnecessary dimensions
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes)

        confidence_cutoff = 0.5
        # Filter boxes with a confidence score less than `confidence_cutoff`
        boxes, scores, classes = filter_boxes(confidence_cutoff, boxes, scores, classes)

        # The current box coordinates are normalized to a range between 0 and 1.
        # This converts the coordinates actual location on the image.
        width, height = image.size
        box_coords = to_image_coords(boxes, height, width)

        # Each class with be represented by a differently colored box
        # draw_boxes(image, box_coords, classes)

        # plt.figure(figsize=(12, 8))
        # plt.imshow(image)

        max_score_idx = np.argmax(scores)
        image = crop_box(image, box_coords[max_score_idx])
        image = draw_circles(image)
        image.save('./result.png')

if __name__ == '__main__':
    main()
