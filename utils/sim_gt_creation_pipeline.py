import subprocess
import rosbag
import glob
from std_msgs.msg import Int32, String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
from PIL import Image
from tl_classifier_simple_test import TLClassifierSimple
from os.path import join
import pandas as pd


IMG_TOPIC = '/image_color'
TL_TOPIC = '/vehicle/traffic_lights'

IMG_PREFIX = "tl_"
IMG_SUFFIX = ".jpg"

EXPORT_CTR = 0
X_SIZE = Y_SIZE = 16

bridge = CvBridge()
tl_classifier = TLClassifierSimple()


def process_image_msg(msg):
    cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

    # extract traffic lights
    img_list = tl_classifier.extract_detections(cv_image)

    #resize to 16 x 16 as need for CNN input
    img_list = [cv.resize(img, (X_SIZE, Y_SIZE))
                for img in img_list if img.shape[0] >= X_SIZE and img.shape[1] >= Y_SIZE]

    return img_list


def process_bag(bag_file_path, out_path):
    global EXPORT_CTR
    tl_state_set = False
    tl_state = 0
    csv_list = []

    bag = rosbag.Bag(bag_file_path)
    for topic, msg, t in bag.read_messages(topics=[IMG_TOPIC, TL_TOPIC]):
        if topic == TL_TOPIC:
            tl_state = msg.lights[0].state
            tl_state_set = True
        elif topic == IMG_TOPIC and tl_state_set:
            img_list = process_image_msg(msg)

            # Save images
            if len(img_list) > 0:
                for img in img_list:
                    img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                    image = Image.fromarray(img)
                    export_name = IMG_PREFIX + '{:06d}'.format(EXPORT_CTR) + IMG_SUFFIX
                    image.save(join(out_path, export_name))
                    EXPORT_CTR += 1

                    # Save name and label for export to csv
                    csv_list.append((export_name, tl_state))

    bag.close()
    return csv_list


def main():
    in_path = "/home/markus/Udacity/data/own_bagfiles"
    out_path = "/home/markus/Udacity/data/own_bagfiles/images"
    # load rosbags

    column_names = ['filename', 'label']
    csv_list = []

    for bag_file_path in glob.glob(in_path + '/*.bag'):
        print(bag_file_path)
        csv_bag = process_bag(bag_file_path, out_path)
        csv_list += csv_bag

    csv_df = pd.DataFrame(csv_list, columns=column_names)
    csv_df.to_csv(join(out_path, 'labels.csv'), index=None)

if __name__ == "__main__":
    main()
