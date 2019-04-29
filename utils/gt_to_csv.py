#!/usr/bin/python

import sys
import glob
import pandas as pd
import numpy as np
from os.path import basename
from PIL import Image
from termcolor import colored


def read_gt(gt_line):

    gt = gt_line.split()

    label = int(gt[0])
    box = gt[1:]
    box = [float(x) for x in box]

    return label, box


def to_image_coords(box, height_pix, width_pix):
    """
    Converts image coordinates from YOLO format ('center_X', 'center_Y', 'width_X', 'width_Y')
    in range [0, 1] to ['xmin', 'ymin', 'xmax', 'ymax'] in pixels.
    """

    center_X, center_Y, width_X, width_Y = box[:]
    xmin = int((center_X - 0.5 * width_X) * width_pix)
    ymin = int((center_Y - 0.5 * width_Y) * height_pix)
    xmax = int((center_X + 0.5 * width_X) * width_pix)
    ymax = int((center_Y + 0.5 * width_Y) * height_pix)

    return [xmin, ymin, xmax, ymax]

def check_bounding_box_big_enough(box, min_size=33):
    xmin, ymin, xmax, ymax = box[:]
    x_width = xmax - xmin
    y_width = ymax - ymin
    if x_width < min_size or y_width < min_size:
        return False
    else:
        return True


def txt_gt_to_csv(path):
    txt_list = []
    for txt_file_path in glob.glob(path + '/*.txt'):
        with open(txt_file_path, 'r') as f:
            file_name = basename(txt_file_path).replace(".txt", "")
            img_file_name = file_name + ".jpg"
            image_path = txt_file_path.replace(".txt", ".jpg")
            try:
                image = Image.open(image_path)
                label, box = read_gt(f.readline())
                width, height = image.size
                box_pix = to_image_coords(box, height, width)
                value = (img_file_name,
                        width,
                        height,
                        label,
                        box_pix[0],
                        box_pix[1],
                        box_pix[2],
                        box_pix[3])
                txt_list.append(value)
            except Exception as e:
                print(e)

    column_name = ['filename', 'width', 'height',
                   'class', 'xmin', 'ymin', 'xmax', 'ymax']
    txt_df = pd.DataFrame(txt_list, columns=column_name)
    return txt_df


def main():
    assert(len(sys.argv) == 3)
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    txt_df = txt_gt_to_csv(in_path)
    txt_df.to_csv(out_path, index=None)
    print('Successfully converted YOLO to csv.')


if __name__ == "__main__":
    main()
