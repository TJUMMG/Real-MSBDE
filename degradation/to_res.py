import cv2
import numpy as np
import glob
import os
import re


def make_train_data_list(x_data_path):
    x_input_images = glob.glob(os.path.join(x_data_path, "*"))
    return x_input_images


_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def ImageReader(file_list, step):
    test_line_content = file_list[step]
    test_image_name, _ = os.path.splitext(os.path.basename(test_line_content))
    test_image = cv2.imread(test_line_content, -1)
    return test_image, test_image_name



input_path_16 = "./dataset/test/real_16bit/"
output_path_LBDQ = "./dataset/test/LBDQ/"
output_path_res = "./dataset/test/res/"



image_list_16 = make_train_data_list(input_path_16)
image_list_16.sort(key=natural_sort_key, reverse=False)


if not os.path.exists(output_path_LBDQ):
    os.makedirs(output_path_LBDQ)
if not os.path.exists(output_path_res):
    os.makedirs(output_path_res)

for step in range(len(image_list_16)):
    image_16, image_name_16 = ImageReader(image_list_16, step)
    image_LBDQ = np.floor(image_16/2**8)
    image_res = image_16-image_LBDQ*(2**8)
    image_LBDQ = image_LBDQ.astype(np.uint8)
    image_res = image_res.astype(np.uint8)

    path_LBDQ = output_path_LBDQ + image_name_16 + '.png'
    path_res = output_path_res + image_name_16 + '.png'
    cv2.imwrite(path_LBDQ, image_LBDQ)
    cv2.imwrite(path_res, image_res)
    print(step)