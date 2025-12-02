import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
tf.disable_eager_execution()

import numpy as np
import glob
import cv2
import re

from model_class import Degradation_Class

parser = argparse.ArgumentParser(description='')

parser.add_argument("--snapshot_dir", default='./weights', help="path of weights")
parser.add_argument("--fake_dir", default='./test_results/', help="path of fake_LBD outputs")
parser.add_argument("--image_size", type=int, default=128, help="load image size")
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--base_lr', type=float, default=0.001, help='initial learning rate for adam')
parser.add_argument('--epoch', dest='epoch', type=int, default=100, help='# of epoch')
parser.add_argument("--lamda", type=float, default=10.0, help="L1 lamda")
parser.add_argument('--beta1', dest='beta1', type=float, default=0.9, help='momentum term of adam')
parser.add_argument("--LBDQ_test_data_path", default='./dataset/test/LBDQ/', help="path of LBDQ training datas.")
parser.add_argument("--res_test_data_path", default='./dataset/test/res/', help="path of res training datas.")

args = parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model'
    checkpoint_path = os.path.join(logdir, model_name)
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')


_nsre = re.compile('([0-9]+)')
def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]

def TestImageReader(q_list, res_list, step):

    LBDQ_line_content = q_list[step]
    LBDQ_image = cv2.imread(LBDQ_line_content, -1)
    LBDQ_image_resize = LBDQ_image / 255
    LBDQ_image_name, _ = os.path.splitext(os.path.basename(LBDQ_line_content))

    res_line_content = res_list[step]
    res_image = cv2.imread(res_line_content, -1)
    res_image_resize = res_image / 255
    res_image_name, _ = os.path.splitext(os.path.basename(res_line_content))

    return (LBDQ_image_name, LBDQ_image_resize, res_image_name, res_image_resize)


def make_train_data_list(x_data_path):
    x_input_images = glob.glob(os.path.join(x_data_path, "*"))
    return x_input_images


def get_write_picture(x_image, fake_y):
    row = np.concatenate((x_image, fake_y), axis=1)
    return row.astype(np.uint8)


def save_test_out(LBDQ, fake, name):
    a = LBDQ[0]
    LBDQ_img = (a * 255).astype(np.uint8)
    b = LBDQ_img + (fake[0]-1)
    fake_img = np.clip(b, 0, 255)
    path = args.fake_dir + "/" + name + ".png"
    cv2.imwrite(path, fake_img)




def main():
    if not os.path.exists(args.fake_dir):
        os.makedirs(args.fake_dir)

    LBDQ_datalists = make_train_data_list(args.LBDQ_test_data_path)
    res_datalists = make_train_data_list(args.res_test_data_path)

    LBDQ_datalists.sort(key=natural_sort_key, reverse=False)
    res_datalists.sort(key=natural_sort_key, reverse=False)


    LBDQ_img = tf.placeholder(tf.float32, shape=[args.batch_size, args.image_size, args.image_size, 3], name='LBDQ_img')
    res_img = tf.placeholder(tf.float32, shape=[args.batch_size, args.image_size, args.image_size, 3], name='res_img')
    is_training = tf.placeholder(tf.bool, [])

    model = Degradation_Class(LBDQ_img, res_img, is_training, args.batch_size)

    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=1)
    checkpoint = tf.train.latest_checkpoint(args.snapshot_dir)
    saver.restore(sess, checkpoint)


    total_step = len(LBDQ_datalists)
    for step in range(total_step):
        LBDQ_image_name, LBDQ_image, res_image_name, res_image = TestImageReader(LBDQ_datalists, res_datalists, step)

        batch_LBDQ_image = np.expand_dims(np.array(LBDQ_image).astype(np.float32), axis=0)
        batch_res_image = np.expand_dims(np.array(res_image).astype(np.float32), axis=0)

        feed_dict = {LBDQ_img: batch_LBDQ_image, res_img: batch_res_image, is_training: True}

        fake = sess.run(model.category, feed_dict=feed_dict)
        save_test_out(batch_LBDQ_image, fake, LBDQ_image_name)

        print('step {:d}'.format(step))



if __name__ == '__main__':
    main()




