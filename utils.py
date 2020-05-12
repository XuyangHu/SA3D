import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
from scipy.misc import imread, imresize, imsave

def check_folder(log_dir):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

def random_list(src_list, dst_list):
    data_list = list(zip(src_list, dst_list))
    random.shuffle(data_list)
    src_list, dst_list = zip(*data_list)
    return src_list, dst_list

def load_data_list(src_path, dst_path):
    src_list = []
    gt_list = []

    for pic_name in os.listdir(src_path):
        src_list.append(src_path+pic_name)
        gt_list.append(dst_path+pic_name)

    return random_list(src_list, gt_list)

def load_batch(src_list, gt_list, img_size):
    input_batch = [load_image(img_path, img_size) for img_path in src_list]
    input_batch = np.array(input_batch).astype(np.float32)
    gt_batch = [load_image(img_path, img_size) for img_path in gt_list]
    gt_batch = np.array(gt_batch).astype(np.float32)
    return input_batch, gt_batch


def load_image(img_path, img_size):
    img = imread(img_path)
    img = imresize(img, (img_size, img_size), interp='bicubic')

    img = (img/127.5)-1
    return img

def save_image(img, img_path):
    imsave(img_path, img)


def plot(samples, img_size):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(8, 8)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(img_size, img_size, 3), cmap=None)
        # plt.show()
    return fig