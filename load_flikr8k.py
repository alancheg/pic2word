import tensorflow as tf
import os
import numpy as np
import re
import sys
import tarfile
import gzip
import zipfile
from . import visualize
from . import nlp
import pickle
from six.moves import urllib
from six.moves import cPickle
from six.moves import zip
from tensorflow.python.platform import gfile

# load flikr8k dataset


def load_flikr8k_dataset(shape=(), path="data/flikr8k/"):
    """Automatically download MNIST dataset
    and return the training, validation and test set with 50000, 10000 and 10000
    digit images respectively.

    Parameters
    ----------
    shape : tuple
        The shape of digit images, defaults to (-1,784)
    path : string
        Path to download data to, defaults to data/mnist/

    Examples
    --------
    # >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1,784))
    # >>> X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28, 28, 1))
    """
    def load_flikr8k_images(root = "data/flickr8k/"):
        # load images of flikr8k
        # filepath = maybe_download_and_extract(filename, path, 'http://yann.lecun.com/exdb/mnist/')

        # Read the inputs in Yann LeCun's binary format.

        # The inputs are vectors now, we reshape them to monochrome 2D images,
        # following the shape convention: (examples, channels, rows, columns)

        # data = data.reshape(shape)

        # The inputs come as bytes, we convert them to float32 in range [0,1].
        # (Actually to range [0, 255/256], for compatibility to the version
        # provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)

        # return data / np.float32(256)

        # 首先将 flikr8k 进行预处理，删除 _mac 文件夹
        filepath = root + str("Flickr8k_min.zip")
        print(filepath)

        """
        读取图片的主要流程是：
        1. 从压缩文件中读取图片
        2. 将图片处理成灰度图片
        """

        with gzip.open(filepath, 'rb') as f:

            data = np.frombuffer(f.read(), np.uint8, offset=16)

        data = data.reshape(shape)

        return data / np.float32(256)

