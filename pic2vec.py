# encoding='utf-8'

"""
本程序的主要功能就是将 flickr 中的图片数据进行处理
1. 将图像的描述拆分成标签
2. 将图像进行预处理
"""

import numpy as np
from PIL import Image
import os
import jieba
import jieba.analyse
import csv
import matplotlib.pylab as plt


def gen_pic_lable(txtpath = './data/flickr8k/text.txt'):
    """
    将图片的描述文件标签化并且存储为 csv 格式
    :param txtpath: 描述文件的路径
    :return:无返回，存储一个 csv 文件
    """
    def gen_lable(sentence):
        """
        对给定的句子进行分词处理
        :param sentence:给定的句子
        :return:返回的是句子中的中文词，作为图片的标签
        """
        label_list = jieba.analyse.extract_tags(sentence, topK=5)
        # print(sentence)
        # for item in label_list:
        #     print(item)
        return label_list

    def csv_output(file_name, txt_list):
        """
        对于给定的文件名和标签，写入 csv 文件
        :param file_name: 图片名称
        :param txt_list: 描述标签
        :return:无返回，直接在指定目录生成 csv 文件
        """
        file_path = "data/flickr8k/"
        csv_name = "pic_label.csv"

        with open(file_path + csv_name, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)

            lable_str = ""
            for item in txt_list:
                lable_str = lable_str + str(item) + str(',')
            writer.writerow([file_name.replace('#0', ''), lable_str])

        # csvfile.close()

    txt_file = list(open(txtpath, 'r', encoding='utf-8'))
    # print(txt_file)
    for line in txt_file:

        print(line)
        # line = line.encode('utf-8')
        if line != '\n':
            file_name, sentence = line.strip().split('#0')

            # print(file_name)
            sentence = sentence.strip()
            # print(sentence)

            lable_list = gen_lable(sentence)
            csv_output(file_name, lable_list)


def gen_pic_vec():

    """
    产生图片向量
    :return:
    """

    def gen_gray_pic(s_path = "./data/flickr8k/Flickr8k_min/", t_path = "./data/flickr8k/Flick8k_black/"):

        # with os.path.exists(s_path):
        """
        生成黑白以及给定像素的图片
        :param s_path:图片源地址
        :param t_path:生成图片的目标地址
        :return:无返回，生成基于给定图片的指定像素黑白图像
        """

        file_list = list(os.listdir(s_path))

        for filename in file_list:
            print(filename)

            """
            图片处理：
            1. 将图片进行灰度处理
            2. 将图片化成统一像素 1000 * 1000
            """

            image_file = Image.open(s_path + str(filename), 'r')  # open colour image
            image_file = image_file.convert('1')  # convert image to black and white
            image_file.resize((1000, 1000))  # .show()
            # image_file.show()
            # image_file.save(tar_path + str(filename))

    def find_labeled_pic(lable_csv = 'data/flickr8k/pic_label.csv', pic_source_path = 'data/flickr8k/Flicker8k_Dataset/'):

        csvfile = open(lable_csv, 'r')
        reader = csv.reader(csvfile)

        for item in list(reader):
            file_name, lables = item

            if os.path.exists(pic_source_path + file_name):
                image_file = Image.open(pic_source_path + file_name, 'r')  # open colour image
                image_file = image_file.convert('1')  # convert image to black and white
                image_file.resize((1000, 1000)).show()

    find_labeled_pic()

if __name__ == "__main__":
    source_path = "./data/flickr8k/Flickr8k_min/"
    tar_path = "./data/flickr8k/Flick8k_black/"

    # gen_black_and_white_pic(s_path=source_path, t_path=tar_path)
    # sent = "一个穿粉色衣服的女孩在爬楼梯"
    # gen_lable(sentence=sent)

    # gen_pic_lable()
    gen_pic_vec()   