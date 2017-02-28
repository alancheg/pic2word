# dataETL.py

'''
@author；alancheg

本程序主要用于进行数据的预处理。
----
v0.1
这个版本主要的作用是对挑选的 flikr_8k 数据集进行规范化。
实现了以下功能：
· 将图片和对应的描述语句抽取出来。
----
'''

import csv
import os

def data_etl():
    # 文件根目录
    root_path = 'D:\\# nlp\\flikr_8k_min\\'
    pic_data = 'pic\\'
    token = 'Flickr8k.token.txt'

    # 训练数据，key 是图片的文件名称，value 是它的一句话描述
    train_dict = {}

    # 列出所有的图片文件
    filename_list = []
    for parent, dirnames, filenames in os.walk(root_path):
        for filename in filenames:
            # print(filename)
            filename_list.append(filename)

    # 找出它们对应的语句
    if os.path.exists(root_path + token):
        token_list = list(open(root_path + token))

    # 遍历，将图片和对应的描述一一对应
    for token in token_list:
        for filename in filename_list:
            if filename in token:

                # print('---- start ----')
                # print(filename)
                # print(token)
                # print('---- end ----')

                token_list.remove(token)
                filename_list.remove(filename)

                train_dict.setdefault(filename,' ')

                # token.replace(filename, ' ')
                # token.replace('#', ' ')
                train_dict[filename] = token.replace(filename, ' ').replace('#', ' ')[4:]

    for key in train_dict.keys():
        print(key)
        print(train_dict[key])

if __name__ == "__main__":
    data_etl()