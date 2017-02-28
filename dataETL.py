# dataETL.py

'''

本程序主要用于进行数据的预处理。

v0.1
这个版本主要的作用是对挑选的 flikr_8k 数据集进行规范化。

'''


import csv
import os

def data_etl():
    # 文件根目录
    root_path = 'D:\\# nlp\\flikr_8k_min\\'
    pic_data = 'pic\\'

    pic_list = os.path.dirname(root_path)
    print(pic_list)

if __name__ == "__main__":
    data_etl()