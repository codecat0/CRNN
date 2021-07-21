"""
@File : my_dataset.py
@Author : CodeCat
@Time : 2021/7/15 下午8:48
"""
import os
import numpy as np
from torch.utils.data import Dataset

from .transforms import DecodeImage, CTCLabelEncode, RecResizeImg, KeepKeys



class SimpleDataSet(Dataset):
    def __init__(self, label_file_path, data_dir='./train_data/icdar2015', delimiter='\t', mode='train', character_type='ch', character_dict_path=''):
        # 获取标签文件路径
        self.label_file_path = label_file_path

        # 获取数据路径
        self.data_dir = data_dir

        # 分割符
        self.delimiter = delimiter

        self.mode = mode

        self.data = self.get_image_info_list()
        self.data_idx_order_list = list(range(len(self.data)))

        self.decode_img = DecodeImage()
        self.ctc_label_encoder = CTCLabelEncode(character_dict_path=character_dict_path, character_type=character_type)
        self.resize_img = RecResizeImg(character_type=character_type)
        self.keep_keys = KeepKeys()

    def get_image_info_list(self, ):
        with open(self.label_file_path, 'rb') as f:
            data = f.readlines()
        return data

    def transform(self, data):
        data = self.decode_img(data)
        if data is None:
            return None
        data = self.ctc_label_encoder(data)
        if data is None:
            return None
        data = self.resize_img(data)
        if data is None:
            return None
        data = self.keep_keys(data)
        if data is None:
            return None
        return data

    def __getitem__(self, item):
        file_idx = self.data_idx_order_list[item]
        data_file = self.data[file_idx]
        try:
            data_line = data_file.decode('utf-8')
            substr = data_line.strip('\n').split(self.delimiter)
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(self.data_dir, file_name)
            data = {'img_path': img_path, 'label': label}
            if not os.path.exists(img_path):
                raise Exception("{} does not exist!".format(img_path))
            with open(data['img_path'], 'rb') as f:
                img = f.read()
                data['image'] = img
            outs = self.transform(data)
        except Exception as e:
            print("When parsing line {}, error happened with msg: {}".format(data_line, e))
            outs = None


        if outs is None:
            rnd_idx = np.random.randint(self.__len__()) if self.mode == 'train' else (item + 1) % self.__len__()
            return self.__getitem__(rnd_idx)
        return outs

    def __len__(self):
        return len(self.data_idx_order_list)




