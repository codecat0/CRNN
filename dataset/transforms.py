"""
@File : transforms.py
@Author : CodeCat
@Time : 2021/7/15 下午8:49
"""
import math
import cv2
import numpy as np


class DecodeImage(object):
    """解码图像"""
    def __init__(self, img_mode='BGR', channel_first=False):
        self.img_mode = img_mode
        self.channel_first = channel_first

    def __call__(self, data):
        img = data['image']
        assert type(img) is bytes and len(img) > 0, "invalid input 'img' in DecodeImage"
        # 从内存数据中读取图像
        img = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(img, 1)
        if img is None:
            return None
        if self.img_mode == 'GRAY':
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif self.img_mode == 'RGB':
            assert img.shape[2] == 3, "invalid shape of image[{}]".format(img.shape)
            # RGB -> BGR
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        data['image'] = img
        return data


class BaseRecLabelEncode(object):
    """将文本标签转换为文本索引"""
    def __init__(self,
                 max_text_length=25,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False):
        self.max_text_len = max_text_length
        self.beg_str = "sos"
        self.end_str = "eos"
        if character_type == 'en':
            self.character_str = "0123456789abcdefghijklmnopqrstuvwxyz"
            dict_character = list(self.character_str)
        elif character_type == 'ch':
            self.character_str = ""
            assert character_dict_path is not None, "character_dict_path should not be None when character_type is {}".format(character_type)
            with open(character_dict_path, 'rb') as fin:
                lines = fin.readlines()
                for line in lines:
                    line = line.decode('utf-8').strip('\n').strip('\r\n')
                    self.character_str += line
            if use_space_char:
                self.character_str += " "
            dict_character = list(self.character_str)
        else:
            raise NotImplementedError
        self.character_type = character_type
        dict_character = self.add_special_char(dict_character)
        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i
        self.character = dict_character

    def add_special_char(self, dict_character):
        return dict_character

    def encode(self, text):
        """将文本标签转换为文本索引"""
        if len(text) == 0 or len(text) > self.max_text_len:
            return None
        if self.character_type == 'en':
            text = text.lower()
        text_list = []
        for char in text:
            if char not in self.dict:
                continue
            text_list.append(self.dict[char])
        if len(text_list) == 0:
            return None
        return text_list


class CTCLabelEncode(BaseRecLabelEncode):
    def __init__(self,
                 max_text_length=25,
                 character_dict_path=None,
                 character_type='ch',
                 use_space_char=False,
                 **kwargs):
        super(CTCLabelEncode, self).__init__(max_text_length, character_dict_path, character_type, use_space_char)

    def __call__(self, data):
        text = data['label']
        text = self.encode(text)
        if text is None:
            return None
        data['length'] = np.array(len(text))
        text = text + [0] * (self.max_text_len - len(text))
        data['label'] = np.array(text)
        return data

    def add_special_char(self, dict_character):
        dict_character = ['blank'] + dict_character
        return dict_character


class RecResizeImg(object):
    """将图像调整到(32, 100)"""
    def __init__(self,
                 image_shape=(3, 32, 100),
                 infer_mode=False,
                 character_type='ch'):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_type = character_type

    def __call__(self, data):
        img = data['image']
        if self.infer_mode and self.character_type == 'ch':
            norm_img = resize_norm_img_chinese(img, self.image_shape)
        else:
            norm_img = resize_norm_img(img, self.image_shape)
        data['image'] = norm_img
        return data


class KeepKeys(object):
    """将字典转换成列表"""
    def __init__(self, keep_keys=('image', 'label', 'length')):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


def resize_norm_img_chinese(img, image_shape):
    # [3， 32， 100]
    imgC, imgH, imgW = image_shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h
    max_wh_ratio = max(max_wh_ratio, ratio)
    imgW = int(32 * max_wh_ratio)
    # 将图像的高度设置为32，保证图像比例不变，调整图像的宽度
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))

    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    # image_shape -> [1, c, h, w]
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[None, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im


def resize_norm_img(img, image_shape):
    imgC, imgH, imgW = image_shape
    h, w = img.shape[0], img.shape[1]
    ratio = w * 1.0 / h

    # 将图像的高度设置为32，保证图像比例不变，调整图像的宽度且宽度不超过100
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))

    resized_image = cv2.resize(img, (resized_w, imgH))
    resized_image = resized_image.astype('float32')
    # image_shape -> [1, c, h, w]
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im[:, :, 0:resized_w] = resized_image
    return padding_im