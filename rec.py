"""
@File : rec.py
@Author : CodeCat
@Time : 2021/7/17 上午9:13
"""
import argparse

import torch
from PIL import Image

import matplotlib.pyplot as plt

from dataset.my_dataset import SimpleDataSet
from dataset.postprocess import CTCLabelDecode

from models.crnn import CRNN




def main(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    img = Image.open(opt.pic_path)
    plt.imshow(img)

    post_process_class = CTCLabelDecode(
        character_dict_path=opt.character_dict_path,
        use_space_char=True
    )
    char_num = len(post_process_class.character)
    model = CRNN(out_channels=char_num,
                 type='large',
                 hidden_size=96).to(device)
    model.load_state_dict(torch.load(opt.model_path))

    dataset = SimpleDataSet(
        label_file_path=opt.label_file_path,
        data_dir='./test',
        mode='valid',
        character_dict_path=opt.character_dict_path
    )

    data = dataset[0]
    model.eval()
    preds = model(torch.from_numpy(data[0]).unsqueeze(0).to(device))
    label = torch.from_numpy(data[1]).unsqueeze(0).to(device)
    post_result = post_process_class(preds.cpu(), label)
    print(post_result)

    pred_label = post_result[0][0][0]
    pred_prob = post_result[0][0][1]
    title = "pred: {}, prob: {:.4f}".format(pred_label, pred_prob)
    plt.title(title)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--character_dict_path', type=str, default='./ic15_dict_short.txt')
    parser.add_argument('--model_path', type=str, default='./weights/model_epoch_369.pth')
    parser.add_argument('--label_file_path', type=str, default='./test/gt.txt')
    parser.add_argument('--pic_path', type=str, default='./test/pic/word.png')

    opt = parser.parse_args()
    print(opt)
    main(opt)





