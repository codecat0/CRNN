"""
@File : train.py
@Author : CodeCat
@Time : 2021/7/15 下午10:04
"""
import os
import datetime
import math
import argparse

import torch
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset.my_dataset import SimpleDataSet
from dataset.postprocess import CTCLabelDecode
from models.crnn import CRNN
from rec_ctc_loss import CTCLoss
from rec_metric import RecMetric


def main(opt):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # Build Dataset
    trainset = SimpleDataSet(label_file_path=opt.train_label_file_path,
                             data_dir=opt.data_dir,
                             mode='train',
                             character_type=opt.character_type,
                             character_dict_path=opt.character_dict_path)
    train_dataloader = DataLoader(
        dataset=trainset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )

    validset = SimpleDataSet(label_file_path=opt.valid_label_file_path,
                             data_dir=opt.data_dir,
                             mode='valid',
                             character_type=opt.character_type,
                             character_dict_path=opt.character_dict_path)

    valid_dataloader = DataLoader(
        dataset=validset,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    # Build post process
    post_process_class = CTCLabelDecode(
        character_dict_path=opt.character_dict_path,
        use_space_char=True,
        character_type=opt.character_type
    )

    # Build crnn model
    char_num = len(post_process_class.character)

    model = CRNN(out_channels=char_num,
                 type='large',
                 hidden_size=96)

    model.to(device)
    start_epoch = 0
    if opt.weights != '':
        weight_dict = torch.load(opt.weights, map_location=device)
        model.load_state_dict(weight_dict, strict=False)

        start_epoch = int(opt.weights.split('.')[-2].split('_')[-1])

    # Build loss
    loss_class = CTCLoss()

    # Build optim
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / opt.epochs)) / 2) * (1 - opt.lrf) + opt.lrf
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # Build Metric
    eval_class = RecMetric()

    for epoch in range(start_epoch, opt.epochs):
        model.train()
        epoch_loss = 0
        for idx, batch in enumerate(train_dataloader):
            images = batch[0].to(device)
            preds = model(images)
            loss = loss_class(preds, batch)
            avg_loss = loss['loss']

            epoch_loss += avg_loss.item()
            optimizer.zero_grad()
            avg_loss.backward()
            optimizer.step()

            # print('Epoch is [{}/{}], mini-batch is [{}/{}], batch_loss is {:.8f}'.format(epoch+1, opt.epochs, idx+1, int(len(trainset) / opt.batch_size), avg_loss.item()))
        # print(optimizer.param_groups[0]['lr'])
        scheduler.step()
        print('[{}] Epoch is [{}/{}], epoch_loss is {:.8f}, lr is {:.8f}'.format(str(datetime.datetime.now())[:19], epoch+1, opt.epochs, epoch_loss/int(len(trainset)/opt.batch_size), optimizer.param_groups[0]['lr']))

        model.eval()
        with torch.no_grad():
            valid_loss = 0.0
            best_acc = 0.0
            for idx, batch in enumerate(valid_dataloader):
                images = batch[0].to(device)
                preds = model(images)
                loss = loss_class(preds, batch)
                avg_loss = loss['loss']
                valid_loss += avg_loss.item()

                batch = [item.numpy() for item in batch]
                post_result = post_process_class(preds, batch[1])
                eval_class(post_result)

            metric = eval_class.get_metric()
            print('valid losss is {:.8f}, acc_metric is {:.8f}, norm_edit_dist is {:.8f}'.format(valid_loss, metric['acc'], metric['norm_edit_dis']))
            if metric['acc'] > best_acc:
                best_acc = metric['acc']
                if not os.path.exists(opt.pths_path):
                    os.mkdir(opt.pths_path)
                torch.save(model.state_dict(), os.path.join(opt.pths_path, 'model_epoch_{}.pth'.format(epoch+1)))
                if os.path.exists(os.path.join(opt.pths_path, 'model_epoch_{}.pth'.format(start_epoch))):
                    os.remove(os.path.join(opt.pths_path, 'model_epoch_{}.pth'.format(start_epoch)))
                start_epoch = epoch + 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_label_file_path', type=str, default='./train_data/icdar2015/rec_gt_train.txt')
    parser.add_argument('--valid_label_file_path', type=str, default='./train_data/icdar2015/rec_gt_test.txt')
    parser.add_argument('--data_dir', type=str, default='./train_data/icdar2015')
    parser.add_argument('--character_dict_path', type=str, default='./ic15_dict_short.txt')
    parser.add_argument('--character_type', type=str, default='ch')
    parser.add_argument('--weights', type=str, default='')
    parser.add_argument('--pths_path', type=str, default='./weights')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=2000)

    opt = parser.parse_args()
    print(opt)
    main(opt)
