"""
@File : rec_metric.py
@Author : CodeCat
@Time : 2021/7/16 上午9:19
"""
import Levenshtein


class RecMetric(object):
    def __init__(self, main_indicator='acc'):
        self.main_indicator = main_indicator
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0


    def __call__(self, pred_label):
        preds, labels = pred_label
        correct_num = 0
        all_num = 0
        norm_edit_dis = 0
        for (pred, pred_conf), (target, _) in zip(preds, labels):
            pred = pred.replace(" ", "")
            target = target.replace(" ", "")
            norm_edit_dis += Levenshtein.distance(pred, target) / max(len(pred), len(target), 1)
            if pred == target:
                correct_num += 1
            all_num += 1
        self.correct_num += correct_num
        self.all_num += all_num
        self.norm_edit_dis += norm_edit_dis
        return {
            'acc': correct_num / all_num,
            'norm_edit_dis': 1 - norm_edit_dis / all_num
        }

    def get_metric(self):
        acc = 1.0 * self.correct_num / self.all_num
        norm_edit_dis = 1 - self.norm_edit_dis / self.all_num
        self.reset()
        return {
            'acc': acc,
            'norm_edit_dis': norm_edit_dis
        }

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0