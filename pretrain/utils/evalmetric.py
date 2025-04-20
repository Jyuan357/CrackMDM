import torch
import numpy as np

def get_statistics(pred, gt):
    """
    return tp, fp, fn
    """
    tp = np.sum((pred == 1) & (gt == 1))
    fp = np.sum((pred == 1) & (gt == 0))
    fn = np.sum((pred == 0) & (gt == 1))

    return [tp, fp, fn]

def cal_prf_metrics(pred_list, gt_list, thresh_step=0.01):
    final_accuracy_all = []
    for thresh in np.arange(0.0, 1.0, thresh_step):
        statistics = []

        for pred, gt in zip(pred_list, gt_list):
        
            gt_img = (gt / 255).to(torch.uint8).cpu().numpy()
            pred_img = ((pred / 255) > thresh).to(torch.uint8).cpu().numpy()
            # calculate each image
            statistics.append(get_statistics(pred_img, gt_img))

        # get tp, fp, fn
        tp = np.sum([v[0] for v in statistics])
        fp = np.sum([v[1] for v in statistics])
        fn = np.sum([v[2] for v in statistics])

        # calculate precision
        p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
        # calculate recall
        r_acc = tp / (tp + fn)
        # calculate f-score
        if p_acc == 0 and r_acc == 0:
            print("!!!!!!!!!!")
        final_accuracy_all.append([thresh, p_acc, r_acc, 2 * p_acc * r_acc / (p_acc + r_acc + 0.0001)])

    return final_accuracy_all


def cal_ois_metrics(pred_list, gt_list, thresh_step=0.01):
    final_acc_all = []
    for pred, gt in zip(pred_list, gt_list):
        statistics = []
        for thresh in np.arange(0.0, 1.0, thresh_step):
            gt_img = (gt / 255).to(torch.uint8).cpu().numpy()
            pred_img = (pred / 255 > thresh).to(torch.uint8).cpu().numpy()
            tp, fp, fn = get_statistics(pred_img, gt_img)
            p_acc = 1.0 if tp == 0 and fp == 0 else tp / (tp + fp)
            r_acc = tp / (tp + fn)

            if p_acc + r_acc == 0:
                f1 = 0
            else:
                f1 = 2 * p_acc * r_acc / (p_acc + r_acc)
            statistics.append([thresh, f1])
        max_f = np.amax(statistics, axis=0)
        final_acc_all.append(max_f[1])
    return np.mean(final_acc_all)