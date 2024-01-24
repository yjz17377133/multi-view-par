# most borrow from: https://github.com/HCPLab-SYSU/SSGRL/blob/master/utils/metrics.py

import numpy as np
import torch

def voc_ap(rec, prec, true_num):
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def voc_mAP(imagessetfilelist, num, return_each=False):
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    lines = []
    for imagessetfile in imagessetfilelist:
        with open(imagessetfile, 'r') as f:
            lines.extend(f.readlines())
    
    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    gt_label = seg[:,num:].astype(np.int32)#num of val
    num_target = np.sum(gt_label, axis=1, keepdims = True)


    sample_num = len(gt_label)
    class_num = num
    tp = np.zeros(sample_num)
    fp = np.zeros(sample_num)
    aps = []

    for class_id in range(class_num):
        confidence = seg[:,class_id]
        sorted_ind = np.argsort(-confidence)
        sorted_scores = np.sort(-confidence)
        sorted_label = [gt_label[x][class_id] for x in sorted_ind]

        for i in range(sample_num):
            tp[i] = (sorted_label[i]>0)
            fp[i] = (sorted_label[i]<=0)
        true_num = 0
        true_num = sum(tp)
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / np.maximum(float(true_num), np.finfo(np.float64).eps)
        prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = voc_ap(rec, prec, true_num)
        aps += [ap]

    np.set_printoptions(precision=6, suppress=True)
    aps = np.array(aps) * 100
    mAP = np.mean(aps)
    if return_each:
        return mAP, aps
    return mAP

def calculate_accuracy_in_detail(imagessetfilelist, num, return_each=False):
    if isinstance(imagessetfilelist, str):
        imagessetfilelist = [imagessetfilelist]
    lines = []
    for imagessetfile in imagessetfilelist:
        with open(imagessetfile, 'r') as f:
            lines.extend(f.readlines())
    
    seg = np.array([x.strip().split(' ') for x in lines]).astype(float)
    gt_result = gt_label = seg[:,num:].astype(np.int32)#num of val
    confidence = seg[:,:num]

    pt_result = (torch.sigmoid(torch.from_numpy(confidence)) > 0.5).cpu().detach().numpy()
    
    # compute the label-based accuracy
    if gt_result.shape != pt_result.shape:
        print('Shape beteen groundtruth and predicted results are different')
    # compute the label-based accuracy
    result = {}
    gt_pos = np.sum((gt_result == 1).astype(float), axis=0)
    gt_neg = np.sum((gt_result == 0).astype(float), axis=0)
    pt_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=0)
    pt_neg = np.sum((gt_result == 0).astype(float) * (pt_result == 0).astype(float), axis=0)
    label_pos_acc = 1.0 * pt_pos / (gt_pos + 1e-15)
    label_neg_acc = 1.0 * pt_neg / (gt_neg + 1e-15)
    label_acc = (label_pos_acc + label_neg_acc) / 2
    result['label_pos_acc'] = label_pos_acc
    result['label_neg_acc'] = label_neg_acc
    result['label_acc'] = label_acc
    # 1_2['label_ma'] = np.sum(label_acc) / len(label_acc)
    result['label_ma'] = np.sum(label_acc) / len(label_acc)
    
    #return result['instance_acc'], result['instance_precision'], result['instance_recall'], result['instance_F1'], \
    #       result['label_ma'], result['label_acc']
    mAP, aps = result['label_ma'], result['label_acc']
    if return_each:
        return mAP, aps
    return mAP