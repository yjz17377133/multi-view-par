import numpy as np


def calculate_accuracy_lsl(y_true, y_pred):
    mA = lsl_mA(y_pred, y_true)
    acc = lsl_acc(y_pred, y_true)
    recall = lsl_rec(y_pred, y_true)
    prec = lsl_prec(y_pred, y_true)
    return acc, prec, recall, mA


def lsl_mA(y_pred, y_true):
    M = len(y_pred)
    L = len(y_pred[0])
    res = 0
    for i in range(L):
        P = sum(y_true[:, i])
        N = M - P
        TP = sum(y_pred[:, i] * y_true[:, i])
        TN = list(y_pred[:, i] + y_true[:, i] == 0).count(True)
        # print(P,',', N,',', TP,',', TN)
        if P != 0:
            res += TP / P + TN / N
        else:
            res += TN / N
    return res / (2 * L)


def lsl_acc(y_pred, y_true):
    M = len(y_pred)
    M_ = 0
    res = 0
    for i in range(M):
        # print(np.shape(y_pred[i]*y_true[i]))
        if sum(y_pred[i]) + sum(y_true[i]) - sum(y_pred[i] * y_true[i]) != 0:
            res += sum(y_pred[i] * y_true[i]) / (sum(y_pred[i]) + sum(y_true[i]) - sum(y_pred[i] * y_true[i]))
            M_ += 1
    return res / M_


def lsl_prec(y_pred, y_true):
    M = len(y_pred)
    M_ = 0
    res = 0
    for i in range(M):
        if sum(y_pred[i]) != 0:
            res += sum(y_pred[i] * y_true[i]) / sum(y_pred[i])
            M_ += 1
    if M_ == 0:
        return 0
    return res / M_



def mA_F(y_true, y_pred):
    """
    y_pred_np = K.eval(y_pred)
    y_true_np = K.eval(y_true)
    M = len(y_pred_np)
    L = len(y_pred_np[0])
    res = 0
    for i in range(L):
        P = sum(y_true_np[:, i])
        N = M - P
        TP = sum(y_pred_np[:, i]*y_true_np[:, i])
        TN = list(y_pred_np[:, i]+y_true_np[:, i] == 0.).count(True)
        #print(TP, P, TN, N)
        #print(P,',', N,',', TP,',', TN)
        #if P != 0:
        res += TP/P + TN/N
    return res / (2*L)
    """
    # print(K.int_shape(y_true))
    x = 1e-7
    P = np.sum(y_true, axis=-1) + x
    # print("P", P)
    N = np.sum(1 - y_true, axis=-1) + x
    # print("N", N)
    TP = np.sum(y_pred * y_true, axis=-1)
    # print("TP", TP)
    TN = np.sum(y_pred + y_true == 0, axis=-1)
    # print("TN", TN)
    return np.mean(TP / P + TN / N) / 2


def lsl_rec(y_pred, y_true):
    M = len(y_pred)
    M_ = 0
    res = 0
    for i in range(M):
        if sum(y_true[i]) != 0:
            res += sum(y_pred[i] * y_true[i]) / sum(y_true[i])
            M_ += 1
    if M_ == 0:
        return 0
    return res / M_


def calculate_accuracy(gt_result, pt_result, classes=None, info=''):
    '''  '''
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
    if classes is not None:
        data = pd.DataFrame(columns=classes, data=label_acc)
        data.to_csv(info + 'res.csv')

    equ = np.equal(gt_result, pt_result)
    acc = np.mean(equ, axis=0)
    acc = acc.mean()
    # compute the instance-based accuracy
    # precision
    gt_pos = np.sum((gt_result == 1).astype(float), axis=1)
    pt_pos = np.sum((pt_result == 1).astype(float), axis=1)
    floatersect_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=1)
    union_pos = np.sum(((gt_result == 1) + (pt_result == 1)).astype(float), axis=1)
    # avoid empty label in predicted results
    cnt_eff = float(gt_result.shape[0])
    for iter, key in enumerate(gt_pos):
        if key == 0:
            union_pos[iter] = 1
            pt_pos[iter] = 1
            gt_pos[iter] = 1
            cnt_eff = cnt_eff - 1
            continue
        if pt_pos[iter] == 0:
            pt_pos[iter] = 1
    instance_acc = np.sum(floatersect_pos / union_pos) / cnt_eff
    instance_precision = np.sum(floatersect_pos / pt_pos) / cnt_eff
    instance_recall = np.sum(floatersect_pos / gt_pos) / cnt_eff
    floatance_F1 = 2 * instance_precision * instance_recall / (instance_precision + instance_recall)
    result['instance_acc'] = instance_acc
    result['instance_precision'] = instance_precision
    result['instance_recall'] = instance_recall
    result['instance_F1'] = floatance_F1
    return result['instance_acc'], result['instance_precision'], result['instance_recall'], result['instance_F1'], \
           result['label_ma']


def calculate_accuracy_in_detail(gt_result, pt_result, classes=None, info=''):
    '''  '''
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
    if classes is not None:
        data = pd.DataFrame(columns=classes, data=label_acc)
        data.to_csv(info + 'res.csv')

    equ = np.equal(gt_result, pt_result)
    acc = np.mean(equ, axis=0)
    acc = acc.mean()
    # compute the instance-based accuracy
    # precision
    gt_pos = np.sum((gt_result == 1).astype(float), axis=1)
    pt_pos = np.sum((pt_result == 1).astype(float), axis=1)
    floatersect_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=1)
    union_pos = np.sum(((gt_result == 1) + (pt_result == 1)).astype(float), axis=1)
    # avoid empty label in predicted results
    cnt_eff = float(gt_result.shape[0])
    for iter, key in enumerate(gt_pos):
        if key == 0:
            union_pos[iter] = 1
            pt_pos[iter] = 1
            gt_pos[iter] = 1
            cnt_eff = cnt_eff - 1
            continue
        if pt_pos[iter] == 0:
            pt_pos[iter] = 1
    instance_acc = np.sum(floatersect_pos / union_pos) / cnt_eff
    instance_precision = np.sum(floatersect_pos / pt_pos) / cnt_eff
    instance_recall = np.sum(floatersect_pos / gt_pos) / cnt_eff
    floatance_F1 = 2 * instance_precision * instance_recall / (instance_precision + instance_recall)
    result['instance_acc'] = instance_acc
    result['instance_precision'] = instance_precision
    result['instance_recall'] = instance_recall
    result['instance_F1'] = floatance_F1
    return result['instance_acc'], result['instance_precision'], result['instance_recall'], result['instance_F1'], \
           result['label_ma'], result['label_acc']
           

def calculate_accuracy_in_detail_with_Focus(gt_result, pt_result, classes=None, info=''):
    '''  '''
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
    if classes is not None:
        data = pd.DataFrame(columns=classes, data=label_acc)
        data.to_csv(info + 'res.csv')

    equ = np.equal(gt_result, pt_result)
    acc = np.mean(equ, axis=0)
    acc = acc.mean()
    # compute the instance-based accuracy
    # precision
    gt_pos = np.sum((gt_result == 1).astype(float), axis=1)
    pt_pos = np.sum((pt_result == 1).astype(float), axis=1)
    floatersect_pos = np.sum((gt_result == 1).astype(float) * (pt_result == 1).astype(float), axis=1)
    union_pos = np.sum(((gt_result == 1) + (pt_result == 1)).astype(float), axis=1)
    # avoid empty label in predicted results
    cnt_eff = float(gt_result.shape[0])
    for iter, key in enumerate(gt_pos):
        if key == 0:
            union_pos[iter] = 1
            pt_pos[iter] = 1
            gt_pos[iter] = 1
            cnt_eff = cnt_eff - 1
            continue
        if pt_pos[iter] == 0:
            pt_pos[iter] = 1
    instance_acc = np.sum(floatersect_pos / union_pos) / cnt_eff
    instance_precision = np.sum(floatersect_pos / pt_pos) / cnt_eff
    instance_recall = np.sum(floatersect_pos / gt_pos) / cnt_eff
    floatance_F1 = 2 * instance_precision * instance_recall / (instance_precision + instance_recall)
    result['instance_acc'] = instance_acc
    result['instance_precision'] = instance_precision
    result['instance_recall'] = instance_recall
    result['instance_F1'] = floatance_F1
    return result['instance_acc'], result['instance_precision'], result['instance_recall'], result['instance_F1'], \
           result['label_ma'], result['label_acc'], result['label_pos_acc'], result['label_neg_acc']


def mA_1(gt_result, pt_result):
    '''  '''
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
    return label_acc


def make_compare(pre_a, pre_b):
    thre_high = 0.5
    thre_low = 0.3
    s = (pre_a + pre_b) / 2.
    s[((pre_a > thre_high).astype(np.float32) + (pre_b > thre_high).astype(np.float32)) > 0] = 1
    s[((pre_a < thre_low).astype(np.float32) + (pre_b < thre_low).astype(np.float32)) > 0] = 0
    s[s >= 0.5] = 1
    s[s < 0.5] = 0
    return s


def make_compare3(pre_a, pre_b, pre_c):
    thre_high = 0.5
    thre_low = 0.1
    s = (pre_a + pre_b + pre_c) / 3.

    s[((pre_a > thre_high).astype(np.float32) + (pre_b > thre_high).astype(np.float32) + (pre_c > thre_high).astype(
        np.float32)) > 0] = 1
    s[((pre_a < thre_low).astype(np.float32) + (pre_b < thre_low).astype(np.float32) + (pre_c < thre_low).astype(
        np.float32)) > 0] = 0
    s[s >= 0.5] = 1
    s[s < 0.5] = 0
    return s


def vote(pre_a, pre_b, pre_c):
    s = (pre_a > 0.5).astype(np.int)
    b = (pre_b > 0.5).astype(np.int)
    g = (pre_c > 0.5).astype(np.int)

    return (s + b + g >= 2).astype(np.int)


def calcute2(gt_result, pre_a, pre_b):
    pt_result = make_compare(pre_a, pre_b)
    return calculate_accuracy(gt_result, pt_result)


def calcute3(gt_result, pre_a, pre_b, pre_c):
    pt_result = make_compare3(pre_a, pre_b, pre_c)
    return calculate_accuracy(gt_result, pt_result)


def calcute1(gt_result, pre):
    pt_result = (pre > 0.5).astype(np.int)
    return calculate_accuracy(gt_result, pt_result)


def save_to_csv_single(label, pre, path, dat='RAP'):
    import pandas as pd
    names = ['acc', 'pre', 'recall', 'F1', 'mA']
    inds = calcute1(label, pre)
    data = pd.DataFrame(inds, index=names)
    data.to_csv(path + '_5.csv')
    print(data)
    if dat == 'PETA':
        from dataset import PETA
        data = PETA.PETA()
    elif dat == 'RAP':
        from dataset import RAP
        data = RAP.RAP('../../data/RAP')
    names = data.ac_classes
    pre = pre >= 0.5
    mAs = mA_1(label, pre)
    data = pd.DataFrame(mAs, index=names)
    data.to_csv(path + '_mA.csv')
    print(data)


def save_to_csv_two(label, pre1, pre2, path):
    import pandas as pd
    names = ['acc', 'pre', 'recall', 'F1', 'mA']
    inds = calcute2(label, pre1, pre2)
    data = pd.DataFrame(inds, index=names)
    data.to_csv(path + '_5.csv')
    print(data)
    from dataset import PETA
    peta = PETA.PETA()
    names = peta.ac_classes
    pre = make_compare(pre1, pre2)
    mAs = mA_1(label, pre)
    data = pd.DataFrame(mAs, index=names)
    data.to_csv(path + '_mA.csv')
    print(data)


def save_to_csv_three(label, pre1, pre2, pre3, path, dat='RAP'):
    import pandas as pd
    names = ['acc', 'pre', 'recall', 'F1', 'mA']
    inds = calcute3(label, pre1, pre2, pre3)
    data = pd.DataFrame(inds, index=names)
    data.to_csv(path + '_5.csv')
    print(data)
    if dat == 'RAP':
        from dataset import RAP
        data = RAP.RAP('../../data/RAP')
    elif dat == 'PETA':
        from dataset import PETA
        data = PETA.PETA()
    names = data.ac_classes
    pre = make_compare3(pre1, pre2, pre3)
    mAs = mA_1(label, pre)
    data = pd.DataFrame(mAs, index=names)
    data.to_csv(path + '_mA.csv')
    print(data)


def make_pre_result(img, pre, save_path, dat='RAP'):
    import pandas as pd
    if dat == 'RAP':
        from dataset import RAP
        data = RAP.RAP('../../data/RAP')
    elif dat == 'PETA':
        from dataset import PETA
        data = PETA.PETA()
    names = data.ac_classes
    pre = (pre > 0.5).astype(np.int32)
    df = pd.DataFrame(pre, index=img, columns=names)
    df.to_csv(save_path + '.csv')

def save_res_for_img():
    from dataset import RAP

    data = RAP.RAP('../../data/RAP')
    names = data.ac_classes
    label = data.label_val
    img = data.img_val
    make_pre_result(img[:-1], label[:-1], 'rap_label')

def RapTest():
    from dataset import RAP
    data = RAP.RAP('../../data/RAP')
    names = data.ac_classes
    label = data.label_val
    pre = np.load("/data2/fanhn/AttributeReg/saved_module/2020_10_17/ori_1017/Resnet101_com__SubNet_0.6861896243910388_pre.npy")#("/data3/fanhn/SpatialAttributeReg/saved_module/2020_10_27/rap_1026_2_relation_test/sum_16__0.8522894148521789_pre.npy")#("/data3/fanhn/SpatialAttributeReg/saved_module/2020_10_27/rap_10262_relation_test/sum_16__0.8522894148521789_pre.npy")#("/data2/fanhn/model/2020_10_18/test2/Resnet101_all__SubNet_0.7806241277412591_pre.npy")#("/data2/fanhn/model/2020_10_18/test2/Resnet101_all__SubNet_0.7379300279419815_pre.npy")#("/data2/fanhn/model/2020_10_18/test2/Resnet101_all__SubNet_0.7094619604071273_pre.npy") # ("/data2/fanhn/AttributeReg/saved_module/2020_10_17/ori_1017/Resnet101_com__SubNet_0.7157857861614748_pre.npy")#("/data2/fanhn/AttributeReg/saved_module/2020_10_17/all_1017/Resnet101_all__SubNet_0.8418062866146587_pre.npy")# ('/home/fanhn/code/fanhn/SpatialAttributeReg/saved_module/2020_10_15/1015_relation/sum_10__0.8662297497119881_pre.npy')#("/data3/fanhn/AttributeReg/saved_module/2019_11_10/hierarchical_test/Resnet101_all_6___0.8470704194820273_pre.npy")
        #'/data2/fanhn/AttributeReg/sa(ved_module/2020_09_09/mark5/Resnet101_hie_4_SubNet_0.8460675599375188_pre.npy')
    pre = pre >= 0.5
    a = mA_1(label[:-1], pre)

    arr = []
    for i in range(len(a)):
        print(names[i], a[i])
        arr.append([names[i], a[i]])
    print(sum(a)/len(a))
    import pandas as pd
    # pd.DataFrame(np.array(arr)).to_csv('/home/fanhn/haha/8522.csv')

def PetaTest():
    from dataset import PETA
    data = PETA.PETA()
    names = data.ac_classes
    label = data.label_val

    pre = np.load(
        "/data3/fanhn/SpatialAttributeReg/saved_module/2020_10_25/peta_1025_relation_test/x_4_10__0.8809985471591346_pre.npy")  # ('/home/fanhn/code/fanhn/SpatialAttributeReg/saved_module/2020_10_18/peta_1015_relation_test/x_0_6__0.8843948035187944_pre.npy')#("/data3/fanhn/AttributeReg/saved_module/2019_11_24/peta_spa_hiera/Resnet101_spa_hiera_8___0.8709098598152715b_pre.npy")#("/data2/fanhn/AttributeReg/saved_module/2020_09_10/peta_mark5/Resnet101_hie_29_SubNet_0.8549986377304231_pre.npy")
    pre = pre >= 0.5
    a = mA_1(label, pre)
    arr = []
    for i in range(len(a)):
        print(names[i], a[i])
        arr.append([names[i], a[i]])
    print(sum(a) / len(a))
    import pandas as pd
    pd.DataFrame(np.array(arr)).to_csv('/home/fanhn/result/peta_1025.csv')

if __name__ == '__main__':
    # label = np.load('label.npy')
    # pre = np.load("/data3/fanhn/AttributeReg/saved_module/2019_11_06/ori/Resnet101_main_32___0.7564465432723106_pre.npy")

    # peta 1+3 "/data3/fanhn/AttributeReg/saved_module/2019_11_13/PETA_all_spa_1/Resnet101_all_spa_34___0.958323242715044a_pre.npy"
    # peta 1+3 "/data3/fanhn/AttributeReg/saved_module/2019_11_13/PETA_all_spa_1/Resnet101_all_spa_34___0.958323242715044b_pre.npy"

    # pre_all = np.load("/data3/fanhn/AttributeReg/saved_module/2019_11_13/PETA_all_spa_1/Resnet101_all_spa_34___0.958323242715044a_pre.npy")
    # pre_spa = np.load("/data3/fanhn/AttributeReg/saved_module/2019_11_13/PETA_all_spa_1/Resnet101_all_spa_34___0.958323242715044b_pre.npy")

    # peta: all "/data2/fanhn/AttributeReg/saved_module/2019_12_09/all_peta/Resnet101_com_19___0.8565622847761819_pre.npy"
    # peta: spa "/data3/fanhn/AttributeReg/saved_module/2019_11_24/peta_spa/Resnet101_com_7___0.8621072628694828_pre.npy"
    # peta : hiera "/data3/fanhn/AttributeReg/saved_module/2019_11_24/peta_hiera/Resnet101_com_10___0.861239458801259_pre.npy"
    # peta: all+spa "/data3/fanhn/AttributeReg/saved_module/2019_11_24/peta_all_spa/Resnet101_all_spa_12___0.86588364697179b_pre.npy"
    # peta : all+hiera "/data3/fanhn/AttributeReg/saved_module/2019_11_24/peta_all_hiera/Resnet101_all_hiera_12___0.8636401507296481b_pre.npy"
    # peta : spa+hiera "/data3/fanhn/AttributeReg/saved_module/2019_11_24/peta_spa_hiera/Resnet101_spa_hiera_11___0.8689786735985424b_pre.npy"
    # peta : all+spa+hiera "/data2/fanhn/AttributeReg/saved_module/2019_12_09/all_spa_hiera_peta/Resnet101_all_spa_hiera_10___0.8708037830867306c_pre.npy"
    from dataset import PETA
    # peta: all + hiera
    # pre1 = np.load("/data3/fanhn/AttributeReg/saved_module/2019_11_24/peta_all_hiera/Resnet101_all_hiera_12___0.8636401507296481b_pre.npy")
    # pre2 = np.load("/data3/fanhn/AttributeReg/saved_module/2019_11_24/peta_all_hiera/Resnet101_all_hiera_12___0.8636401507296481a_pre.npy")
    # peta : all + spa
    # pre1 = np.load("/data3/fanhn/AttributeReg/saved_module/2019_11_24/peta_all_spa/Resnet101_all_spa_12___0.86588364697179a_pre.npy")
    # pre2 = np.load("/data3/fanhn/AttributeReg/saved_module/2019_11_24/peta_all_spa/Resnet101_all_spa_12___0.86588364697179b_pre.npy")

    # peta : spa + hiera
    # pre1 = np.load("/data3/fanhn/AttributeReg/saved_module/2019_11_24/peta_spa_hiera/Resnet101_spa_hiera_11___0.8689786735985424a_pre.npy")
    # pre2 = np.load("/data3/fanhn/AttributeReg/saved_module/2019_11_24/peta_spa_hiera/Resnet101_spa_hiera_11___0.8689786735985424b_pre.npy")

    # peta: all+spa+hiera

    # pre1 = np.load("/data2/fanhn/AttributeReg/saved_module/2019_12_09/all_spa_hiera_peta/Resnet101_all_spa_hiera_10___0.8708037830867306a_pre.npy")
    # pre2 = np.load("/data2/fanhn/AttributeReg/saved_module/2019_12_09/all_spa_hiera_peta/Resnet101_all_spa_hiera_10___0.8708037830867306b_pre.npy")
    # pre3 = np.load("/data2/fanhn/AttributeReg/saved_module/2019_12_09/all_spa_hiera_peta/Resnet101_all_spa_hiera_10___0.8708037830867306c_pre.npy")
    # peta = PETA.PETA()
    # label = peta.label_val
    # print(calcute2(label, pre2, pre1))"/data2/fanhn/AttributeReg/saved_module/2019_12_09/all_spa_hiera_peta/Resnet101_all_spa_hiera_10___0.8708037830867306b_pre.npy"
    # save_to_csv_two(label,pre2,pre1,'test_spa_hiera')
    # save_to_csv_three(label,pre1, pre2,pre3, path='test_all_spa_hiera')
    # save_to_csv_single(label, pre1, 'test_all_hiera')
    # BCE : "/data2/fanhn/AttributeReg/saved_module/2020_01_04/BCE/Resnet101_main_11___0.7544132260829667_pre.npy"

    # pre = np.load("/data2/fanhn/AttributeReg/saved_module/2020_01_04/BCE/Resnet101_main_11___0.7544132260829667_pre.npy")
    # from dataset import RAP
    # data = RAP.RAP('../../data/RAP')
    # names = data.ac_classes
    # label = data.label_val
    # img = data.img_val
    # make_pre_result(img[:-1], pre, 'rap_val_result')
    # save_to_csv_single(label[:-1,:],pre,'nihaoa')

    # #pre = make_compare(pre_all, pre_spa)
    # pre = calcute1(label, pre1)
    # s = mA_1(label, pre)
    #
    # classes = peta.ac_classes
    # import pandas as pd
    # data = pd.DataFrame(s , index = classes)
    # data.to_csv('peta_attribtues2.csv')
    # print(s)
    # pre_hirea = np.load('/home/fanhn/code/AttributeReg/src/utils/1_2_3/hiera_pre.npy')
    # pre_nmb = np.load("/home/fanhn/code/AttributeReg/src/utils/1_2_3/spa_pre.npy")
    # pre = make_compare(pre_all, pre_hirea)
    # pre = vote(pre_all, pre_hirea, pre_nmb)
    # pre = make_compare3(pre_all, pre_hirea, pre_nmb)
    # s = calculate_accuracy(label, pre)
    # s = mA_1(label, pre_all)
    # import pandas as pd
    # from dataset import RAP
    # rap = RAP.RAP('/home/fanhn/code/AttributeReg/data/RAP')
    # classes = rap.ac_classes
    # data = pd.DataFrame(s, index=classes)
    # data.to_csv('3.csv')
    # print(s)
    # pre_all = np.load('/home/fanhn/code/AttributeReg/src/utils/1_2_3/all_pre.npy .npy')
    # pre_spa = np.load('/home/fanhn/code/AttributeReg/src/utils/1_2_3/spa_pre.npy')
    # pre_hiera = np.load('/home/fanhn/code/AttributeReg/src/utils/1_2_3/hiera_pre.npy')

    # # save_to_csv_three(label[:-1],pre_all,pre_hiera,pre_spa,'rap_three')
    # pre = make_compare3(pre_all, pre_hiera, pre_spa)
    # make_pre_result(img[:-1],pre,'rap_my')


    RapTest()