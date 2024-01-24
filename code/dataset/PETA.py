import numpy as np
import json


class PETA():
    def __init__(self):
        self.img_train, self.label_train, self.img_val, self.label_val, att = self.get_PETA_train_test()
        self.classes_num = 39

    def get_PETA_train_test(self):
        import pandas as pd
        data = pd.read_csv("/src/dataset/shufflenewpeta.csv", encoding='latin1')
        labels = data.iloc[:, 2:].values
        path = data.iloc[:, 1].values
        path_re = []
        ori = 'D:/dataset/pedestrian_attributes_PETA/PETA/'
        des = 'dataset/PETA/PETA_dataset/'

        for item in path:
            path_re.append(item.replace(ori, des))



        low_level = [14, 18, 32, 34]
        mid_level = [4, 5, 10, 11, 12, 13, 15, 16, 20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 33, 35, 36, 37, 38]
        high_level = [0, 1, 2, 3, 6, 7, 8, 9, 17, 19, 24]

        self.hierarchical = {
            'low': low_level,
            'mid': mid_level,
            'high': high_level
        }
        self.hierarchical_re_index = [28, 29, 30, 31, 4, 5, 32, 33, 34, 35, 6, 7, 8, 9, 0, 10, 11, 36, 1, 37, 12, 13, 14, 15, 38, 16, 17, 18, 19, 20, 21, 22, 2, 23, 3, 24, 25, 26, 27]

        whole = [0, 1, 2, 3, 19]
        hs = [10, 15, 21, 22, 33]
        ub = [7, 9, 11, 14, 24, 29, 32, 34, 36, 37, 38]
        lb = [6, 8, 12, 16, 17, 18, 28, 30, 35]
        sh = [13, 26, 27, 31]
        at = [4, 5, 20, 23, 25]

        self.spatial_parts = {
            'whole': whole,
            'hs': hs,
            'ub': ub,
            'lb': lb,
            'sh': sh,
            'at': at,
        }

        self.parts_re_index = [0, 1, 2, 3, 34, 35, 21, 10, 22, 11, 5, 12, 23, 30, 13, 6, 24, 25, 26, 4, 36, 7, 8, 37, 14, 38, 31, 32, 27, 15, 28, 33, 16, 9, 17, 29, 18, 19, 20]

        index = [2, 3, 4, 5, 7, 11, 14, 15, 18, 19, 21, 23, 24, 26, 27, 28, 29, 31, 32, 34, 35, 36, 37, 38, 40, 41, 42, 43, 45, 46, 47, 48, 50, 54, 56, 57, 58, 59, 60]

        labels = labels[:, index]
        size = 11400#13300
        train_img = path_re[:size]#[:-3000]
        train_label = labels[:size]#[:-3000]
        val_img = path_re[size:]
        val_label = labels[size:]

        return train_img, train_label, val_img, val_label, self.ac_classes



class PETA35():
    def __init__(self):
        self.img_train, self.label_train, self.img_val, self.label_val, att = self.get_PETA_train_test()
        self.classes_num = len(self.ac_classes)

    def get_PETA_train_test(self):
        import pandas as pd
        data = pd.read_csv("src/dataset/shufflenewpeta.csv", encoding='latin1')
        labels = data.iloc[:, 2:].values
        path = data.iloc[:, 1].values
        path_re = []
        ori = 'D:/dataset/pedestrian_attributes_PETA/PETA/'
        des = 'dataset/PETA/PETA_dataset/'

        for item in path:
            path_re.append(item.replace(ori, des))


        low_level = [14, 29, 31]
        mid_level = [4, 5, 10, 11, 12, 13, 15, 17, 18, 19, 20, 22, 23, 24, 25, 26, 27, 28, 30, 32, 33, 34, 35]
        high_level = [0, 1, 2, 3, 6, 7, 8, 9, 16, 21]

        self.hierarchical = {
            'low': low_level,
            'mid': mid_level,
            'high': high_level
        }
        self.hierarchical_re_index = [25, 26, 27, 28, 4, 5, 29, 30, 31, 32, 6, 7, 8, 9, 0, 10, 11, 33, 1, 34, 12, 13, 14, 15, 35, 16, 17, 18, 19, 2, 20, 3, 21, 22, 23, 24]

        whole = [0, 1, 2, 3, 19]
        hs = [10, 15, 18, 19, 30]
        ub = [7, 9, 11, 14, 21, 26, 29, 31, 33, 34, 35]
        lb = [6, 8, 12, 25, 27, 32]
        sh = [13, 23, 24, 28]
        at = [4, 5, 17, 20, 22]

        self.spatial_parts = {
            'whole': whole,
            'hs': hs,
            'ub': ub,
            'lb': lb,
            'sh': sh,
            'at': at,
        }

        self.parts_re_index = [0, 1, 2, 3, 31, 32, 18, 10, 19, 11, 5, 12, 10, 27, 13, 6, 21, 22, 23, 4, 33, 7, 8, 34, 14, 38, 28, 29, 24, 15, 25, 30, 9, 26, 16, 17]

        index = [2, 3, 4, 5, 7, 11, 14, 15, 18, 19, 21, 23, 24, 26, 27, 28, 34, 35, 36, 37, 38, 40, 41, 42, 43, 45, 46, 47, 48, 50, 54, 57, 58, 59, 60]

        labels = labels[:, index]
        size = 11400#13300
        train_img = path_re[:size]#[:-3000]
        train_label = labels[:size]#[:-3000]
        val_img = path_re[size:]
        val_label = labels[size:]

        return train_img, train_label, val_img, val_label, self.ac_classes



def save_np(name, arr):
    np.save(name, np.array(arr))
    print('save to '+ name)

if __name__ == '__main__':
    dataset_val = PETA()
    img1 = dataset_val.img_train
    img2 = dataset_val.img_val

    label1 = dataset_val.label_train
    for i in range( len( label1 )) :
        if label1[i].sum()>11:
            print(i)
    print(img1[809])
    print('hello')
