import numpy as np
import json
import os
import pickle

class PA100k():
    def __init__(self):
        self.get_PETA_train_test()

    def get_PETA_train_test(self):
        root = 'dataset/PA100k'
        data_path = os.path.join(root, 'dataset_all.pkl')

        print("which pickle", data_path)

        dataset_info = pickle.load(open(data_path, 'rb+'))

        img_id = dataset_info.image_name

        attr_label = dataset_info.label
        attr_label[attr_label == 2] = 0
        self.attr_id = dataset_info.attr_name
        self.classes_num = len(self.attr_id)

        self.eval_attr_idx = dataset_info.label_idx.eval
        self.eval_attr_num = len(self.eval_attr_idx)

        attr_label = attr_label[:, self.eval_attr_idx]
        self.attr_id = [self.attr_id[i] for i in self.eval_attr_idx]
        self.attr_num = len(self.attr_id)


        self.dataset = 'PA100K'

        self.root_path = dataset_info.root
        self.attr_num = len(self.attr_id)

        self.train_img_idx = dataset_info.partition['trainval']
        self.test_img_idx = dataset_info.partition['test']

        self.train_img_num = self.train_img_idx.shape[0]
        self.train_img_id = [img_id[i] for i in self.train_img_idx]
        self.label_train = self.train_label = attr_label[self.train_img_idx]  # [:, [0, 12]]
        self.test_img_num = self.test_img_idx.shape[0]
        self.test_img_id = [img_id[i] for i in self.test_img_idx]
        self.test_label = attr_label[self.test_img_idx]


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

        self.ac_classes =  self.attr_id





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
