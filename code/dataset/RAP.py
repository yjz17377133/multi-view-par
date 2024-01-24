import os

import numpy as np
import scipy.io as sio
from torch.utils import data


class RAP(data.Dataset):
    def __init__(self, root='dataset/RAP'):
        self.root = root
        self.classes_num = 51
        self.save_path_partion = self.root
        self.label_size = self.classes_num
        self.get_train_test()
        self.attribute_adjust()
        self.datasetname = 'RAP'
        self.train_num = len(self.label_train)
        self.get_adj_matrix()
        self.val_att_index = [0, 1, 7, 8, 9, 10, 11, 12, 15, 17, 21, 23, 24, 25, 26, 27, 28, 29, 32, 33, 35, 36, 37, 39, 40, 41, 43, 47, 48, 50]

    def attribute_adjust(self):
        whole = [0, 1, 2, 3, 4, 5, 6, 7, 8, 43, 44, 45, 46, 47, 48, 49, 50] # 17
        hs = [9, 10, 11, 12, 13, 14] # 6
        ub = [15, 16, 17, 18, 19, 20, 21, 22, 23] # 9
        lb = [24, 25, 26, 27, 28, 29] # 6
        sh = [30, 31, 32, 33, 34] # 5
        at = [35, 36, 37, 38, 39, 40, 41, 42] # 8
        self.spatial_parts = {
            'whole': whole,
            'hs': hs,
            'ub': ub,
            'lb': lb,
            'sh': sh,
            'at': at,
        }

        self.parts_re_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                               31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 9,
                               10, 11, 12, 13, 14, 15, 16]

        low_level = [11]
        mid_level = [9, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34,
                     35, 36, 37, 38, 39, 40, 41, 42]
        high_level = [0, 1, 2, 3, 4, 5, 6, 7, 8, 43, 44, 45, 46, 47, 48, 49, 50]

        self.hierarchical = {
            'low': low_level,
            'mid': mid_level,
            'high': high_level
        }
        self.hierarchical_re_index = [34, 35, 36, 37, 38, 39, 40, 41, 42, 1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 43, 44, 45, 46, 47, 48, 49, 50]

        self.adj_part_name = ['whole', 'hs', 'ub', 'lb', 'sh', 'at']
        self.att_parts_count = [len(whole), len(at), len(hs), len(ub), len(lb), len(sh)]

        index = np.hstack((whole, at, hs, ub, lb, sh)) # **********



    def get_small_adj(self):
        # adj=None
        adj_dic = {}
        # label_train=self.label_train
        att_parts_count = self.att_parts_count
        part_name = self.adj_part_name = ['whole', 'hs', 'ub', 'lb', 'sh', 'at']
        for i in range(len(self.att_parts_count)):
            part1 = part_name[i]
            for j in range(len(self.att_parts_count)):
                part2 = part_name[j]
                key = part1 + '-' + part2
                value = self.get_adj_part(start1=int(np.sum(np.array(att_parts_count[:i]))), num1=att_parts_count[i],
                                          start2=int(np.sum(np.array(att_parts_count[:j]))), num2=att_parts_count[j])
                adj_dic[key] = value
        self.adj_dic_to_csv(adj_dic)
        return adj_dic

    def get_adj_part(self, start1, num1, start2, num2):
        label_train = self.label_train
        print(list(range(start1, start1 + num1)) + list(range(start2, start2 + num2)))
        des_label = label_train[:, list(range(start1, start1 + num1)) + list(range(start2, start2 + num2))]
        adj = np.zeros(shape=[num1, num2])
        for label in des_label:
            for i in range(num1):
                if label[i] == 1:
                    for j in range(num1, num1 + num2):
                        if label[j] == 1:
                            adj[i, j - num1] += 1
        count = np.sum(des_label, axis=0)
        size = len(count)
        count = np.reshape(count, newshape=(size, 1))
        adj = adj / count[:num1]
        adj[adj >= self.threshold] = 1
        adj[adj < self.threshold] = 0
        adj = adj.astype(np.float32)
        return adj

    def adj_dic_to_csv(self, adj_dic):
        import pandas as pd
        att_parts_count = self.att_parts_count
        for key, value in adj_dic.items():
            arr = key.split('-')
            id1 = self.adj_part_name.index(arr[0])
            id2 = self.adj_part_name.index(arr[1])
            start1 = int(np.sum(np.array(att_parts_count[:id1])))
            num1 = att_parts_count[id1]
            index = self.ac_classes[start1:start1 + num1]

            start2 = int(np.sum(np.array(att_parts_count[:id2])))
            num2 = att_parts_count[id2]
            columns = self.ac_classes[start2:start2 + num2]

            data = pd.DataFrame(value, index=index, columns=columns)
            data.to_csv('part2part/' + key + '.csv')

    def get_adj_location(self):
        adj_ori = self.get_adj_matrix()
        adj = np.zeros(shape=[1, self.classes_num, self.classes_num, len(self.adj_part_name) ** 2])
        for i in range(len(self.classes_num)):
            for j in range(len(self.classes_num)):
                channel = (i + 1) * (j + 1) - 1
                adj[0, i, j, channel] = adj_ori[i, j]
        return adj

    def get_train_test(self, partion=0):

        data_root = self.root
        data = sio.loadmat(data_root + "/RAP_annotation/RAP_annotation.mat")["RAP_annotation"]
        attributes_list = []
        self.classes_ch = [x[0][0] for x in data[0][0][2]][:51]
        for i in range(data["attribute_eng"][0][0].shape[0]):
            attributes_list.append(data["attribute_eng"][0][0][i][0][0])
        X_data = []
        y_data = []
        for i in range(41585):
            X_data.append(os.path.join(data_root + "/RAP_dataset", data['imagesname'][0][0][i][0][0]))
            y_data.append(data['label'][0][0][i])
        X_data = np.asarray(X_data)
        y_data = np.asarray(y_data)
        train_indices, test_indices = data['partion'][0][0][partion][0][0][0]
        train_indices, test_indices = list(train_indices[0] - 1), list(test_indices[0] - 1)
        self.ac_classes = attributes_list[:self.classes_num]

        self.img_train = X_data[train_indices]# [:10000]
        self.label_train = y_data[train_indices][:, :self.classes_num]#[:10000]
        self.img_val = X_data[test_indices]
        self.label_val = y_data[test_indices][:, :self.classes_num]

        return X_data[train_indices], y_data[train_indices], X_data[test_indices], y_data[test_indices], attributes_list

    def get_adj_matrix(self):
        '''
        return adj matrix
        '''
        adj_size = self.classes_num
        size = adj_size
        adj = np.zeros([adj_size, adj_size])
        d_mat = np.zeros([adj_size, adj_size])

        for item in self.label_train:
            for i in range(self.classes_num):
                if item[i] == 1:
                    for j in range(i + 1, self.classes_num):
                        if item[j] == 1:
                            adj[i, j] += 1
                            adj[j, i] += 1
        count = np.sum(self.label_train, axis=0)[:adj_size]
        self.count = count
        for i in range(adj_size):
            d_mat[i, i] = count[i]
            adj[i, i] = 0
        self.adj_ori = adj.copy()
        adj = adj / np.reshape(count, newshape=(size, 1))
        self.th_change(adj, 1, 0.8, 0.8)
        self.th_change(adj, 0.8, 0.6, 0.6)
        self.th_change(adj, 0.6, 0.4, 0.4)
        self.th_change(adj, 0.4, 0, 0)

        adj = adj * 1.0 / (adj.sum(0) + 1e-6)
        adj = adj + np.identity(size, np.float32)
        d_mat = np.power(adj.sum(1), -0.5)
        d_mat = np.diag(d_mat)
        adj = np.matmul(d_mat, adj)
        adj = np.matmul(adj, d_mat)
        self.adj = adj.astype(np.float32)
        return adj.astype(np.float32)

    def th_change(self, a, max_, min_, des):
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if a[i, j] >= min_ and a[i, j] < max_:
                    a[i, j] = des

    def adj_to_csv(self, adj_csv_path='adj.csv', adj_ori_csv_path='adj_ori.csv'):
        import pandas as pd
        adj_csv_path = self.datasetname + str(len(self.ac_classes)) + adj_csv_path
        adj_ori_csv_path = self.datasetname + str(len(self.ac_classes)) + adj_ori_csv_path

        if not os.path.exists(adj_csv_path):
            df = pd.DataFrame(self.adj, index=self.ac_classes, columns=self.ac_classes)
            df.to_csv(adj_csv_path)
        else:
            print("%s already exists" % adj_csv_path)

        if not os.path.exists(adj_ori_csv_path):
            df = pd.DataFrame(self.adj_ori, index=self.ac_classes, columns=self.ac_classes)
            df.to_csv(adj_ori_csv_path)
        else:
            print("%s already exists" % adj_ori_csv_path)

    def get_one_hot_vector(self):
        size = self.classes_num
        words = np.zeros(shape=(size, size))
        for i in range(size):
            words[i, i] = 1
        return words

    def get_glove_vector(self):
        import pickle
        vec_path = self.root + '/' + 'rap.att'
        with open(vec_path, 'rb') as f:
            dic = pickle.load(f)
        word = self.ac_classes

        for index, key in enumerate(word):
            if index == 0:
                word_vec = dic[key]
            else:
                word_vec = np.vstack((word_vec, dic[key]))
        return word_vec.astype(np.float32)


if __name__ == '__main__':
    rap = RAP()
    rap.get_train_test()
