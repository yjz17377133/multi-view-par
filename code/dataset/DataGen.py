import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T

from dataset.PETA import PETA
from dataset.RAP import RAP
from dataset.PA100K import PA100k
from dataset.UPAR import UPAR
from math import pi
import cv2
import numpy as np
from random import random

import glob
import os
import pickle

from PIL import Image



def sample_homography_np(
        shape, shift=0, perspective=True, scaling=True, rotation=True, translation=True,
        n_scales=2, n_angles=25, scaling_amplitude=0.1, perspective_amplitude_x=0.1,
        perspective_amplitude_y=0.1, patch_ratio=0.8, max_angle=pi/6,
        allow_artifacts=False, translation_overflow=0.):

    pts1 = np.stack([[0., 0.], [0., 1.], [1., 1.], [1., 0.]], axis=0)
    # Corners of the input patch
    margin = (1 - patch_ratio) / 2
    pts2 = margin + np.array([[0, 0], [0, patch_ratio],
                                 [patch_ratio, patch_ratio], [patch_ratio, 0]])

    from numpy.random import normal
    from numpy.random import uniform
    from scipy.stats import truncnorm

    std_trunc = 2

    if perspective:
        if not allow_artifacts:
            perspective_amplitude_x = min(perspective_amplitude_x, margin)
            perspective_amplitude_y = min(perspective_amplitude_y, margin)

        perspective_displacement = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_y/2).rvs(1)

        h_displacement_left = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)

        h_displacement_right = truncnorm(-1*std_trunc, std_trunc, loc=0, scale=perspective_amplitude_x/2).rvs(1)
        pts2 += np.array([[h_displacement_left, perspective_displacement],
                          [h_displacement_left, -perspective_displacement],
                          [h_displacement_right, perspective_displacement],
                          [h_displacement_right, -perspective_displacement]]).squeeze()

    if scaling:
        scales = truncnorm(-1*std_trunc, std_trunc, loc=1, scale=scaling_amplitude/2).rvs(n_scales)
        scales = np.concatenate((np.array([1]), scales), axis=0)

        center = np.mean(pts2, axis=0, keepdims=True)
        scaled = (pts2 - center)[np.newaxis, :, :] * scales[:, np.newaxis, np.newaxis] + center
        if allow_artifacts:
            valid = np.arange(n_scales)  
        else:
            valid = (scaled >= 0.) * (scaled < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = scaled[idx,:,:]

    if translation:
        t_min, t_max = np.min(pts2, axis=0), np.min(1 - pts2, axis=0)
        if allow_artifacts:
            t_min += translation_overflow
            t_max += translation_overflow
        pts2 += np.array([uniform(-t_min[0], t_max[0],1), uniform(-t_min[1], t_max[1], 1)]).T

    if rotation:
        angles = np.linspace(-max_angle, max_angle, num=n_angles)
        angles = np.concatenate((angles, np.array([0.])), axis=0) 
        center = np.mean(pts2, axis=0, keepdims=True)
        rot_mat = np.reshape(np.stack([np.cos(angles), -np.sin(angles), np.sin(angles),
                                       np.cos(angles)], axis=1), [-1, 2, 2])
        rotated = np.matmul( (pts2 - center)[np.newaxis,:,:], rot_mat) + center
        if allow_artifacts:
            valid = np.arange(n_angles) 
        else:
            valid = (rotated >= 0.) * (rotated < 1.)
            valid = valid.prod(axis=1).prod(axis=1)
            valid = np.where(valid)[0]
        idx = valid[np.random.randint(valid.shape[0], size=1)].squeeze().astype(int)
        pts2 = rotated[idx,:,:]
    shape = shape[::-1]  
    pts1 *= shape[np.newaxis,:]
    pts2 *= shape[np.newaxis,:]

    def ax(p, q): return [p[0], p[1], 1, 0, 0, 0, -p[0] * q[0], -p[1] * q[0]]

    def ay(p, q): return [0, 0, 0, p[0], p[1], 1, -p[0] * q[1], -p[1] * q[1]]

    homography = cv2.getPerspectiveTransform(np.float32(pts1+shift), np.float32(pts2+shift))
    homography_inv = cv2.getPerspectiveTransform( np.float32(pts2+shift), np.float32(pts1+shift))
    return homography, homography_inv


def img_aff(img):
    img = np.array(img)
    h, w, _ = img.shape
    homography, homography_inv = sample_homography_np(shape=np.array([h, w]))
    res_img = cv2.warpPerspective(img, homography, (w, h))
    res_img = Image.fromarray(res_img.astype('uint8')).convert('RGB')
    return res_img


def img_crop(img):
    img = np.array(img)
    h, w, _ = img.shape
    ratio = 0.10
    a = min(1 - ratio, random())
    begin, end =  int(h * a), int( h * ( a + ratio))
    img[begin: end, :, :] = 0
    res_img = Image.fromarray(img.astype('uint8'))
    return res_img

def img_crop_square(img):
    img = np.array(img)
    h, w, _ = img.shape
    ratio = 0.15
    a = min(1 - ratio, random())
    h_b, h_e, w_h, w_e =  int(h * a), int( h * ( a + ratio)),  int(w * a), int( w * ( a + ratio))
    img[h_b: h_e, w_h: w_e, :] = 0
    res_img = Image.fromarray(img.astype('uint8'))
    return res_img

def img_crop_rectangle(img):
    img = np.array(img)
    h, w, _ = img.shape
    ratio_h = 0.15
    ratio_w = 0.1
    a, b = min(1 - ratio_h, random()), min(1 - ratio_w, random())
    h_b, h_e, w_h, w_e = int(h * a), int(h * (a + ratio)), int(w * b), int(w * (b + ratio))
    img[h_b: h_e, w_h: w_e, :] = 0
    res_img = Image.fromarray(img.astype('uint8'))
    return res_img


class PA100K(data.Dataset):

    def __init__(self, args, train=True):
        

        self.key_group = {}
        self.key_group['head'] = [0,1]
        self.key_group['arm'] = [2,3]
        self.key_group['upper'] = [4,5,6,7,15,16,17,18]
        self.key_group['lower'] = [8,9,10,11,12,13]
        self.key_group['foot'] = [14]
        dataset = PA100k()
        print('RAP IS LOADED')

        train_imgs = np.load('dataset/PA100k/pa100k_train_image2.npy')
        train_keypoints = np.load('dataset/PA100k/pa100k_train_key.npy')
        train_labels = np.load('dataset/PA100k/pa100k_train_label.npy')
        train_labels[train_labels>1]=1

        test_imgs = np.load('dataset/PA100k/pa100k_test_image2.npy')
        test_keypoints = np.load('dataset/PA100k/pa100k_test_key.npy')
        test_labels = np.load('dataset/PA100k/pa100k_test_label.npy')
        test_labels[test_labels>1]=1

        self.dataset = dataset
        self.imgs, self.keys, self.labels= (train_imgs, train_keypoints, train_labels) if train else (test_imgs, test_keypoints, test_labels)
        
        self.trans = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.2)], 0.5),
            T.Resize((args.resolution, args.resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if train else T.Compose([
            T.Resize((args.resolution, args.resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):

        img = self.imgs[index]
        keypoint = self.keys[index]
        label = self.labels[index]
        img = Image.open(img)
        # img_aff(img)
        # img = img_crop(img)
        img = img.convert("RGB")
        img = self.trans(img)
        label = torch.tensor(label, dtype =torch.float32)
        keypoint = torch.tensor(keypoint, dtype =torch.float32)
        return img, keypoint, label
    

    def __len__(self):
        return len(self.imgs)

class Rap(data.Dataset):
    
    def __init__(self, args, train=True):
        self.key_group = {}
        self.key_group['head'] = [25, 26, 27, 28, 29, 30]
        self.key_group['arm'] = [19, 20, 21, 22, 23, 24]
        self.key_group['upper'] = [17, 18, 31, 32, 33, 34, 35, 36, 37, 38, 39]
        self.key_group['lower'] = [40, 41, 42, 43, 44, 45]
        self.key_group['foot'] = [46, 47, 48, 49, 50]
        dataset = RAP()
        print('RAP IS LOADED')
        train_imgs = np.load('dataset/RAP/train_rap_poseimg2.npy')
        train_keypoints = np.load('dataset/RAP/train_rap_posekey.npy')
        train_labels = np.load('dataset/RAP/train_rap_poselabel.npy')
        train_labels[train_labels>1]=1

        test_imgs = np.load('dataset/RAP/test_rap_poseimg2.npy')
        test_keypoints = np.load('dataset/RAP/test_rap_posekey.npy')
        test_labels = np.load('dataset/RAP/test_rap_poselabel.npy')
        test_labels[test_labels>1]=1

        self.dataset = dataset
        self.imgs, self.keys, self.labels= (train_imgs, train_keypoints, train_labels) if train else (test_imgs, test_keypoints, test_labels)
        
        self.rand = np.random.randint(1, 10, size=(self.labels.shape))
        
        self.mask = np.ones((self.labels.shape))
        self.mask[self.rand>=1000000] = 0

        #self.labels[self.mask==0] = -1

        self.trans = T.Compose([
            T.RandomHorizontalFlip(0.5),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.2)], 0.5),
            T.Resize((args.resolution, args.resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if train else T.Compose([
            T.Resize((args.resolution, args.resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __getitem__(self, index):
        
        img = self.imgs[index]
        label = self.labels[index]
        key = self.keys[index]
        img = Image.open(img)
        img_aff(img)
        img = self.trans(img)
        label = torch.tensor(label, dtype =torch.float32)
        key = torch.tensor(key, dtype =torch.float32)
        return img, key, label
    
    def __len__(self):
        return len(self.imgs)


class Peta(data.Dataset):
    
    def __init__(self, args, train=True):
        self.key_group = {}
        self.key_group['head'] = [10, 15, 21, 33]
        self.key_group['arm'] = [5, 20, 23, 25, 29]
        self.key_group['upper'] = [4, 7, 9, 11, 14, 24, 32, 34, 35, 36, 37, 38]
        self.key_group['lower'] = [6, 8, 12, 16, 17, 18, 28, 30, 35]
        self.key_group['foot'] = []
        
        dataset = PETA()
        print('PETA IS LOADED')
        train_imgs = np.load('dataset/PETA/train_new_images2.npy')
        train_keypoints = np.load('dataset/PETA/train_pose_keypoint.npy')
        train_labels = np.load('dataset/PETA/train_new_labels.npy')

        test_imgs = np.load('dataset/PETA/test_new_images2.npy')
        test_keypoints = np.load('dataset/PETA/test_pose_keypoint.npy')
        test_labels = np.load('dataset/PETA/test_new_labels.npy')

        self.dataset = dataset
        self.imgs, self.keypoints, self.labels= (train_imgs, train_keypoints, train_labels) if train else (test_imgs, test_keypoints, test_labels)

        #self.labels[self.mask==0] = -1

        self.trans = T.Compose([
            #T.RandomHorizontalFlip(0.5),
            T.RandomApply([T.ColorJitter(0.2, 0.2, 0.2, 0.2)], 0.5),
            T.Resize((args.resolution, args.resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]) if train else T.Compose([
            T.Resize((args.resolution, args.resolution)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __getitem__(self, index):
        
        img = self.imgs[index]
        keypoint = self.keypoints[index]
        label = self.labels[index]
        img = Image.open(img)
        img = img.convert("RGB")
        img = self.trans(img)
        label = torch.tensor(label, dtype =torch.float32)
        keypoint = torch.tensor(keypoint, dtype =torch.float32)
        return img, keypoint, label
    
    def __len__(self):
        return len(self.imgs)