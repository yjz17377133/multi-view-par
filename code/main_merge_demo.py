import argparse

import manager_merge

import torch.backends.cudnn as cudnn

from PIL import Image

parser = argparse.ArgumentParser()

parser.add_argument('-c', '--cuda', default=[0], type=int, nargs='+', metavar='cuda',
                    help='gpu if cuda else cpu')

parser.add_argument('-l', '--lr', default=1e-4, type=float, metavar='learning_rate',
                    help='initial learning rate')

parser.add_argument('-b', '--batch_size', default=32, type=int, metavar='batch_size', # 64
                    help='batch_size for training')
                    
parser.add_argument('-b2', '--batch_size2', default=32, type=int, metavar='batch_size2', # 64
                    help='batch_size for val')

parser.add_argument('-rs', '--resolution', default=224, type=int, metavar='resolution', # 224
                    help='resolution for img')
                    
parser.add_argument('-s', '--train_step', default=30, type=int, metavar='train_step',
                    help='max training step')

parser.add_argument('-d', '--dataset', default='PA100k', type=str, metavar='dataset',
                    help="dataset (e.g. PETA,RAP)", choices=['PETA', 'RAP', 'PA100k'])

parser.add_argument('-bb', '--backbone', default='ConvBase', type=str, metavar='backbone',
                    help='backbone, e.g. Vgg19, resnet50, resnet101',
                    choices=['ConvBase', 'ConvLarge', 'ConvXlarge', 'ConvSmall', 'ConvTiny', 'Resnet50', 'Resnet101'])
                    
parser.add_argument('--ema-decay', default=0.9997, type=float, metavar='M',
                        help='decay of model ema')
parser.add_argument('--ema-epoch', default=0, type=int, metavar='M',
                        help='start ema epoch')

parser.add_argument('-o', '--optim', default='Adam', type=str, metavar='optim',
                    help='optim, e.g. Adam, optim', choices=['Adam', 'SGD'])

parser.add_argument('-cr', '--criterion', default='BCE', type=str, metavar='criterion',
                    help='criterion, e.g. BCE', choices=['BCE', 'MSE', 'Weight', 'Weight_MSE', 'Weight2', 'Weight3', 'Weight4', 'BCE_ALM', 'smooth'])

parser.add_argument('-sn', '--SubNet', default=True, type=bool, metavar='SubNet',
                    help='max training step')

parser.add_argument('-g', '--GCN', default=2, type=int, metavar='GCN',
                    help='GCN layers')

parser.add_argument('-lg', '--learn_gcn', default=False, type=bool, metavar='learn_gcn')

parser.add_argument('-info', '--information', default='save_peta', type=str)

parser.add_argument('-lga', '--learn_gcn_attention', default=True, type=bool, metavar='learn_gcn_attention')

parser.add_argument('-sp', '--saved_path', default="", type=str)

parser.add_argument('-se', '--embedding_se', default=False, type=bool, metavar='embedding_se')

parser.add_argument('-m', '--mark', default=0, type=int, metavar='mark',
                    help='0 - all 1 spatial 2 hiera ')
parser.add_argument('-n', '--net', default=0, type=int, metavar='net', help='0 ori, 1 vlad')

parser.add_argument('-p', '--platform', default="torch", type=str, metavar='platform', help='torch, paddle')

parser.add_argument('-detail', '--detail', default="False", type=bool, metavar='detail acc for label', help='true or false')

parser.add_argument('-aD', '--augData', default="normal", type=str, metavar='data Augment Operation', help='facon')

parser.add_argument('-sF', '--superFocus', default=0, nargs='+', type=int, metavar='super Focus label', help='num of index of attr')

parser.add_argument('-mB', '--multiBranch', default=False, type=bool, metavar='multi branche button', help='bool of multi branche of model')

parser.add_argument('-bu', '--but', default=False, type=bool, metavar='multi decoder object', help='index of multi decoder object of model')

import paddle
from PIL import Image
from paddle import io as data #paddle.io.Dataset
from paddle.vision import transforms as T

from dataset.PETA2_MHD import PETA
from dataset.RAP_MHD import RAP
from math import pi
import cv2
import numpy as np
from random import random

import sys
import os

from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def trans_paddle(args):
    transform = T.Compose([
        T.Resize((args.resolution, args.resolution)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform

import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms as T2


import cv2
import numpy as np
from random import random

import torchvision

from tqdm import tqdm
 
import torchsummary
import time


def trans_torch(args):
    transform = T2.Compose([
            T2.Resize((args.resolution, args.resolution)),
            T2.ToTensor(),
            T2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    return transform


def get_keypoint(image, model_key, check = False):
 
    model_key.eval()
    img = image

    img = Image.open(img)
    img = img.convert("RGB")

    w = img.size[0]
    h = img.size[1]
    
    transform_d = T2.Compose([T2.ToTensor()])
    image_p = Image.new('RGB', (w+400, h+400), (255, 255, 255))
    image_p.paste(img, (200, 200, w+200, h+200))
    
    
    image_t = transform_d(image_p).to(device)


    transform = trans_torch(args)
    img = transform(img)
    if check:
        torchsummary.summary(model_key)

        memory_summary = torch.cuda.memory_summary(device=None, abbreviated=False)
        print("KEY_____memoty:", memory_summary)

        
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()

        pred = model_key([image_t])

        end_event.record()
        torch.cuda.synchronize()  

        forward_time = start_event.elapsed_time(end_event)
        print(f"KEY_____Forward time: {forward_time} milliseconds")
    else:
        pred = model_key([image_t])

    
    pred_keypoint = pred[0]['keypoints']
    if pred_keypoint.shape[0]==0:
        print("No keypoint found, check the input plz")
    keypoints = pred_keypoint[0]


    keypoints[:, 0] = (keypoints[:, 0] - 200) / w
    keypoints[:, 1] = (keypoints[:, 1] - 200) / h


    return keypoints

def load_control(control, saved_param):

    restore_dict = control.load(saved_param)

    print('restored from %s' % saved_param)

    control.net.load_state_dict(restore_dict['model_state_dict'])    


def image_eval_API1(image, control, model_key, check = False):


    # Check if a GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    img = image
    image_name = image

    img = Image.open(img)
    img = img.convert("RGB")

    

    transform = trans_torch(args)
    img = transform(img)

    keypoints = get_keypoint(image_name, model_key, check=False)
    keypoints = torch.tensor(keypoints).float().cuda()


    out = control.Framework_predict(img, keypoints, check)
    res = (out>0.5).astype(int)
    return res

def image_eval_API2(image, control, model_key,check = False):


    # Check if a GPU is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    img = image
    image_name = image

    img = Image.open(img)
    img = img.convert("RGB")

    

    transform = trans_torch(args)
    img = transform(img)

    keypoints = get_keypoint(image_name, model_key, check=False)
    keypoints = torch.tensor(keypoints).float().cuda()

    out, feature1, feature2 = control.Framework_predict_feature(img, keypoints, check)

    res = (out>0.5).astype(int)
    return res, feature1, feature2

if __name__ == '__main__':

    saved_param = 'model_peta.pt'

    args = parser.parse_args()

    image = '1.jpg'

    control = manager_merge.Manager(args)

    load_control(control, saved_param)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model_key = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained = True).to(device)
    model_key.eval()
    
    
    onlyfiles = np.load('./predict/hktest.npy')
    
    result = []

    for image in tqdm(onlyfiles):

        res, feature1, feature2 = image_eval_API2(image, control, model_key)
        result.append(res)
    result = np.array(result)

    print(result.shape)
    np.save('./result.npy', result)

    
