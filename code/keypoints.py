import torch
import torchvision
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

from tqdm import tqdm

import pickle

import os

device = torch.device('cuda')

model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained = True).to(device)
model.eval()


COCO_PERSON_KEYPOINT_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear',                                   
    'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow',                                                                                                                          
    'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

train_image = np.load('images.npy')
train_label = np.load('labels.npy')

train_pose_keypoint = []
train_pose_image = []
train_pose_label = []

for i in tqdm(range(len(train_image))):
    image = Image.open(train_image[i])
    image_name = train_image[i]
    w = image.size[0]
    h = image.size[1]
    image_p = Image.new('RGB', (w+400, h+400), (255, 255, 255))
    image_p.paste(image, (200, 200, w+200, h+200))
    transform_d = transforms.Compose([transforms.ToTensor()])
    image_t = transform_d(image_p).to(device)

    pred = model([image_t])
    image2 = image.copy()
    pred_keypoint = pred[0]['keypoints']
    if (len(pred_keypoint)==0): continue
    pred_keypoint = pred_keypoint[0].cpu().detach().numpy()
    pred_keypoint[:, 0] = (pred_keypoint[:, 0] - 200) / w
    pred_keypoint[:, 1] = (pred_keypoint[:, 1] - 200) / h

    train_pose_keypoint.append(pred_keypoint)
    train_pose_image.append(image_name)
    train_pose_label.append(train_label[i])


train_pose_keypoint = np.array(train_pose_keypoint)
train_pose_image = np.array(train_pose_image)
train_pose_label = np.array(train_pose_label)
print(train_pose_keypoint.shape)
np.save('new_keypoints.npy', train_pose_keypoint)
np.save('new_images.npy', train_pose_image)
np.save('new_labels.npy', train_pose_label)