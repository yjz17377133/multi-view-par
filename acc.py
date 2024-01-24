import numpy as np
import csv
import json
import os

csvFile = open("real_world_label.csv", "r")

dict_reader = csv.DictReader(csvFile)

dict = {}

for row in dict_reader:
    dict[int(row['path'])] = row

img = np.load('real_world.npy')
test1 = np.load('real_world_result.npy')

num = test1.shape[0]

gender_count = 0
hat_count = 0
jacet_count = 0
trouser_count = 0
bag_count = 0
hair_count = 0

for i in range(len(img)):
    name = os.path.basename(img[i])
    id = int(name.split('_')[0])
    
    test = test1[i]
    gender = test[19]
    hat = test[10]
    jacetType = 1 - test[29]
    trousersType = 1 - test[28]
    bag = test[4]
    hair = test[15]

    gender_count += int(gender==int(dict[id]['gender']))
    hat_count += int(hat==int(dict[id]['hat']))
    jacet_count += int(jacetType==int(dict[id]['jacetType']))
    trouser_count += int(trousersType==int(dict[id]['trousersType']))
    bag_count += int(bag==int(dict[id]['bag']))
    hair_count += int(hair==int(dict[id]['hair']))

gender_acc = gender_count/num
hat_acc = hat_count/num
jacet_acc = jacet_count/num
trouser_acc = trouser_count/num
bag_acc = bag_count/num
hair_acc = hair_count/num

print("gender_acc:", gender_acc)
print("hat_acc:", hat_acc)
print("jacet_acc:", jacet_acc)
print("trouser_acc:", trouser_acc)
print("bag_acc:", bag_acc)
print("hair_acc:", hair_acc)
