# author wulei
# date 2019-10-08

import os
import random
import time
import numpy as np
import json
import uuid

from anti_deform import from_label_deform
from cover_texture import *
from deform import *

config = {}
with open('config.json') as f:
    config = json.load(f)

img_folder = config['img_folder']
texture_folder = config['texture_folder']
target_folder = config['target_folder']
deform_rounds = config['deform_rounds']
numbers = config['numbers']

assert os.path.exists(img_folder), 'img_folder do not exists'
assert os.path.exists(texture_folder), 'texture_folder do not exists'
assert os.path.exists(target_folder), 'target_folder do not exists'

target_img_folder = os.path.join(target_folder, 'image')
target_label_folder = os.path.join(target_folder, 'label')

if not os.path.exists(target_img_folder):
    print(f'mkdir {target_img_folder}')
    os.mkdir(target_img_folder)
if not os.path.exists(target_label_folder):
    print(f'mkdir {target_label_folder}')
    os.mkdir(target_label_folder)

src_images = os.listdir(img_folder)
texture_images = os.listdir(texture_folder)

random.shuffle(src_images)
random.shuffle(texture_images)

count = 0
while True:
    for img in src_images:
        print(f'------\ncount: {count}')
        start = time.time()
        img_path = img_folder + '/' + img
        texture_path = texture_folder + '/' + texture_images[random.randint(0, len(texture_images) - 1)]
        operation = []
        fold_rounds = int(deform_rounds * 0.7)
        for _ in range(fold_rounds):
            operation.append(0)
        for _ in range(deform_rounds - fold_rounds):
            operation.append(1)
        random.shuffle(operation)

        file_name = img[: img.index('.')] + '-' + uuid.uuid1().hex[0:8]

        # generate
        label_x, label_y = gen_deform_label(img_path, operation)

        img_b, img_g, img_r, label_x, label_y = from_label_deform(label_x, label_y, img_path)
        img = texture(label_x, img_b, img_g, img_r, texture_path)
        cv2.imwrite(os.path.join(target_img_folder, file_name) + '.png', img)
        np.savez_compressed(os.path.join(target_label_folder, file_name), x = label_x, y = label_y)
        
        print(f'image deformation finished: {file_name + ".png"}, time: {time.time() - start} s')
        count += 1
        if count == numbers:
            print(f'{numbers} images finished')
            exit(0)

