import sys
sys.path.insert(0, '/home/wulei/DocUNet')

import argparse
import os

import matplotlib.image as mpimg
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms

from data.inverse_deform import inverse_deform
from model import Net


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', help='model path')
    parser.add_argument('--img-path', help='image path')

    return parser.parse_args()

def predict_mesh(model: nn.Module, model_path, input_img, transforms):
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        input_ = transforms(input_img).cuda()
        input_ = input_.unsqueeze(0)
        output = model(input_)

    return output

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    parser = get_args()

    model_path = parser.model_path
    img_path = parser.img_path

    assert os.path.exists(model_path), 'model path is not exists'
    assert os.path.exists(img_path), 'image path is not exists'

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.740908, 0.709801, 0.702356), std=(0.283316, 0.299720, 0.311700)),
    ])

    img = mpimg.imread(img_path)
    model = Net().cuda()

    output = predict_mesh(model, model_path, img, transforms)
    output = output.squeeze()

    output = output.cpu().numpy()
    label_x, label_y = output[0, :, :], output[1, :, :]

    label = {
        'x': label_x,
        'y': label_y,
    }
    file_basename = os.path.basename(img_path)[:-4]
    file_save_folder = '/home/wulei/DocUNet/test/' + file_basename
    if not os.path.exists(file_save_folder):
        os.mkdir(file_save_folder)
    os.system('cp ' + img_path + ' ' + file_save_folder + '/' + file_basename + '.png')
    os.system('cp ' + os.path.dirname(os.path.dirname(img_path)) + '/label/' + file_basename + '.npz' + ' ' + file_save_folder + '/' + file_basename + '.npz')
    save_path = os.path.join(file_save_folder, file_basename + '-prediction.png')
    np.savez_compressed(os.path.join(file_save_folder, file_basename + '-prediction.npz'), x=label_x, y=label_y)

    inverse_deform(img_path, label, save_path)
