import argparse
import os
import time

import torch.optim as optim
import torchvision.transforms
from torch.utils.data import DataLoader, Dataset

from loss import Loss
from model import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.002, help='learning rate')
    parser.add_argument('--epochs', default=80, help='epochs')
    parser.add_argument('--batch-size', default=8, help='batch size')
    parser.add_argument('--data-path', help='dataset path')
    parser.add_argument('--pre-trained', default=False, help='use pre trained model')
    parser.add_argument('--pre-trained-path', help='pre trained model path')
    
    return parser.parse_args()

def train(model, batch_size, epoch, train_data: Dataset, optimizer, logger, save_path):
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model.train()
    loss = Loss()
    # loss = nn.MSELoss()

    for epoch_idx in range(epoch):
        train_sample_sum, train_acc_sum, start = 0, 0., time.time()
        for batch_idx, (inputs, label_x, label_y) in enumerate(data_loader):
            inputs, label_x, label_y = inputs.cuda(), label_x.cuda(), label_y.cuda()
            
            outputs = model(inputs)
            loss_output = loss(outputs, label_x, label_y)
            # loss_output = loss(outputs, torch.stack([label_x, label_y], 1).float())
            # loss_output = loss_output / batch_size  
            optimizer.zero_grad()
            loss_output.backward()
            # print(inputs.grad)
            optimizer.step()

            train_sample_sum += len(inputs)
            train_acc_sum += loss_output

            print('epoch {}, process {:.2f}, time {:.2f}, loss {:.3f}'.format(str(epoch_idx), train_sample_sum / len(train_data), time.time() - start, train_acc_sum / train_sample_sum))
        with open(save_path + 'loss.txt', 'a') as f:
            f.write(str(train_acc_sum / train_sample_sum) + '\n')
        torch.save(model.state_dict(), save_path + str(epoch_idx) + '.pt')

    return model

# def loss(output, label_x, label_y, lamda = 0.1):
    
#     back_sign_x, back_sign_y = (label_x == 0.).int(), (label_y == 0.).int()
#     # assert back_sign_x == back_sign_y

#     back_sign = ((back_sign_x + back_sign_y) == 2).float()
#     fore_sign = 1 - back_sign

#     loss_term_1_x = torch.sum(torch.abs(output[:, 0, :, :] - label_x) * fore_sign) / torch.sum(fore_sign)
#     loss_term_1_y = torch.sum(torch.abs(output[:, 1, :, :] - label_y) * fore_sign) / torch.sum(fore_sign)
#     loss_term_1 = loss_term_1_x + loss_term_1_y

#     loss_term_2_x = torch.abs(torch.sum((output[:, 0, :, :] - label_x) * fore_sign)) / torch.sum(fore_sign)
#     loss_term_2_y = torch.abs(torch.sum((output[:, 1, :, :] - label_y) * fore_sign)) / torch.sum(fore_sign)
#     loss_term_2 = loss_term_2_x + loss_term_2_y

#     loss_term_3_x = torch.max(torch.zeros(label_x.size()).cuda(), output[:, 0, :, :].squeeze(dim=1))
#     loss_term_3_y = torch.max(torch.zeros(label_x.size()).cuda(), output[:, 1, :, :].squeeze(dim=1))
#     loss_term_3 = torch.sum((loss_term_3_x + loss_term_3_y) * back_sign) / torch.sum(back_sign)

#     loss = loss_term_1 - lamda * loss_term_2 + loss_term_3

#     return loss


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    parser = get_args()

    model_save_path = './model_save/'
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)

    data_path = parser.data_path
    assert data_path and os.path.isdir(data_path), 'data path not specified or not existed'

    model = Net()
    if parser.pre_trained:
        assert os.path.exists(parser.pre_trained_path), 'pre trained path is not exists'
        model.load_state_dict(torch.load(parser.pre_trained_path))
        print(f'model {parser.pre_trained_path} loaded')

    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=float(parser.lr))

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.740908, 0.709801, 0.702356), std=(0.283316, 0.299720, 0.311700)),
        ])
    data_set = DistortedDataSet(os.path.join(data_path, 'image'), os.path.join(data_path, 'label'), transform, is_train=True)

    train(model, int(parser.batch_size), parser.epochs, data_set, optimizer, None, model_save_path)
