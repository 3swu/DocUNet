import argparse
import os
import time

from torch.utils.data import DataLoader, Dataset

from model import *
import torch.optim as optim


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', default=0.0002, help='learning rate')
    parser.add_argument('--epochs', default=60, help='epochs')
    parser.add_argument('--batch-size', default=8, help='batch size')
    parser.add_argument('--data-path', help='dataset path')
    
    return parser.parse_args()

def train(model, batch_size, epoch, train_data: Dataset, optimizer, logger):
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model = model.train()

    for epoch_idx in range(epoch):
        train_sample_sum, train_acc_sum, start = 0, 0., time.time()
        for batch_idx, (inputs, label_x, label_y) in enumerate(data_loader):
<<<<<<< HEAD
            inputs, label_x, label_y = inputs.cuda(),\
                                     label_x.cuda(),\
                                     label_y.cuda()
=======
            inputs, label_x, label_y = torch.from_numpy(inputs).cuda(),\
                                     torch.from_numpy(label_x).cuda(),\
                                     torch.from_numpy(label_y).cuda()
>>>>>>> 6ca9587f032c410d9807cd73f023003fc190f744
            
            outputs = model(inputs)
            loss_output = loss(outputs, label_x, label_y)
            optimizer.zero_grad()
            loss_output.backward()
            optimizer.step()

            train_sample_sum += len(inputs)
            train_acc_sum += loss_output

            print(f'epoch {str(epoch_idx)}, process {train_sample_sum / len(train_data)}, time {time.time() - start}s, loss {train_acc_sum / train_sample_sum}')
        
    return model

def loss(output, label_x, label_y, lamda = 0.1):
    
    back_sign_x, back_sign_y = (label_x == 0), (label_y == 0)
    # assert back_sign_x == back_sign_y

    back_sign = (back_sign_x + back_sign_y == 2).float()
    fore_sign = 1 - back_sign

    loss_term_1_x = torch.sum(torch.abs(output[:, 0, :, :] - label_x) * fore_sign) / torch.sum(fore_sign)
    loss_term_1_y = torch.sum(torch.abs(output[:, 1, :, :] - label_y) * fore_sign) / torch.sum(fore_sign)
    loss_term_1 = loss_term_1_x + loss_term_1_y

    loss_term_2_x = torch.abs(torch.sum((output[:, 0, :, :] - label_x) * fore_sign)) / torch.sum(fore_sign)
    loss_term_2_y = torch.abs(torch.sum((output[:, 1, :, :] - label_y) * fore_sign)) / torch.sum(fore_sign)
    loss_term_2 = loss_term_2_x + loss_term_2_y

    loss_term_3_x = torch.max(torch.zeros(label_x.size()).cuda(), output[:, 0, :, :].squeeze(dim=1))
    loss_term_3_y = torch.max(torch.zeros(label_x.size()).cuda(), output[:, 1, :, :].squeeze(dim=1))
    loss_term_3 = torch.sum((loss_term_3_x + loss_term_3_y) * back_sign) / torch.sum(back_sign)

    loss = loss_term_1 - lamda * loss_term_2 + loss_term_3

    return loss


if __name__ == '__main__':
    parser = get_args()

    model_save_path = './model_save/'
    if not os.path.isdir(model_save_path):
        os.mkdir(model_save_path)

    data_path = parser.data_path
    assert data_path and os.path.isdir(data_path), 'data path not specified or not existed'

    model = Net()
    model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=parser.lr)

    data_set = DistortedDataSet(os.path.join(data_path, 'image'), os.path.join(data_path, 'label'))

    train(model, parser.batch_size, parser.epochs, data_set, optimizer, None)



    
    
    
