import argparse
import os
import time

from torch.utils.data import DataLoader, Dataset

from model import *


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', help='learning rate')
    parser.add_argument('--epochs', help='epochs')
    parser.add_argument('--batch-size', help='batch size')
    parser.add_argument('--data-path', help='dataset path')
    
    return parser.parse_args()

def train(model, batch_size, epoch, train_data: Dataset, optimizer, logger):
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    model = model.train()

    for epoch_idx in range(epoch):
        train_sample_sum, train_acc_sum, start = 0, 0., time.time()
        for batch_idx, (inputs, label_x, label_y) in enumerate(data_loader):
            inputs, label_x, label_y = torch.from_numpy(inputs).cuda(),\
                                    torch.from_numpy(label_x).cuda(),\
                                    torch.from_numpy(label_y).cuda()
            
            outputs = model(inputs)
            loss_output = loss(output, label_x, label_y)
            optimizer.zero_grad()
            loss_output.backward()
            optimizer.step()

            train_sample_sum += len(inputs)
            train_acc_sum += loss_output

            print(f'epoch {str(epoch_idx)}, process {train_sample_su  m / len(train_data)}, time {time.time() - start}s, loss {train_acc_sum / train_sample_sum}')
        
    return model

def loss(output, label_x, label_y, lamda = 0.1):
    
    back_sign_x, back_sign_y = label_x == 0, label_y == 0
    assert back_sign_x == back_sign_y

    back_sign = (back_sign_x + back_sign_y == 2).float()
    fore_sign = 1 - back_sign

    loss_term_1_x = torch.sum(torch.abs(output[:, 0, :, :] - label_x) * fore_sign) / torch.sum(fore_sign)
    loss_term_1_y = torch.sum(torch.abs(output[:, 1, :, :] - label_y) * fore_sign) / torch.sum(fore_sign)
    loss_term_1 = loss_term_1_x + loss_term_1_y

    loss_term_2_x = torch.abs(torch.sum((output[:, 0, :, :] - label_x) * fore_sign)) / torch.sum(fore_sign)
    loss_term_2_y = torch.abs(torch.sum((output[:, 1, :, :] - label_y) * fore_sign)) / torch.sum(fore_sign)
    loss_term_2 = loss_term_2_x + loss_term_2_y

    loss_term_3_x = torch.max(torch.zeros(label_x.size()), output[:, 0, :, :].squeeze(dim=1))
    loss_term_3_y = torch.max(torch.zeros(label_x.size()), output[:, 1, :, :].squeeze(dim=1))
    loss_term_3 = torch.sum((loss_term_3_x + loss_term_3_y) * back_sign) / torch.sum(back_sign)

    loss = loss_term_1 - lamda * loss_term_2 + loss_term_3

    return loss


if __name__ == '__main__':
    