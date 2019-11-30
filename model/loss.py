import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, lamda=0.1):
        super(Loss, self).__init__()
        self.lamda = lamda

    def forward(self, output, label_x, label_y):
        back_sign_x, back_sign_y = (label_x == 0.).int(), (label_y == 0.).int()
        # assert back_sign_x == back_sign_y

        back_sign = ((back_sign_x + back_sign_y) == 2).float()
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

        loss = loss_term_1 - self.lamda * loss_term_2 + loss_term_3

        return loss


if __name__ == '__main__':
    from data import DistortedDataSet
    from torch.utils.data import DataLoader
    import torchvision.transforms
    import os

    data_path = '/home/cc/wulei/data_gen'
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.740908, 0.709801, 0.702356), std=(0.283316, 0.299720, 0.311700)),
        ])
    data_set = DistortedDataSet(os.path.join(data_path, 'image'), os.path.join(data_path, 'label'), transform, is_train=True)
    data_loader = DataLoader(data_set, batch_size=8, shuffle=True)
    
    loss = Loss()
    for i, (_, label_x, label_y) in enumerate(data_loader):
        label_x, label_y = label_x.cuda(), label_y.cuda()
        output = torch.randn(8, 2, 352, 272, requires_grad=True).cuda()
        l = loss(output, label_x, label_y)
        print(l)
        l.backward()
        print(output.grad)
