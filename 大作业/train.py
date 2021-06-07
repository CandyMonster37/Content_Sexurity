#  -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from models import ResNet18, ResNet50
from makedata import get_data_loader
from tqdm import trange
import numpy as np
import os
from evaluator import getAUC, getACC, save_results
import datetime


def train(model, optimizer, loss_function, train_loader, device):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = targets.squeeze().long.to(device)
        loss = loss_function(outputs, targets)

        loss.backword()
        optimizer.step()


def val(model, val_loader, device, val_auc_list, dir_path, epoch):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            targets = targets.squeeze().long.to(device)
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch
    }

    value = round(auc, 5)
    path = os.path.join(dir_path, 'ckpt_{0}_auc_{1}.pth'.format(epoch, value))
    torch.save(state, path)


def test(model, data_loader, device, output_root, mode='train'):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            targets = targets.squeeze().long.to(device)
            m = nn.Softmax(dim=1)
            outputs = m(outputs).to(device)
            targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score)
        acc = getACC(y_true, y_score)
        print('{0} AUC: {1} ACC: {2}'.format(mode, round(auc, 5), round(acc, 5)))

        if output_root is not None:
            output_dir = output_root
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)
            out_path = os.path.join(output_dir, '{0}.csv'.format(mode))
            save_results(y_true, y_score, out_path)


def main(in_dir, output_root, end_epoch=100, batch_size=64):
    start_epoch = 0
    lr = 0.001
    val_auc_list = []
    if not os.path.exists(output_root):
        os.mkdir(output_root)

    print('==> Preparing data...')
    train_loader, test_loader, dev_loader = \
        get_data_loader(in_dir=in_dir, out_dir='./dataset', batch_size=batch_size)
    print('Data prepared!')

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('==> Building and training model...')
    print('Begin at: ', now)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model = ResNet50(in_channels=3, num_classes=4).to(device)
    loss_function = nn.CrossEntropyLoss()  # 设置损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 设置优化器和学习率

    for epoch in trange(start_epoch, end_epoch):
        train(model, optimizer, loss_function, train_loader, device)
        val(model, test_loader, device, val_auc_list, output_root, epoch)

    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Finish at: ', now)
    print('epoch {0} is the best model'.format(index))

    print('==> Testing model...')
    value = auc_list[index]
    value = round(value, 5)
    restore_model_path = os.path.join(output_root, 'ckpt_{0}_auc_{1}.pth'.format(index, value))
    model.load_state_dict(torch.load(restore_model_path)['net'])
    test(model, train_loader, device, output_root, mode='train')
    test(model, dev_loader, device,  output_root, mode='val')
    test(model, test_loader, device, output_root, mode='test')


if __name__ == '__main__':

    in_dir = './data'
    output_root = './ckpt'
    end_epoch = 100
    batch_size = 128
    main(in_dir, output_root, end_epoch=end_epoch, batch_size=batch_size)
