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
import matplotlib.pyplot as plt


def train(model, optimizer, loss_function, train_loader, device, num_class=2):
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        targets = targets.squeeze().long().to(device)
        loss = loss_function(outputs, targets)

        loss.backward()
        optimizer.step()


def val(model, val_loader, device, val_auc_list, dir_path, epoch, num_class=2):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            outputs = model(inputs.to(device))

            if num_class == 2:
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, num_class=num_class)
        val_auc_list.append(auc)

    state = {
        'net': model.state_dict(),
        'auc': auc,
        'epoch': epoch
    }

    value = round(auc, 5)
    path = os.path.join(dir_path, 'ckpt_{0}_auc_{1}.pth'.format(epoch, value))
    torch.save(state, path)


def test(model, data_loader, device, auc_list, acc_list, mode='train', num_class=2):
    model.eval()
    y_true = torch.tensor([]).to(device)
    y_score = torch.tensor([]).to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            outputs = model(inputs.to(device))

            if num_class == 2:
                targets = targets.to(torch.float32).to(device)
                m = nn.Sigmoid()
                outputs = m(outputs).to(device)
            else:
                targets = targets.squeeze().long().to(device)
                m = nn.Softmax(dim=1)
                outputs = m(outputs).to(device)
                targets = targets.float().resize_(len(targets), 1)

            y_true = torch.cat((y_true, targets), 0)
            y_score = torch.cat((y_score, outputs), 0)

        y_true = y_true.cpu().numpy()
        y_score = y_score.detach().cpu().numpy()
        auc = getAUC(y_true, y_score, num_class=num_class)
        auc_list.append(auc)
        acc = getACC(y_true, y_score, num_class=num_class)
        acc_list.append(acc)
        print('{0} AUC: {1} ACC: {2}'.format(mode, round(auc, 5), round(acc, 5)))

        # if output_root is not None:
        #     output_dir = output_root
        #     if not os.path.exists(output_dir):
        #         os.mkdir(output_dir)
        #     out_path = os.path.join(output_dir, '{0}.csv'.format(mode))
        #     save_results(y_true, y_score, out_path)


def main(in_dir, loader_dir, output_root, lr=0.001, end_epoch=100, batch_size=64, num_class=2):
    start_epoch = 0
    val_auc_list = []
    train_auc_list = []
    test_auc_list = []
    dev_auc_list = []
    train_acc_list = []
    test_acc_list = []
    dev_acc_list = []
    if not os.path.exists(output_root):
        os.mkdir(output_root)

    print('==> Preparing data...')
    train_loader, test_loader, dev_loader = \
        get_data_loader(in_dir=in_dir, out_dir=loader_dir, batch_size=batch_size)
    print('Data prepared!')

    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('==> Building and training model...')
    print('Begin at: ', now)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    model = ResNet50(in_channels=3, num_classes=num_class).to(device)
    loss_function = nn.CrossEntropyLoss()  # 设置损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 设置优化器和学习率

    for epoch in trange(start_epoch, end_epoch):
        train(model, optimizer, loss_function, train_loader, device, num_class=num_class)
        val(model, test_loader, device, val_auc_list, output_root, epoch, num_class=num_class)
        test(model, train_loader, device, auc_list=train_auc_list, acc_list=train_acc_list,
             mode='train', num_class=num_class)
        test(model, test_loader, device, auc_list=test_auc_list, acc_list=test_acc_list,
             mode='test', num_class=num_class)
        test(model, dev_loader, device, auc_list=dev_auc_list, acc_list=dev_acc_list,
             mode='dev', num_class=num_class)

    auc_list = np.array(val_auc_list)
    index = auc_list.argmax()
    now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Finish at: ', now)
    print('epoch {0} is the best model'.format(index))

    print('==> Testing model...')
    value = auc_list[index]
    value = round(value, 5)
    tar_res = 'ckpt_{0}_auc_{1}.pth'.format(index, value)

    models = os.listdir(output_root)
    for item in models:
        if item == tar_res:
            continue
        else:
            to_del = os.path.join(output_root, item)
            to_del = os.path.abspath(to_del)
            tmp = os.popen('del {0}'.format(to_del))

    show_res_imgs(end_epoch, train_auc_list, test_auc_list, dev_auc_list,
                  y_name='auc value', img_dir='./auc_classes_{}.png'.format(str(num_class)))
    show_res_imgs(end_epoch, train_acc_list, test_acc_list, dev_acc_list,
                  y_name='acc value', img_dir='./acc_classes_{}.png'.format(str(num_class)))
    # restore_model_path = os.path.join(output_root, tar_res)
    # model.load_state_dict(torch.load(restore_model_path)['net'])


def show_res_imgs(end_epoch, train, test, dev, y_name, img_dir):
    x_axis = []
    for i in range(end_epoch):
        x_axis.append(i + 1)

    plt.plot(x_axis, train, color='red', label='train ' + y_name)
    plt.plot(x_axis, test, color='green', label='test' + y_name)
    plt.plot(x_axis, dev, color='skyblue', label='dev' + y_name)

    plt.title('epochs = {0}'.format(str(end_epoch)))
    plt.xlabel('epoch')
    plt.ylabel(y_name)
    plt.legend()

    lim = int(end_epoch / 5) * end_epoch + 5
    x_ticks = np.linspace(0, lim, 5)
    plt.xticks(x_ticks)
    y_ticks = np.arange(0, 1, 0.05)
    plt.yticks(y_ticks)

    plt.figure(figsize=(12, 9), dpi=300)
    # figsize，图像尺寸，默认为[6.4, 4.8], 640 * 480
    # dpi，分辨率，默认为100.0

    plt.savefig(img_dir)
    plt.show()


# def a_try(in_dir, loader_dir, output_root, batch_size=4, num_class=2):
#     lr = 0.001
#     train_loader, test_loader, dev_loader = \
#         get_data_loader(in_dir=in_dir, out_dir=loader_dir, batch_size=batch_size)
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     # device = torch.device("cpu")
#
#     model = ResNet50(in_channels=3, num_classes=num_class).to(device)
#     # loss_function = nn.CrossEntropyLoss()  # 设置损失函数
#     # optimizer = optim.Adam(model.parameters(), lr=lr)  # 设置优化器和学习率
#
#     restore_model_path = os.path.join(output_root, 'ckpt_21_auc_0.84517.pth')
#     model.load_state_dict(torch.load(restore_model_path)['net'])
#     test(model, train_loader, device, output_root=output_root, mode='train', num_class=num_class)
#     test(model, dev_loader, device, output_root=output_root, mode='val', num_class=num_class)
#     test(model, test_loader, device, output_root=output_root, mode='test', num_class=num_class)


if __name__ == '__main__':
    num_class = 4
    in_dir = './data_' + str(num_class)
    output_root = './ckpt_' + str(num_class)
    loader_dir = './dataset_' + str(num_class)

    end_epoch = 100
    batch_size = 8
    lr = 0.001
    main(in_dir=in_dir, loader_dir=loader_dir, output_root=output_root,
         end_epoch=end_epoch, batch_size=batch_size, lr=lr, num_class=num_class)
    # a_try(in_dir, output_root, batch_size=4, num_class=num_class)
