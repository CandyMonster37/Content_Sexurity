#  -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.optim as optim
from models import ResNet18, ResNet50
from makedata import *
from tqdm import trange
import numpy as np
import os
from evaluator import getAUC, getACC
import datetime
import matplotlib.pyplot as plt
from PIL import Image


def train(model, optimizer, loss_function, train_loader, device):
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


def detect(in_dir, model_dir, num_class=2):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = ResNet50(in_channels=3, num_classes=num_class).to(device)
    model_loader = os.listdir(model_dir)
    restore_model_path = os.path.join(model_dir, model_loader[0])
    model.load_state_dict(torch.load(restore_model_path)['net'])

    h = 56
    w = 56

    tar_dir = './to_check'
    os.mkdir(tar_dir)
    files = []
    if os.path.isdir(in_dir):
        all_files = os.listdir(in_dir)
        for file in all_files:
            ori = os.path.join(in_dir, file)
            ori = os.path.abspath(ori)
            tar = os.path.abspath(tar_dir)
            tmp = os.popen('copy {0} {1}'.format(ori, tar))
            files.append(file)
    else:
        tar = os.path.abspath(tar_dir)
        ori = os.path.abspath(in_dir)
        tmp = os.popen('mv {0} {1}'.format(ori, tar))
        files.append(in_dir)

    to_check_transform = transforms.Compose([
        transforms.Resize([h, w]),  # resize到w * h大小
        transforms.ToTensor(),  # 转化成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for i in range(len(files)):
        img_file = os.path.join(tar_dir, files[i])
        img = Image.open(img_file).convert('RGB')
        img = to_check_transform(img)
        img = img.unsqueeze(0)
        ans = predict(img, model, device)
        if ans == 0:
            print('Image \'{0}\' is considered no problem.'.format(files[i]))
        elif ans == 1:
            print('Image \'{0}\' may contain illegal content.'.format(files[i]))
        else:
            print('Image \'{0}\' contains illegal content!'.format(files[i]))
        # print('{0} is recognized as class {1}'.format(files[i], int(ans)))
    path = os.path.abspath('./to_check')
    tmp = os.popen('rmdir /s/q {0}'.format(path))


def predict(input, model, device):
    model.to(device)
    with torch.no_grad():
        input = input.to(device)
        out = model(input)
        _, pre = torch.max(out.data, 1)
        return pre.item()


def show_res_imgs(end_epoch, train, test, dev, y_name, img_dir):
    x_axis = []
    for i in range(end_epoch):
        x_axis.append(i + 1)

    plt.figure(figsize=(10.8, 7.2), dpi=100)
    # figsize=(a, b)，图像尺寸; dpi，分辨率，default=100; 最终图像大小(a*dip) : (b*dpi)
    # eg: figsize=(10.8, 7.2), dpi=100; 最终图像大小1080 * 720像素

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

    plt.savefig(img_dir)
    plt.show()


def train_and_test(in_dir, loader_dir, output_root, lr=0.001, end_epoch=100, batch_size=64, num_class=2):
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

    model = ResNet50(in_channels=3, num_classes=num_class).to(device)
    loss_function = nn.CrossEntropyLoss()  # 设置损失函数
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 设置优化器和学习率

    for epoch in trange(start_epoch, end_epoch):
        print('\n')
        train(model, optimizer, loss_function, train_loader, device)
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


if __name__ == '__main__':
    num_class = 4
    in_dir = './data_' + str(num_class)
    output_root = './ckpt_' + str(num_class)
    loader_dir = './dataset_' + str(num_class)

    end_epoch = 100
    batch_size = 8
    lr = 0.001
    train_and_test(in_dir=in_dir, loader_dir=loader_dir, output_root=output_root,
         end_epoch=end_epoch, batch_size=batch_size, lr=lr, num_class=num_class)
    path = os.path.abspath(loader_dir)
    tmp = os.popen('rmdir /s/q {0}'.format(path))

    num_class = 2
    in_dir = './data_' + str(num_class)
    output_root = './ckpt_' + str(num_class)
    loader_dir = './dataset_' + str(num_class)
    train_and_test(in_dir=in_dir, loader_dir=loader_dir, output_root=output_root,
         end_epoch=end_epoch, batch_size=batch_size, lr=lr, num_class=num_class)
    path = os.path.abspath(loader_dir)
    dele = os.popen('rmdir /s/q {0}'.format(path))

    # in_dir = './tocheck'
    # model_dir = './ckpt_4_yes'
    # detect(in_dir, model_dir, num_class=4)
