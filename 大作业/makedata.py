import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import os
import pickle


def prepare_data(in_dir='./data', out_dir='./dataset', batch_size=128):
    labels = os.listdir(in_dir)
    splited = ['train', 'test', 'dev']
    filenums = 0
    s1 = 0
    s2 = 0
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    for kind in splited:
        dir_name = os.path.join(out_dir, kind)
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
    for label in labels:
        path = os.path.join(in_dir, label)
        files = os.listdir(path)
        filenums += len(files)
        s1_ = round(0.8 * len(files))
        s1 += s1_
        s2_ = round(0.1 * len(files))
        s2 += s2_
        s2_ += s1_

        #  训练集 : 测试集 : 验证集 = 8 : 1 : 1
        tar = os.path.join(out_dir, splited[0])
        tar = os.path.join(tar, label)
        tar = os.path.abspath(tar)
        os.mkdir(tar)
        for file in files[:s1_]:  # 训练集
            ori = os.path.join(path, file)
            ori = os.path.abspath(ori)
            tmp = os.popen('copy {0} {1}'.format(ori, tar))

        tar = os.path.join(out_dir, splited[1])
        tar = os.path.join(tar, label)
        tar = os.path.abspath(tar)
        os.mkdir(tar)
        for file in files[s1_:s2_]:  # 测试集
            ori = os.path.join(path, file)
            ori = os.path.abspath(ori)
            tmp = os.popen('copy {0} {1}'.format(ori, tar))

        tar = os.path.join(out_dir, splited[2])
        tar = os.path.join(tar, label)
        tar = os.path.abspath(tar)
        os.mkdir(tar)
        for file in files[s2_:]:  # 验证集
            ori = os.path.join(path, file)
            ori = os.path.abspath(ori)
            tmp = os.popen('copy {0} {1}'.format(ori, tar))

    print('Make data sets successfully!')
    print('kinds: {0}\ndata nums: {4}\ntrain nums: {1}\ntest nums: {2}\ndev nums: {3}'
          .format(len(labels), s1, s2, filenums-s1 - s2, filenums))

    make_data_loader(root_dir=out_dir, batch_size=batch_size)

    # for item in splited:
    #     to_del = os.path.join(out_dir, item)
    #     to_del = os.path.abspath(to_del)
    #     tmp = os.popen('rd /s/q {0}'.format(to_del))
    print('Make dataloader successfully!')


def make_data_loader(root_dir='./dataset', batch_size=128):
    mode = ['train', 'test', 'dev']
    h = 56
    w = 56
    # 设置训练集
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop([h, w]),  # 重置分辨率为w * h大小
        transforms.RandomHorizontalFlip(),  # 随机水平翻转
        transforms.RandomRotation(degrees=15),  # 随机旋转
        transforms.ToTensor(),  # 转化成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
    ])
    train_dataset = ImageFolder(os.path.join(root_dir, mode[0]), transform=train_transform)  # 训练集数据
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # 加载数据
    to_save = os.path.join(root_dir, 'train_loader.pkl')
    with open(to_save, 'wb') as f:
        pickle.dump(train_loader, f)

    # 设置测试集
    test_transform = transforms.Compose([
        transforms.Resize([h, w]),  # resize到w * h大小
        transforms.ToTensor(),  # 转化成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
    ])
    test_dataset = ImageFolder(os.path.join(root_dir, mode[1]), transform=test_transform)  # 测试集数据
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    to_save = os.path.join(root_dir, 'test_loader.pkl')
    with open(to_save, 'wb') as f:
        pickle.dump(test_loader, f)

    # 设置验证集
    dev_transform = transforms.Compose([
        transforms.Resize([h, w]),  # resize到w * h大小
        transforms.ToTensor(),  # 转化成Tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 正则化
    ])
    dev_dataset = ImageFolder(os.path.join(root_dir, mode[2]), transform=dev_transform)  # 验证集数据
    dev_loader = DataLoader(dataset=dev_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    to_save = os.path.join(root_dir, 'dev_loader.pkl')
    with open(to_save, 'wb') as f:
        pickle.dump(dev_loader, f)


def get_data_loader(in_dir='./data', out_dir='./dataset', batch_size=128):
    if not os.path.exists('./dataset/train_loader.pkl'):
        prepare_data(in_dir=in_dir, out_dir=out_dir, batch_size=batch_size)

    to_load = os.path.join(out_dir, 'train_loader.pkl')
    with open(to_load, 'rb') as f:
        train_loader = pickle.load(f)
    to_load = os.path.join(out_dir, 'test_loader.pkl')
    with open(to_load, 'rb') as f:
        test_loader = pickle.load(f)
    to_load = os.path.join(out_dir, 'dev_loader.pkl')
    with open(to_load, 'rb') as f:
        dev_loader = pickle.load(f)

    return train_loader, test_loader, dev_loader


if __name__ == '__main__':
    prepare_data(in_dir='./data', out_dir='./dataset', batch_size=4)

