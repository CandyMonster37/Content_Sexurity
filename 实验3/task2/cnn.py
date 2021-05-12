import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import pickle
from utils import label_2_id, id_2_label
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import cv2 as cv
import os


class CnnImage(object):
    def __init__(self, data_dir='../digit_data', save_dir='./data', batch_size=64,
                 epochs=50, learning_rate=0.0005, use_gpu=False):
        self.data_dir = data_dir
        self.mode = 'train'
        self.save_dir = save_dir
        self.batch_size = batch_size
        if not use_gpu:
            self.device = torch.device("cpu")
            print('use device: cpu')
        else:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if self.device == torch.device("cpu"):
                print('no usable cuda detected! Use cpu instead')
            else:
                print('use device: gpu')
        self.epochs = epochs
        self.lr = learning_rate
        self.loss_ = []
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.model = nn.Sequential(nn.Conv2d(1, 25, 3, 1),
                                   nn.MaxPool2d(2, 2),
                                   nn.Conv2d(25, 50, 3, 1),
                                   nn.MaxPool2d(2, 2),
                                   nn.Flatten(),
                                   nn.Linear(50 * 5 * 5, 10))
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self, train_loader, test_loader, save=True):
        best_test = None
        tmp_loss = []
        best_model = self.model
        for epoch in range(self.epochs):
            for batch, label in train_loader:
                y_predict = self.model(batch)
                loss = F.cross_entropy(y_predict, label)
                tmp_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

            self.loss_.append(sum(tmp_loss)/len(tmp_loss))

            correct_percent = 0
            error_num = 0
            with torch.no_grad():
                for batch, label in test_loader:
                    y_predict = self.model(batch)
                    top_p, top_class = torch.topk(y_predict, 1)
                    top_class = top_class.squeeze(1)
                    correct_percent += torch.sum(label == top_class).item()
                    error_num += torch.sum(label != top_class).item()
            correct_percent /= len(test_loader) * self.batch_size

            if best_test is None or best_test < correct_percent:
                best_test = correct_percent
                best_model = self.model

        self.model = best_model
        if save:
            tar = self.save_dir + '/' + 'model_cnn.pkl'
            with open(tar, 'wb') as f:
                pickle.dump(self.model, f)
            print('model saved successfully, see it in: ', tar)

        print('训练出的最好的模型在测试集上的正确率: ' + str(best_test))

    def load_model(self):
        tar = self.save_dir + '/' + 'model_cnn.pkl'
        with open(tar, 'rb') as f:
            self.model = pickle.load(f)
        print('model loaded successfully!')

    def predict(self, path):
        if os.path.isdir(path):
            files = os.listdir(path)
        else:
            print('no files detected!')
            return None
        x = []
        y = []
        for item in files:
            file = path + '/' + item
            img = cv.imread(file, 0)  # 读灰度图
            re_shape = cv.resize(img, (28, 28), interpolation=cv.INTER_CUBIC)
            # shape: (28, 28), size of the reshaped image
            x.append(re_shape)
            y.append(0)
        x_set = np.array(x)
        # array, shape:(n, 28, 28)
        y_set = np.array(y)
        # array, shape:(n,)
        x_tens = torch.tensor(x_set, dtype=torch.float).unsqueeze(1).to(self.device)
        y_tens = torch.tensor(y_set, dtype=torch.long).to(self.device)
        # x_tens: torch.Tensor, torch.Size([n, 1, 28, 28])
        # y_tens: torch.Tensor, torch.Size([n])
        sets = TensorDataset(x_tens, y_tens)
        loader = DataLoader(sets, self.batch_size, True)
        with torch.no_grad():
            for batch, label in loader:
                y_predict = self.model(batch)
                top_p, top_class = torch.topk(y_predict, 1)
                top_class = top_class.squeeze(1)
        return top_class.cpu().tolist()  # 返回的结果在CPU上， 是个列表

    def set_data(self, save=True):
        sets = self.save_dir + '/sets.pkl'
        if os.path.exists(sets):
            with open(sets, 'rb') as f:
                loader = pickle.load(f)
            train_loader = loader[0]
            test_loader = loader[1]
            print('dataset loaded successfully!')
        else:
            self.mode = 'train'
            train_dataset = self.dataset()
            train_loader = DataLoader(train_dataset, self.batch_size, True)
            self.mode = 'test'
            test_dataset = self.dataset()
            test_loader = DataLoader(test_dataset, self.batch_size, True)
            print('dataset built successfully!')

            if save:
                loader = (train_loader, test_loader)
                with open(sets, 'wb') as f:
                    pickle.dump(loader, f)
                print('dataset saved in: ', sets)
        return train_loader, test_loader

    def dataset(self):
        path = self.data_dir + '/' + self.mode
        labels = os.listdir(path)
        x = []
        y = []
        for label in labels:
            img_path = path + '/' + label
            files = os.listdir(img_path)
            for item in files:
                file = img_path + '/' + item
                img = cv.imread(file, 0)  # 读灰度图
                # shape: (28, 28), size of the image
                x.append(img)
                y.append(label_2_id[label])
        x_set = np.array(x)
        # array, shape:(n, 28, 28)
        x_tens = torch.tensor(x_set, dtype=torch.float).unsqueeze(1).to(self.device)
        # torch.Tensor, torch.Size([n, 1, 28, 28])
        y_set = np.array(y)
        # array, shape:(n,)
        y_tens = torch.tensor(y_set, dtype=torch.long).to(self.device)
        # torch.Tensor, torch.Size([n])
        sets = TensorDataset(x_tens, y_tens)
        return sets


def train_():
    epochs = 100
    cnn = CnnImage(data_dir='../digit_data', save_dir='./data', batch_size=64,
                   use_gpu=True, epochs=epochs, learning_rate=0.0005)
    train_loader, test_loader = cnn.set_data(save=True)
    cnn.train(train_loader=train_loader, test_loader=test_loader, save=True)
    loss = cnn.loss_

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('epoch - loss')
    ax.set_xlabel('epoch')
    ax.set_ylabel('loss')
    x_axis = [i for i in range(len(loss))]
    plt.plot(x_axis, loss)
    plt.show()


def test():
    cnn = CnnImage(data_dir='../digit_data', save_dir='./data', batch_size=64,
                   use_gpu=True)
    cnn.load_model()
    test_path = 'testimg'
    pre = cnn.predict(test_path)
    if pre is None:
        return
    else:
        files = os.listdir(test_path)
        for i in range(len(files)):
            res = files[i] + ' 的预测结果为: ' + str(pre[i])
            print(res)


if __name__ == '__main__':
    train_()
    test()
