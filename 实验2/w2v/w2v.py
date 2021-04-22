# -*- coding: utf-8 -*-
from utils_w2v import TASK
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def main():
    if not os.path.exists('../data'):
        print('No source data file!')
        print("Need folder named 'data' unzipped from 'data.zip'!")
        return

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    np.random.seed(100)
    torch.cuda.manual_seed(100)
    rate = 0.7
    keyword_num_list = [1, 3, 5, 10, 15, 20, 30, 50, 100]

    tar = TASK()
    tar.get_corpus()
    tar.get_model()
    tar.train_frequency(rate)

    print('\n计算准确率')
    acc_list = []
    for keyword_num in keyword_num_list:
        average_class_dic = tar.average_class_vector(keyword_num)
        res = tar.acc(keyword_num, rate, average_class_dic)
        acc_list.append(round(res, 3))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('关键词个数与准确率间的关系')
    ax.set_xlabel('关键词个数')
    ax.set_ylabel('准确率')
    plt.plot(keyword_num_list, acc_list, color='black', markerfacecolor='r', marker='o')
    for a, b in zip(keyword_num_list, acc_list):
        plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)
    plt.show()


if __name__ == '__main__':
    main()
