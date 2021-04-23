from nb_utils import TextClassification
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


def main():
    if not os.path.exists('../data'):
        print('No source data file!')
        print("Need folder named 'data' unzipped from 'data.zip'!")
        return

    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    rate = 0.7

    tar = TextClassification()
    tar.proc_data(rate)

    test_error_list = []
    best_acc = 0
    deleteNs = range(0, 600, 20)
    index = 0

    print('探究在朴素贝叶斯模型上，去除高频词个数与准确率间的关系')
    for n in tqdm(deleteNs):
        word_feature_list = tar.words_dict(n)
        acc_epoch = tar.Classfication_Bayes(word_feature_list)
        test_error_list.append(acc_epoch)
        if best_acc < acc_epoch:
            best_acc = acc_epoch
            index = deleteNs.index(n)
    print('best accuracy: ', best_acc)
    print('The corresponding number of removed high-frequency words: ', deleteNs[index])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title('去除高频词个数与准确率间的关系')
    ax.set_xlabel('去除高频词个数')
    ax.set_ylabel('准确率')
    plt.plot(deleteNs, test_error_list)
    # for a, b in zip(deleteNs, test_error_list):
    #    plt.text(a, b, (a, b), ha='center', va='bottom', fontsize=10)
    # 要是把数据点打出来的话就太密集了，不好看
    plt.show()


if __name__ == '__main__':
    main()
