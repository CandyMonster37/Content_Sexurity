# -*- coding: utf-8 -*-
from gensim.models import word2vec, Word2Vec
from tqdm import tqdm
import numpy as np
import jieba.analyse
import os
import re


class TASK:
    def __init__(self, root_path='../data'):
        self.allPOS = ('n', 'ns', 'v', 'vn')
        self.root_path = root_path
        self.class_list = os.listdir(self.root_path)
        self.all_txt_list = []
        self.model = None
        self.class_all_words = {}

    def get_corpus(self):
        for item in self.class_list:
            class_path = self.root_path + '/' + item
            file_list = os.listdir(class_path)
            for file_ in file_list:
                filename = class_path + '/' + file_
                with open(filename, 'r', encoding='utf8') as f:
                    cont = f.read()
                cont = "".join(cont.split())
                word_list = jieba.analyse.textrank(cont, topK=None, withWeight=False, allowPOS=('n', 'ns', 'v', 'vn'))
                if len(word_list) == 0:
                    continue
                else:
                    self.all_txt_list.extend(word_list)

        result = " ".join(self.all_txt_list)
        with open('result.txt', 'w', encoding='utf8') as f:
            f.write(result)

    def get_model(self):
        if os.path.exists('my_model.model'):
            self.model = Word2Vec.load('my_model.model')
            return
        sentences = word2vec.Text8Corpus("result.txt")
        model = word2vec.Word2Vec(sentences, size=200, min_count=1)
        model.save('my_model.model')
        self.model = model

    def train_frequency(self, rate_train):
        print('\n对训练集词频统计')
        for c in tqdm(self.class_list):
            all_words = {}
            class_path = self.root_path + '/' + c
            file_list = os.listdir(class_path)
            nFile = len(file_list)
            for name in file_list[:int(nFile * rate_train)]:
                file_path = class_path + '/' + name
                with open(file_path, 'r', encoding='utf8') as f:
                    cont = f.read()
                    cont = re.sub(re.compile(r'原文网址:.*'), '', cont)
                    cont = "".join(cont.split())
                word_list = jieba.analyse.textrank(cont, topK=None, withWeight=False, allowPOS=self.allPOS)
                if len(word_list) != 0:
                    for word in word_list:
                        if word in all_words.keys():
                            all_words[word] += 1
                        else:
                            all_words[word] = 1
            self.class_all_words[c] = all_words

    def average_class_vector(self, keyword_num):
        """
        获取每个类的平均向量
        """
        average_class_dic = {}
        for c in self.class_all_words:
            all_words_list = sorted(self.class_all_words[c].items(), key=lambda item: item[1], reverse=True)
            shape_ = self.model.wv[all_words_list[0][0]].shape
            total = np.zeros(shape_[0])
            limit = min(keyword_num, len(all_words_list))
            for t in range(limit):
                total += self.model.wv[all_words_list[t][0]]
            average_class_dic[c] = total / limit
        return average_class_dic

    def acc(self, keyword_num, rate, average_class_dic):
        true = 0
        false = 0
        print('\nKeyword_num: {}'.format(keyword_num))
        for c in tqdm(self.class_list):
            class_path = self.root_path + '/' + c
            file_list = os.listdir(class_path)
            nFile = len(file_list)
            # 取30%作测试集
            for name in file_list[int(nFile * rate):]:
                file_path = class_path + '/' + name
                test_data_words = {}
                with open(file_path, 'r', encoding='utf8') as f:
                    txt = f.read()
                    txt = re.sub(re.compile(r'原文网址:.*'), '', txt)
                txt = "".join(txt.split())
                word_list = jieba.analyse.textrank(txt, topK=None, withWeight=False, allowPOS=('n', 'ns', 'v', 'vn'))
                for word in word_list:
                    if word in test_data_words.keys():
                        test_data_words[word] += 1
                    else:
                        test_data_words[word] = 1
                test_words_list = sorted(test_data_words.items(), key=lambda item: item[1], reverse=True)
                if len(test_words_list) == 0:
                    continue
                shape_ = self.model.wv[test_words_list[0][0]].shape
                total = np.zeros(shape_[0])
                limit = min(keyword_num, len(test_words_list))
                for t in range(limit):
                    total += self.model.wv[test_words_list[t][0]]
                average_test_vector = total / limit
                pre = predict(average_class_dic, data=average_test_vector)
                if pre == c:
                    true += 1
                else:
                    false += 1
        return true / (true + false)


def predict(average_class_dic, data):
    sim = {}
    for c in average_class_dic:
        sim[c] = cos_sim(data, average_class_dic[c])
    test_word_list = sorted(sim.items(), key=lambda item: item[1], reverse=True)
    return test_word_list[0][0]


def cos_sim(vector_a, vector_b):
    a_norm = np.linalg.norm(vector_a)
    b_norm = np.linalg.norm(vector_b)
    sim = np.dot(vector_a, vector_b) / (a_norm * b_norm)
    return sim
