import os
import jieba
import nltk


class TextClassification:
    def __init__(self, root_path='../data'):
        self.all_words = {}
        self.root_path = root_path
        self.label_list = os.listdir(self.root_path)
        self.stopwords = []
        self.train_set = {}
        self.test_set = {}
        if os.path.exists('stopwords.txt'):
            with open('stopwords.txt', 'r', encoding='utf8') as f:
                cont = f.read()
                self.stopwords = cont.split()

    def proc_data(self, rate=0.7):
        """
        划分训练集和测试集
        :param rate: 训练集占比
        """
        for label in self.label_list:
            c_path = self.root_path + '/' + label
            if os.path.isdir(c_path):
                file_list = os.listdir(c_path)
                self.train_set[label] = []
                self.test_set[label] = []
                nFile = len(file_list)
                for filename in file_list:
                    file = c_path + '/' + filename
                    with open(file, 'r', encoding='utf8') as f:
                        cont = f.read()
                        cont = "".join(cont.split())
                    word_cut = jieba.lcut(cont, cut_all=False)
                    word_cut_ = []
                    for word in word_cut:
                        if word in self.stopwords:
                            # 停用词，一些标点符号和语气词等
                            continue
                        else:
                            word_cut_.append(word)
                            if word in self.all_words.keys():
                                self.all_words[word] += 1
                            else:
                                self.all_words[word] = 1
                    # 70%作为训练集，剩下30%作为测试集
                    if file_list.index(filename) < int(rate * nFile):
                        self.train_set[label].append(word_cut_)
                    else:
                        self.test_set[label].append(word_cut_)

    def words_dict(self, delete_n):
        n = 0
        word_features_list = []
        all_words_list = sorted(self.all_words.items(), key=lambda item: item[1], reverse=True)
        for t in range(delete_n, len(self.all_words), 1):
            if n > 1000:
                break
            else:
                n += 1
                word_features_list.append(all_words_list[t][0])
        return word_features_list

    @staticmethod
    def doc_features(doc, word_features):
        doc_words = set(doc)
        features = {}
        for word in word_features:
            features['contains{}'.format(word)] = (word in doc_words)
        return features

    def Classfication_Bayes(self, word_features):
        train_data = []
        for label in self.train_set.keys():
            for vec in self.train_set[label]:
                feature = self.doc_features(vec, word_features)
                tmp = (feature, label)
                train_data.append(tmp)

        test_data = []
        for label in self.test_set.keys():
            for vec in self.test_set[label]:
                feature = self.doc_features(vec, word_features)
                tmp = (feature, label)
                test_data.append(tmp)

        classifier = nltk.NaiveBayesClassifier.train(train_data)
        test_acc = nltk.classify.accuracy(classifier, test_data)
        return round(test_acc, 4)
