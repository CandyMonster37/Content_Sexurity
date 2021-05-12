from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from utils import *
import os
import pickle


def train(train_x, train_y):
    clf_ = './data/model_svm.pkl'
    if os.path.exists(clf_):
        with open(clf_, 'rb') as f:
            clf = pickle.load(f)
        print('model loaded successfully!')
    else:
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        print('model built successfully')
        clf.fit(train_x, train_y)
        with open(clf_, 'wb') as f:
            pickle.dump(clf, f)
    return clf


def train_():
    path = '../digit_data'
    train_x, train_y, test_x, test_y = set_data(root_path=path)
    clf = train(train_x=train_x, train_y=train_y)
    pre_y = predict(test_x=test_x, clf=clf)
    acc, recall, pre, f1 = measure(test_y=test_y, pre_y=pre_y)
    print('accuracy: {0}\nrecall: {1}\nprecision: {2}\nf1 measure: {3}'.format(acc, recall, pre, f1))


def test_():
    file = 'testimg'
    detect(file_path=file, clf_='./data/model_svm.pkl', dir_=True)


if __name__ == '__main__':
    train_()
    test_()
