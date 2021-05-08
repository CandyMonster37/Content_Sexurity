from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import cv2 as cv
import os
import pickle

bin_n = 16  # Number of bins
affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR

label_2_id = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
              '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}

id_2_label = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
              5: '5', 6: '6', 7: '7', 8: '8', 9: '9'}


def dataset(root_path='../digit_data', mode='train'):
    path = root_path + '/' + mode
    labels = os.listdir(path)
    x = []  # 图片hog值
    y = []  # 图片所属的类
    for label in labels:
        img_path = path + '/' + label
        files = os.listdir(img_path)
        for item in files:
            file = img_path + '/' + item
            img = cv.imread(file, 0)  # 读灰度图
            img_ = deskew(img)  # 做图像校正
            hog_ = hog(img_)
            x.append(hog_)
            y.append(label_2_id[label])
    return x, y


def set_data(root_path='../digit_data'):
    check = root_path + '/dataset'
    if os.path.exists(check):
        path_dir = check + '/'
        ta = path_dir + '/' + 'train_data.pkl'
        te = path_dir + '/' + 'test_data.pkl'
        with open(ta, 'rb') as f:
            tar = pickle.load(f)
            train_x, train_y = tar[0], tar[1]
        with open(te, 'rb') as f:
            tar = pickle.load(f)
            test_x, test_y = tar[0], tar[1]
        print('dataset loaded successfully!')
    else:
        mode = 'train'
        train_x, train_y = dataset(root_path='../digit_data', mode=mode)
        mode = 'test'
        test_x, test_y = dataset(root_path='../digit_data', mode=mode)
        os.mkdir(check)
        path_dir = check + '/'
        ta = path_dir + '/' + 'train_data.pkl'
        te = path_dir + '/' + 'test_data.pkl'
        with open(ta, 'wb') as f:
            tar = (train_x, train_y)
            pickle.dump(tar, f)
        with open(te, 'wb') as f:
            tar = (test_x, test_y)
            pickle.dump(tar, f)
        print('dataset built successfully!')
    return train_x, train_y, test_x, test_y


def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist


def deskew(img):
    SZ = min(img.shape[0], img.shape[1])
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img


def train(train_x, train_y):
    clf_ = 'model.pkl'
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


def predict(test_x, clf):
    return clf.predict(test_x)


def measure(test_y, pre_y):
    TN = 0
    TP = 0
    FN = 0
    FP = 0
    ALL = len(test_y)
    kinds_test = {}
    kinds_pre = {}
    for no in range(len(test_y)):
        if test_y[no] not in kinds_test.keys():
            kinds_test[test_y[no]] = 1
        else:
            kinds_test[test_y[no]] += 1
        if pre_y[no] not in kinds_pre.keys():
            kinds_pre[pre_y[no]] = 1
        else:
            kinds_pre[pre_y[no]] = 1
        if test_y[no] == pre_y[no]:
            TP += 1
            kinds_pre[pre_y[no]] -= 1
            kinds_test[test_y[no]] -= 1
    for item in kinds_test.keys():
        FN += kinds_test[item]
    for item in kinds_pre.keys():
        FP += kinds_pre[item]
    TN = ALL - TP - FN - FP

    acc = (TP + TN) / ALL
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    f1 = 2 * precision * recall / (precision + recall)
    return acc, recall, precision, f1


def person_detect(file):
    x = []
    img = cv.imread(file, 0)
    shape = img.shape
    img_x = shape[0]
    img_y = shape[1]
    s_x = img_x / 28
    s_y = img_y / 28
    re_shape = cv.resize(img, None, fx=s_x, fy=s_y, interpolation=cv.INTER_CUBIC)

    img_ = deskew(re_shape)  # 做图像校正
    hog_ = hog(img_)
    x.append(hog_)

    clf_ = 'model.pkl'
    if os.path.exists(clf_):
        with open(clf_, 'rb') as f:
            clf = pickle.load(f)
    pre_y = predict(test_x=x, clf=clf)
    pre = id_2_label[pre_y]
    print('预测结果为：', pre)


def main():
    # file = 'test.png'
    # person_detect(file=file)
    path = '../digit_data'
    train_x, train_y, test_x, test_y = set_data(root_path=path)
    clf = train(train_x=train_x, train_y=train_y)
    pre_y = predict(test_x=test_x, clf=clf)
    acc, recall, pre, f1 = measure(test_y=test_y, pre_y=pre_y)
    print('accuracy: {0}\nrecall: {1}\nprecision: {2}\nf1 measure: {3}'.format(acc, recall, pre, f1))


if __name__ == '__main__':
    main()
