import numpy as np
import cv2 as cv
from math import ceil


def change(s_x, s_y, img, fill=False, save=False):
    """
    :param s_x: 宽度比例
    :param s_y: 高度比例
    :param img: 图片对象
    :param fill: 窗口是否适应图片大小，默认为否
    :param save: 是否保存结果，默认为否
    """
    h, w = img.shape[:2]
    A1 = np.array([[s_x, 0, 0], [0, s_y, 0]], np.float32)
    if fill:
        dsize = (ceil(s_x * w), ceil(s_y * h))
        changed = cv.warpAffine(img, A1, dsize)
    else:
        changed = cv.warpAffine(img, A1, (w, h))
    cv.imshow("changed", changed)
    cv.imshow('original', img)
    cv.waitKey()
    cv.destroyAllWindows()
    if save:
        cv.imwrite('after.png', changed)


def main():
    src = 'test.jpg'
    img = cv.imread(src, cv.IMREAD_COLOR)
    s_x = 0.5
    s_y = 0.5
    change(s_x, s_y, img, fill=True, save=False)


if __name__ == '__main__':
    main()
