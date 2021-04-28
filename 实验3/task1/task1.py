import numpy as np
import cv2 as cv
from math import ceil


def change(s_x, s_y, img, fill=False):
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


def main():
    src = 'test.jpg'
    img = cv.imread(src, cv.IMREAD_COLOR)
    s_x = 0.5
    s_y = 0.5
    change(s_x, s_y, img, True)


if __name__ == '__main__':
    main()
