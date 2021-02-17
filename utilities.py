import cv2

import numpy as np

import operator


def distance_between(p1, p2):
    a = p2[0] - p1[0]
    b = p2[1] - p1[1]
    return np.sqrt((a ** 2) + (b ** 2))


def small_sq(arr):
    width = arr.shape[0] // 9
    height = arr.shape[1] // 9
    wise = []
    for i in range(9):
        for j in range(9):
            wise.append(arr[i*width:(i+1)*width, j*height:(j+1)*height])
    return wise


def get_image(str):
    im = cv2.imread(str)
    # im = cv2.resize(im, (900, 600))
    # proc = cv2.GaussianBlur(im, (9, 9), 0)
    proc = cv2.GaussianBlur(im, (9, 9), 0)
    proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)
    # proc = cv2.resize(proc, (450, 300))
    # im = cv2.resize(im, (450, 300))
    # kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], np.uint8)
    # proc = cv2.dilate(proc, kernel)
    return proc, im


def get_puzzle(x, im):
    contours, h = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = contours[0]

    bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
    top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

    bottom_right = contours[0][bottom_right][0]
    bottom_left = contours[0][bottom_left][0]
    top_right = contours[0][top_right][0]
    top_left = contours[0][top_left][0]

    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')
    side = max([distance_between(bottom_right, top_right),
                distance_between(top_left, bottom_left),
                distance_between(bottom_right, bottom_left),
                distance_between(top_left, top_right)])

    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
    m = cv2.getPerspectiveTransform(src, dst)
    imw = cv2.warpPerspective(im, m, (int(side), int(side)))
    #im = cv2.resize(im, (450, 300))
    #imw = cv2.resize(imw, (450, 300))
    return imw, im


def get_out(x):
    # proc = cv2.GaussianBlur(x, (9, 9), 0)
    proc = cv2.GaussianBlur(x, (9, 9), 0)
    proc = cv2.cvtColor(proc, cv2.COLOR_BGR2GRAY)
    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    proc = cv2.bitwise_not(proc, proc)
    return proc


def get_num_contour(a, ww, hh):
    cc, _ = cv2.findContours(a, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    a = cc
    min_dist = 10000
    gt = -1
    for k in range(len(a)):
        for j in a[k]:
            dist = distance_between(j[0], [ww, hh])
            min_dist = min(min_dist, dist)
        if min_dist <= (ww * 2 // 5):
            gt = k
            break
    return gt, cc
