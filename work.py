import numpy as np

import cv2


def check_row(arr, x):
    s = set()
    f = 0
    for i in arr[x]:
        if i == 0:
            continue
        if i in s:
            f = 1
            break
        s.add(i)
    if f == 1:
        return False
    else:
        return True


def check_col(arr, x):
    s = set()
    f = 0
    for i in arr[:, x]:
        if i == 0:
            continue
        if i in s:
            f = 1
            break
        s.add(i)
    if f == 1:
        return False
    else:
        return True


def check_box(arr, x, y):
    s = set()
    f = 0
    x = x // 3
    y = y // 3
    for i in range(x*3, (x+1)*3):
        for j in range(y*3, (y+1)*3):
            if arr[i][j] == 0:
                continue
            if arr[i][j] in s:
                f = 1
                break
            s.add(arr[i][j])
    if f == 1:
        return False
    else:
        return True


ans = np.zeros((9, 9), dtype=int)
fix = np.zeros((9, 9), dtype=int)
def init_work(arr):
    global ans, fix
    ans = arr
    for i in range(9):
        for j in range(9):
            if arr[i][j] == 0:
                continue
            else:
                fix[i][j] = 1


def get_ans(arr):
    init_work(arr)
    a = get_work(arr, 0, 0)
    global ans
    return ans


def get_work(arr, a, b):
    f = 0
    t = 0
    if a >= 9:
        global ans
        ans = arr
        return True
    for i in range(1, 10):
        if fix[a][b]:
            break
        arr[a][b] = i
        cr = check_row(arr, a)
        cl = check_col(arr, b)
        cb = check_box(arr, a, b)
        if cr and cl and cb:
            aa = a
            bb = b + 1
            if bb == 9:
                aa = a + 1
                bb = 0
            f = get_work(arr, aa, bb)
            if f:
                return True
    if fix[a][b]:
        aa = a
        bb = b + 1
        if bb == 9:
            aa = a + 1
            bb = 0
        f = get_work(arr, aa, bb)
        if f:
            return True
    if not fix[a][b]:
        arr[a][b] = 0
    return False
