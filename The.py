import sys

import cv2

import numpy as np

import operator

from utilities import *

from work import *

import tensorflow.keras as keras

import tensorflow as tf

import keras.models as a


# Image Input and basic Pre-processing
file_name = "sudoku.jpeg"
if len(sys.argv) > 1:
    file_name = sys.argv[1]
proc, im = get_image(file_name)

# Analyse the Image to get contours and Extract a Sudoku Grid
imw, im = get_puzzle(proc, im)

# Image thresholding and Pre-processing
proc = get_out(imw)

# Divide the Image (Sudoku Grid) and get each cell separately
im2 = small_sq(proc)

# Load the trained model
model = a.load_model("model")
put = np.zeros((9, 9), dtype=int)

for i in range(81):
    aaa = im2[i]
    ww = aaa.shape[0]
    hh = aaa.shape[1]
    inarea = ww*hh
    ww = ww//2
    hh = hh//2

    # Get the contour representing the digit
    gt, cc = get_num_contour(aaa, ww, hh)
    print(gt)

    if gt == -1:
        # print("skip")
        continue

    # Using the above information to extract the digit
    x, y, w, h = cv2.boundingRect(cc[gt])
    ab = aaa[y:y+h, x:x+w]
    ww = ab.shape[1]
    hh = ab.shape[0]
    farea = ww*hh
    #if (farea/inarea) < 0.15 or (farea/inarea) > 0.85:
        #continue

    # We keep the digit height constant at 20px
    ww = 22*ww//hh
    rw = ww//2
    rh = ww//2

    if ww % 2 == 1:
        rh = rh+1

    ab = cv2.resize(ab, (ww, 22))
    ab = cv2.copyMakeBorder(ab, 3, 3, 14-rw, 14-rh, cv2.BORDER_CONSTANT)
    # ab = cv2.GaussianBlur(ab,(3,3),0)

    ab.astype('float32')
    ab = ab / 255.0

    # Load the saved prediction model and predict the digits
    pred = model.predict(ab.reshape(1, 28, 28, 1), batch_size=1)
    ans = pred.argmax()
    if ans == 0:
        ind = np.argsort(pred)
        ans = ind[1]
    if ans == 7:
        if (ww/22) < 0.5:
            ans = 1
    if ans == 1:
        if (ww/22) > 0.55:
            ans = 7
    put[(i // 9)][(i % 9)] = ans

print(put)
pfix = np.array(put)
anss = get_ans(put)
print(anss)
ww = imw.shape[0] // 9
hh = imw.shape[1] // 9
for i in range(9):
    for j in range(9):
        if pfix[i][j] != 0:
            continue
        asize = cv2.getTextSize(str(anss[i][j]),cv2.FONT_HERSHEY_SIMPLEX,1,2)[0]
        xx = (hh-asize[0])//2
        yy = (ww+asize[1])//2
        imw = cv2.putText(imw,str(anss[i][j]),(hh*j+xx,ww*i+yy),cv2.FONT_HERSHEY_SIMPLEX,1,(0,198,0),2)
cv2.imshow("read",imw)
cv2.waitKey(0)
