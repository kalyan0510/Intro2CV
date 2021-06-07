import cv2 as cv
import numpy as np


def imread(filename, channels=[0, 1, 2], greyScale=False):
    img = cv.imread(filename)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if greyScale else img[:, :, channels]


def imreadFromRep(imageName):
    return imread('../images/' + imageName + '.png')


def imsave(img, fileName):
    cv.imwrite(fileName, img)


def imFixScale(img):
    return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 256)


def p1():
    wideImg = imreadFromRep('mona-lisa')
    tallImg = imreadFromRep('frizzy')
    imsave(wideImg, 'output/ps0-1-a-1.png')
    imsave(tallImg, 'output/ps0-1-a-2.png')


def p2():
    lena = imreadFromRep('lena')
    # BGR = 0 1 2 then by swapping R & B you get [2 1 0]
    imsave(lena[:, :, [2, 1, 0]], 'output/ps0-2-a-1.png')
    imsave(lena[:, :, [1]], 'output/ps0-2-b-1.png')
    imsave(lena[:, :, [0]], 'output/ps0-2-c-1.png')


def p3():
    p2()
    im1 = imread('output/ps0-2-b-1.png', greyScale=True)
    im2 = imread('output/ps0-2-c-1.png', greyScale=True)
    rows = im2.shape[0]
    cols = im2.shape[1]
    rowSlice = slice(rows // 2 - 50, rows // 2 + 50, )
    colSlice = slice(cols // 2 - 50, cols // 2 + 50)
    im2[rowSlice, colSlice] = im1[rowSlice, colSlice]
    imsave(im2, 'output/ps0-3-a-1.png')


def p4():
    im = imread('output/ps0-2-b-1.png', greyScale=True)
    print('4 a')
    print('max of pixels', np.max(im))
    print('min of pixels', np.min(im))
    print('max of pixels', np.average(im))
    print('max of pixels', np.std(im))
    # b
    imB = ((im - np.average(im)) / np.std(im)) * 10.0 + np.average(im)
    imsave(imB, 'output/ps0-4-b-1.png')
    # c
    shift = 1
    shiftColIdx = np.append(np.arange(shift, im.shape[1]), np.arange(0, shift))
    imsave(im[:, shiftColIdx], 'output/ps0-4-c-1.png')
    # d
    subIm = im - im[:, shiftColIdx]
    imsave(subIm, 'output/ps0-4-d-1.png')


def p5():
    im = imreadFromRep('lena')
    sigma1 = 10.0
    noise1 = np.random.normal(0, sigma1, im.shape[0:2])
    imGreen = np.ndarray.copy(im)
    imGreen[:, :, 1] = (imGreen[:, :, 1] + noise1)
    imsave(imGreen, 'output/ps0-5-a-1.png')
    sigma2 = 10.0
    noise2 = np.random.normal(0, sigma2, im.shape[0:2])
    imBlue = np.ndarray.copy(im)
    imBlue[:, :, 0] = (imBlue[:, :, 0] + noise2)
    imsave(imBlue, 'output/ps0-5-b-1.png')

p1()
p2()
p3()
p4()
p5()
