import numpy as np
from ps_hepers.helpers import imread_from_rep, add_gaussian_noise, imread, imsave


def p1():
    wide_img = imread_from_rep('mona-lisa')
    tall_img = imread_from_rep('frizzy')
    imsave(wide_img, 'output/ps0-1-a-1.png')
    imsave(tall_img, 'output/ps0-1-a-2.png')


def p2():
    lena = imread_from_rep('lena')
    # RGB = 0 1 2 then by swapping R & B you get [2 1 0]
    imsave(lena[:, :, [2, 1, 0]], 'output/ps0-2-a-1.png')
    imsave(lena[:, :, [1]], 'output/ps0-2-b-1.png')
    imsave(lena[:, :, [0]], 'output/ps0-2-c-1.png')


def p3():
    p2()
    im1 = imread('output/ps0-2-b-1.png', grey_scale=True)
    im2 = imread('output/ps0-2-c-1.png', grey_scale=True)
    rows = im2.shape[0]
    cols = im2.shape[1]
    row_slice = slice(rows // 2 - 50, rows // 2 + 50, )
    col_slice = slice(cols // 2 - 50, cols // 2 + 50)
    im2[row_slice, col_slice] = im1[row_slice, col_slice]
    imsave(im2, 'output/ps0-3-a-1.png')


def p4():
    im = imread('output/ps0-2-b-1.png', grey_scale=True)
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
    shift_col_idx = np.append(np.arange(shift, im.shape[1]), np.arange(0, shift))
    imsave(im[:, shift_col_idx], 'output/ps0-4-c-1.png')
    # d
    sub_im = im - im[:, shift_col_idx]
    imsave(sub_im, 'output/ps0-4-d-1.png')


def p5():
    im = imread_from_rep('lena')
    sigma1 = 10.0
    im_green = np.ndarray.copy(im)
    im_green[:, :, 1] = add_gaussian_noise(im_green[:, :, 1], sigma1)
    imsave(im_green, 'output/ps0-5-a-1.png')
    sigma2 = 10.0
    im_blue = np.ndarray.copy(im)
    im_blue[:, :, 0] = add_gaussian_noise(im_blue[:, :, 0], sigma2)
    imsave(im_blue, 'output/ps0-5-b-1.png')


if __name__ == '__main__':
    p1()
    p2()
    p3()
    p4()
    p5()
