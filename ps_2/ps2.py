import os
import numpy as np
import cv2
from disparity_ssd import disparity_ssd
from disparity_ncorr import disparity_ncorr
from ps_hepers.helpers import imread_from_rep, imshow, imsave, imfix_scale, np_save, np_load

"""
Problem Set - 2
Problems: https://docs.google.com/document/d/1WcljLaRxL-Pj3VWYz7JtYysYoZtRZoLIrTG2x48uVWE/pub?embedded=true
"""


def p1():
    left = imread_from_rep('pair0-L', grey_scale=True)
    right = imread_from_rep('pair0-R', grey_scale=True)
    d_l = disparity_ssd(left, right)
    d_r = disparity_ssd(right, left)
    imshow([left, right, d_l, d_r])
    imsave(d_l, 'output/ps2-1-a-1.png')
    imsave(d_r, 'output/ps2-1-a-2.png')


def p2():
    left_filename = 'pair1-L copy'
    right_filename = 'pair1-R copy'
    left = imread_from_rep(left_filename, grey_scale=True)
    right = imread_from_rep(right_filename, grey_scale=True)
    d_maps = np_load(4, 'objects/pair1_disparities(L,R) main.npy')
    (d_l, d_r) = (d_maps[2], d_maps[3]) if (
            (d_maps is not None) and d_maps[0] == left_filename and d_maps[1] == right_filename) else (
        disparity_ssd(left, right), disparity_ssd(right, left))
    imshow([left, right, imfix_scale(d_l), imfix_scale(d_r), imread_from_rep('pair1-D_L', grey_scale=True),
            imread_from_rep('pair1-D_R', grey_scale=True)], shape=(3, 2))
    np_save([left_filename, right_filename, d_l, d_r], 'objects/pair1_disparities(L,R) main.npy')
    imsave(imfix_scale(d_l), 'output/ps2-2-a-1.png')
    imsave(imfix_scale(d_r), 'output/ps2-2-a-2.png')


if __name__ == '__main__':
    # p1()
    p2()
