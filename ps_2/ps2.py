import os
import numpy as np
import cv2
from disparity_ssd import disparity_ssd
from disparity_ncorr import disparity_ncorr
from ps_hepers.helpers import imread_from_rep, imshow, imsave, imfix_scale, np_save

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
    left = imread_from_rep('pair1-L copy', grey_scale=True)
    right = imread_from_rep('pair1-R copy', grey_scale=True)
    d_l = disparity_ssd(left, right)
    d_r = disparity_ssd(right, left)
    imshow([left, right, d_l, d_r])
    np_save([d_l, d_r], 'objects/pair1_disparities(L,R)')
    imsave(imfix_scale(d_l), 'output/ps2-2-a-1.png')
    imsave(imfix_scale(d_r), 'output/ps2-2-a-2.png')




# p1()
p2()
