import numpy as np
from ps_hepers.helpers import imread_from_rep, add_gaussian_noise, imread, imsave, imfix_scale

"""
Problem Set - 0
Problems: https://docs.google.com/document/d/1PO9SuHMYhx6nDbB38ByB1QANasP1UaEiXaeGeHmp3II/pub?embedded=true
"""


def p1():
    # a
    wide_img = imread_from_rep('frizzy')
    tall_img = imread_from_rep('mona-lisa')
    imsave(wide_img, 'output/ps0-1-a-1.png')
    imsave(tall_img, 'output/ps0-1-a-2.png')


def p2():
    lena = imread_from_rep('lena')
    # a
    # RGB = 0 1 2 then by swapping R & B you get [2 1 0]
    imsave(lena[:, :, [2, 1, 0]], 'output/ps0-2-a-1.png')
    # b
    imsave(lena[:, :, [1]], 'output/ps0-2-b-1.png')
    # c
    imsave(lena[:, :, [0]], 'output/ps0-2-c-1.png')
    # d
    """
    Which looks more like what you’d expect a monochrome image to look like?
    In case of Lena's image green channel provides a better gray scale image than the red channel. This can because of 
    low saturation of red in the image than that of green.
    But what might not be a coincidence is a standard way of converting RGB images to gray scale gives more weight to
    the green channel.
    Y = 0.299 R + 0.587 G + 0.114 B (formula used by open cv's cv.COLOR_RGB2GRAY conversion)
    This is probably because of the way human's vision system is designed to perceive colors. 
    Ref: https://respuestas.me/q/deteccion-de-la-vision-humana-del-punto-de-luz-debil-parpadeante-o-en-movim-60506457491
    http://cadik.posvete.cz/color_to_gray_evaluation/ 
    
    
    Would you expect a computer vision algorithm to work on one better than the other?
    Of course, yes. A segmentation algorithm will produce different results over different channels of color. For 
    example, an apple (red) and a lemon(yellow = green + red) might just look alike in red channel but are very 
    different in green channel.   
    """


def p3():
    # a
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
    # a
    print('max of pixels', np.max(im))
    print('min of pixels', np.min(im))
    print('average of pixels', np.average(im))
    print('std of pixels', np.std(im))
    # b
    im_b = ((im - np.average(im)) / np.std(im)) * 10.0 + np.average(im)
    imsave(im_b, 'output/ps0-4-b-1.png')
    """
    This step (b) just decreased the contrast as 10.0 < np.std(im)(=52.8) 
    """
    # c
    shift = 2
    shift_col_idx = np.append(np.arange(shift, im.shape[1]), np.arange(0, shift))
    imsave(im[:, shift_col_idx], 'output/ps0-4-c-1.png')
    # d
    sub_im = im.astype(np.float32) - im[:, shift_col_idx].astype(np.float32)
    imsave(imfix_scale(sub_im), 'output/ps0-4-d-1.png')
    """
    What do negative pixel values mean anyways? (in output of #d)
    Negative pixel values occur when I(x,y)<I(x+shift, y). So, a negative value indicates a edge with increasing 
    intensity towards x direction.  
    """


def p5():
    im = imread_from_rep('lena')
    # a
    sigma = 5.0
    im_green = np.ndarray.copy(im)
    im_green[:, :, 1] = add_gaussian_noise(im_green[:, :, 1], sigma)
    imsave(im_green, 'output/ps0-5-a-1.png')
    # b
    im_blue = np.ndarray.copy(im)
    im_blue[:, :, 0] = add_gaussian_noise(im_blue[:, :, 0], sigma)
    imsave(im_blue, 'output/ps0-5-b-1.png')
    # c
    """
    Which looks better?
    The image with noise in the blue channel looks better than the one with green noise.
    
     Why?
     This definitely has something to human color perception. The number of cones sensitive to color blue is a lot 
     lesser than that of color green. 
     The cones distribution in human eye is (Red - 64%, Green - 32%, Blue - 2%)
     So, humans are less sensitive to noise in blue channel.
    """


if __name__ == '__main__':
    p1()
    p2()
    p3()
    p4()
    p5()
