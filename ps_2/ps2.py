from disparity_ssd import disparity_ssd
from disparity_ncorr import disparity_ncorr
from ps_hepers.helpers import imread_from_rep, imshow, imsave, imfix_scale, np_save, np_load, add_gaussian_noise

"""
Problem Set - 2
Problems: https://docs.google.com/document/d/1WcljLaRxL-Pj3VWYz7JtYysYoZtRZoLIrTG2x48uVWE/pub?embedded=true
"""


def p1():
    left = imread_from_rep('pair0-L', grey_scale=True)
    right = imread_from_rep('pair0-R', grey_scale=True)
    d_l = disparity_ssd(left, right)
    d_r = disparity_ssd(right, left)
    imshow([left, right, d_l, d_r], ['left', 'right', 'd_l', 'd_r'])
    imsave(d_l, 'output/ps2-1-a-1.png')
    imsave(d_r, 'output/ps2-1-a-2.png')


def p2():
    left_filename = 'pair1-L copy'
    right_filename = 'pair1-R copy'
    left = imread_from_rep(left_filename, grey_scale=True)
    right = imread_from_rep(right_filename, grey_scale=True)
    d_maps = np_load(4, 'objects/p2_disparities(%s,%s).npy' % (left_filename, right_filename))
    (d_l, d_r) = (d_maps[2], d_maps[3]) if d_maps is not None else (
        disparity_ssd(left, right), disparity_ssd(right, left))
    imshow([left, right, imfix_scale(d_l), imfix_scale(d_r), imread_from_rep('pair1-D_L', grey_scale=True),
            imread_from_rep('pair1-D_R', grey_scale=True)],
           ['left', 'right', 'd_l', 'd_r', 'ground truth l', 'ground truth l'], shape=(3, 2))
    np_save([d_l, d_r],
            'objects/p2_disparities(%s,%s).npy' % (left_filename, right_filename))
    imsave(imfix_scale(d_l), 'output/ps2-2-a-1.png')
    imsave(imfix_scale(d_r), 'output/ps2-2-a-2.png')


def test_disparity_on_noise(disparity_compute, noise_type, object_name_prefix='p2'):
    left_filename = 'pair1-L copy'
    right_filename = 'pair1-R copy'
    left = imread_from_rep(left_filename, grey_scale=True)
    right = imread_from_rep(right_filename, grey_scale=True)
    d_maps = np_load(2, 'objects/%s_disparities(%s,%s).npy' % (object_name_prefix, left_filename, right_filename))
    (d_l, d_r) = (d_maps[0], d_maps[1]) if d_maps is not None else (
        disparity_compute(left, right), disparity_compute(right, left))
    if noise_type == 'gaussian':
        left = add_gaussian_noise(left, 15)
    elif noise_type == 'contrast':
        left = left * 1.1
        left[left > 255] = 255
        left = left.astype('uint8')
    (d_l_noise, d_r_noise) = (disparity_compute(left, right), disparity_compute(right, left))
    imshow([left, right, imfix_scale(d_l), imfix_scale(d_r), imfix_scale(d_l_noise), imfix_scale(d_r_noise),
            imread_from_rep('pair1-D_L', grey_scale=True),
            imread_from_rep('pair1-D_R', grey_scale=True)],
           ['left', 'right', 'd_l', 'd_r', 'd_l noise', 'd_r noise', 'ground truth l', 'ground truth l'], shape=(4, 2))
    imsave(d_l_noise, 'output/ ps2-3-a-1.png')
    imsave(d_r_noise, 'output/ ps2-3-a-2.png')


def p3():
    test_disparity_on_noise(disparity_ssd, 'gaussian')
    test_disparity_on_noise(disparity_ssd, 'contrast')


def p4():
    left_filename = 'pair1-L copy'
    right_filename = 'pair1-R copy'
    left = imread_from_rep(left_filename, grey_scale=True)
    right = imread_from_rep(right_filename, grey_scale=True)
    d_maps = np_load(2, 'objects/p4_disparities(%s,%s).npy' % (left_filename, right_filename))
    (d_l, d_r) = (d_maps[0], d_maps[1]) if d_maps is not None else (
        disparity_ncorr(left, right), disparity_ncorr(right, left))
    imshow([left, right, d_l, d_r, imread_from_rep('pair1-D_L', grey_scale=True),
            imread_from_rep('pair1-D_R', grey_scale=True)],
           ['left', 'right', 'd_l', 'd_r', 'ground truth l', 'ground truth l'], shape=(3, 2))
    np_save([d_l, d_r],
            'objects/p4_disparities(%s,%s).npy' % (left_filename, right_filename))
    imsave(imfix_scale(d_l), 'output/ps2-4-a-1.png')
    imsave(imfix_scale(d_r), 'output/ps2-4-a-2.png')


def p4_with_noise():
    test_disparity_on_noise(disparity_ncorr, 'gaussian', 'p4')
    test_disparity_on_noise(disparity_ncorr, 'contrast', 'p4')


def p5():
    left_filename = 'pair2-L copy'
    right_filename = 'pair2-R copy'
    left = imread_from_rep(left_filename, grey_scale=True)
    right = imread_from_rep(right_filename, grey_scale=True)
    d_maps = np_load(2, 'objects/p5_disparities(%s,%s).npy' % (left_filename, right_filename))
    (d_l, d_r) = (d_maps[0], d_maps[1]) if d_maps is not None else (
        disparity_ncorr(left, right), disparity_ncorr(right, left))
    imshow([left, right, d_l, d_r, imread_from_rep('pair2-D_L', grey_scale=True),
            imread_from_rep('pair2-D_R', grey_scale=True)],
           ['left', 'right', 'd_l', 'd_r', 'ground truth l', 'ground truth l'], shape=(3, 2))
    np_save([d_l, d_r],
            'objects/p5_disparities(%s,%s).npy' % (left_filename, right_filename))
    imsave(imfix_scale(d_l), 'output/ps2-5-a-1.png')
    imsave(imfix_scale(d_r), 'output/ps2-5-a-2.png')


def compare_ssd_ncorr():
    left_filename = 'pair2-L copy'
    right_filename = 'pair2-R copy'
    left = imread_from_rep(left_filename, grey_scale=True)
    right = imread_from_rep(right_filename, grey_scale=True)
    d_l_ssd = disparity_ssd(left, right)
    d_r_ssd = disparity_ssd(right, left)
    d_l_ncorr = disparity_ncorr(left, right)
    d_r_ncorr = disparity_ncorr(right, left)
    imshow([left, right, d_r_ssd, d_r_ssd, d_l_ncorr, d_r_ncorr,
            imread_from_rep('pair2-D_L', grey_scale=True), imread_from_rep('pair2-D_R', grey_scale=True)],
           ['left', 'right', 'd_r_ssd', 'd_r_ssd', 'd_l_ncorr', 'd_r_ncorr', 'ground truth l', 'ground truth r'],
           shape=(4, 2))


if __name__ == '__main__':
    # p1()
    # p2()
    # p3()
    # p4()
    # p4_with_noise()
    # p5()
    compare_ssd_ncorr()
