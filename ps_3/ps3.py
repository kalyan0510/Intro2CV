import random
import sys

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from ps_3.helper import construct_a_mat, solve_for_m_svd, calculate_avg_residual, solve_for_m_leastsq, solve_for_f_svd, \
    fix_f_rank, get_epipolar_line_ends, get_transformation_mat_2d
from ps_hepers.helpers import read_points, imshow, imread, imsave

"""
Problem Set - 3
Problems: https://docs.google.com/document/d/1XsW9k_exgVwCy6CdwgUV3wLKwmFliVdmfAH74Ba4drc/pub?embedded=true
"""


def p1():
    pts_proj = read_points('input/pts2d-norm-pic_a.txt')
    pts_world = read_points('input/pts3d-norm.txt')
    a = construct_a_mat(pts_world, pts_proj)
    w, v = np.linalg.eig(np.matmul(a.T, a))
    m = v[:, w.argmin()].reshape((3, 4))
    print("M matrix: \n%s" % m)
    pts_world_mat = np.append(np.asarray(pts_world), np.ones((len(pts_world), 1)), axis=1).T
    pts_proj_est = np.matmul(m, pts_world_mat)
    pts_proj_est_non_homo = pts_proj_est.T[:, 0:2] / pts_proj_est.T[:, [2, 2]]
    residual = np.linalg.norm(pts_proj_est_non_homo - np.asarray(pts_proj))
    print('Estimated projection of pt corresponding to %s is %s' % (pts_proj[-1], tuple(pts_proj_est_non_homo[-1])))
    print('ssd_error = %s\nresidual = %s' % (residual ** 2, residual))


def p1_b(debug=True, on_normalized_set=False):
    np.set_printoptions(threshold=sys.maxsize)
    if on_normalized_set:
        pts_proj = read_points('input/pts2d-norm-pic_a.txt')
        pts_world = read_points('input/pts3d-norm.txt')
    else:
        pts_proj = read_points('input/pts2d-pic_b.txt')
        pts_world = read_points('input/pts3d.txt')
    k_sizes = range(1, 16)
    num_tests = 100
    residuals = np.zeros((num_tests, len(k_sizes)))
    (res_min, best_m) = (np.PINF, None)
    for i in range(num_tests):
        for k_i, k in zip(range(len(k_sizes)), k_sizes):
            indexes = random.sample(range(0, len(pts_world)), k)
            m = solve_for_m_svd([pts_world[i] for i in indexes], [pts_proj[i] for i in indexes])
            test_set = range(20)#random.sample([i for i in range(0, len(pts_world)) if i not in indexes], 4)  #
            residual = calculate_avg_residual(m, [pts_world[i] for i in test_set], [pts_proj[i] for i in test_set])
            (res_min, best_m) = (min(residual, res_min), m if residual < res_min else best_m)
            residuals[i][k_i] = residual
    # plot residuals
    plt.plot(k_sizes, residuals.mean(axis=0))
    plt.xlabel("Set size K (used to compute M)")
    plt.ylabel("Average Residual")
    plt.xticks(k_sizes)
    plt.show()
    if debug:
        print("Residuals (each row represents avg residual for k size sets k = %s ) :\n%s" % (k_sizes, residuals))
        print(
            "Average residuals over all random tests  for k size sets k = %s: \n%s" % (k_sizes, residuals.mean(axis=0)))
        print("Best M matrix and with residual (%s) \n%s" % (res_min, best_m))
        # imshow([residuals, residuals.mean(axis=0)[np.newaxis, :][[0] * 10, :]],
        #        ['residuals for each k size and test', 'residuals averaged over all tests'],
        #        interpolation='nearest',
        #        shape=(1, 2),
        #        sup_title='Fancy way to see the avg residuals')
    """
    Results: 
    When sampled over many trials, as K increased the average residual decreased. That means, with help of more 
    correspondences, we just got a better estimation of calibration. 
    The effects of over-constraining can be seen in few random trials where residuals increased as K size increased, but
    this could be only due to few correspondences with error.
    Although, overdetermined systems might not have best solutions, I guess the correspondences we have are
    homoschedastic and so, with bigger K sizes the calibration just got better.           
    """
    return best_m, res_min


def p1_c():
    m, residual = p1_b(debug=False, on_normalized_set=True)
    # k, r = scipy.linalg.rq(m[:,:3])
    # print(k/k[2,2])
    cam_center = -np.matmul(np.linalg.inv(m[:, 0:3]), m[:, 3])
    print('Camera Center:\n%s' % cam_center)


def p1_svd_vs_ols():
    pts_proj = read_points('input/pts2d-norm-pic_a.txt')
    pts_world = read_points('input/pts3d-norm.txt')
    m_svd = solve_for_m_svd(pts_world, pts_proj)
    m_ols = solve_for_m_leastsq(pts_world, pts_proj)
    print('residuals with svd: %s' % calculate_avg_residual(m_svd, pts_world, pts_proj))
    print('residuals with least squares: %s' % calculate_avg_residual(m_ols, pts_world, pts_proj))


def read_pic2_im_n_points():
    pts_im_a = read_points('input/pts2d-pic_a.txt')
    pts_im_b = read_points('input/pts2d-pic_b.txt')
    im_a = imread('input/pic_a.jpg').copy()
    im_b = imread('input/pic_b.jpg').copy()
    return np.asarray(pts_im_a), np.asarray(pts_im_b), im_a, im_b


def p2_abc():
    pts_im_a, pts_im_b, im_a, im_b = read_pic2_im_n_points()
    f = solve_for_f_svd(pts_im_a, pts_im_b)
    print('Calculated Fundamental matrix:\n%s\nand its rank is %s' % (f, np.linalg.matrix_rank(f)), end='\n' * 3)
    f = fix_f_rank(f)
    print('Fixed Fundamental matrix:\n%s\nand its rank is %s' % (f, np.linalg.matrix_rank(f)))
    lines_a = get_epipolar_line_ends(f, pts_im_b, im_a.shape)
    lines_b = get_epipolar_line_ends(f.T, pts_im_a, im_b.shape)
    [cv.line(im_a, l, r, (255, 0, 0), thickness=2) for (l, r) in lines_a]
    [cv.line(im_b, l, r, (255, 0, 0), thickness=2) for (l, r) in lines_b]
    imshow([im_a, im_b], shape=(2, 1))
    imsave(im_a, 'output/ps3-2-c-1.png')
    imsave(im_b, 'output/ps3-2-c-2.png')


def p2_de():
    pts_im_a, pts_im_b, im_a, im_b = read_pic2_im_n_points()
    t_a = get_transformation_mat_2d(pts_im_a)
    t_b = get_transformation_mat_2d(pts_im_b)
    pts_a_hom = np.append(pts_im_a, np.ones((pts_im_a.shape[0], 1)), axis=1)
    pts_b_hom = np.append(pts_im_b, np.ones((pts_im_b.shape[0], 1)), axis=1)
    pts_t_a = np.matmul(t_a, pts_a_hom.T).T
    pts_t_b = np.matmul(t_b, pts_b_hom.T).T
    print('Tª:\n%s' % t_a)
    print('Tᵇ:\n%s' % t_b)
    f_ = fix_f_rank(solve_for_f_svd(pts_t_a, pts_t_b))
    print('f_:\n%s' % f_)
    f = np.matmul(np.matmul(t_b.T, f_), t_a)
    print('corrected f:\n%s' % f)
    lines_a = get_epipolar_line_ends(f, pts_im_b, im_a.shape)
    lines_b = get_epipolar_line_ends(f.T, pts_im_a, im_b.shape)
    [cv.line(im_a, l, r, (255, 0, 0), thickness=2) for (l, r) in lines_a]
    [cv.line(im_b, l, r, (255, 0, 0), thickness=2) for (l, r) in lines_b]
    imshow([im_a, im_b], shape=(2, 1))
    imsave(im_a, 'output/ps3-2-e-1.png')
    imsave(im_b, 'output/ps3-2-e-2.png')


if __name__ == '__main__':
    p1()
    p1_b()
    p1_c()
    p1_svd_vs_ols()
    p2_abc()
    p2_de()
