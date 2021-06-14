import random
import cv2 as cv
import numpy as np

from ps_hepers.helpers import read_points, imshow, imread, imsave


def construct_a_mat(pts_world, pts_proj):
    a = []
    for (x, y, z), (u, v) in zip(pts_world, pts_proj):
        a.append([x, y, z, 1, 0, 0, 0, 0, -u * x, -u * y, -u * z, -u])
        a.append([0, 0, 0, 0, x, y, z, 1, -v * x, -v * y, -v * z, -v])
    return np.asarray(a)


def construct_f_mat(pts_im_left, pts_im_right):
    f = []
    for (u, v, *rest), (u_, v_, *rest) in zip(pts_im_left, pts_im_right):
        f.append([u_ * u, u_ * v, u_, v_ * u, v_ * v, v_, u, v, 1])
    return np.asarray(f)


def solve_for_svd(a):
    w, v = np.linalg.eig(np.matmul(a.T, a))
    m = v[:, w.argmin()]
    return m


def solve_with_leastsq(a):
    b = -a[:, -1]
    a = a[:, 0:-1]
    m, res, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return np.append(m, 1)


def solve_for_m_svd(pts_world, pts_proj):
    a = construct_a_mat(pts_world, pts_proj)
    return solve_for_svd(a).reshape((3, 4))


def solve_for_m_leastsq(pts_world, pts_proj):
    a = construct_a_mat(pts_world, pts_proj)
    return solve_with_leastsq(a).reshape((3, 4))


def solve_for_f_svd(pts_im1, pts_im2):
    a = construct_f_mat(pts_im1, pts_im2)
    return solve_for_svd(a).reshape((3, 3))


def solve_for_f_leastsq(pts_im1, pts_im2):
    a = construct_f_mat(pts_im1, pts_im2)
    return solve_with_leastsq(a).reshape((3, 3))


def fix_f_rank(f):
    u, s, vh = np.linalg.svd(f)
    s[s.argmin()] = 0
    return np.matmul(np.matmul(u, np.diag(s)), vh)


def get_epipolar_line_ends(f, pts_b, im_shape, t=None):
    pts_b_t = np.append(np.asarray(pts_b), np.ones((len(pts_b), 1)), axis=1).T
    l_l = np.cross([0, 0, 1], [im_shape[0], 0, 1])
    l_r = np.cross([0, im_shape[1], 1], [im_shape[0], im_shape[1], 1])
    l_a = np.matmul(f.T, pts_b_t)
    if t is not None:
        l_a = np.matmul(t.T, l_a)
    skew = lambda x: np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
    p_l = np.matmul(skew(l_l).T, l_a).T
    p_r = np.matmul(skew(l_r).T, l_a).T
    p_l = np.asarray([(int(x[0] / x[2]), int(x[1] / x[2])) for x in p_l])
    p_r = np.asarray([(int(x[0] / x[2]), int(x[1] / x[2])) for x in p_r])
    return tuple(zip(p_l, p_r))


def get_transformation_mat_2d(points):
    s_a = 1 / np.max(points)
    t_a_s = np.diag([s_a, s_a, 1])
    c_a = np.mean(points, axis=0)
    t_a_c = np.asarray([[1, 0, -c_a[0]], [0, 1, -c_a[1]], [0, 0, 1]])
    return np.matmul(t_a_s, t_a_c)


def calculate_residual(m, pts_world, pts_proj):
    pts_world_mat = np.append(np.asarray(pts_world), np.ones((len(pts_world), 1)), axis=1).T
    pts_proj_est = np.matmul(m, pts_world_mat)
    pts_proj_est_non_homo = pts_proj_est.T[:, 0:2] / pts_proj_est.T[:, [2, 2]]
    ssd_error = np.linalg.norm(pts_proj_est_non_homo - np.asarray(pts_proj))
    return np.sqrt(ssd_error)


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
    ssd_error = np.linalg.norm(pts_proj_est_non_homo - np.asarray(pts_proj))
    print('Estimated projection of pt corresponding to %s is %s' % (pts_proj[-1], tuple(pts_proj_est_non_homo[-1])))
    print('ssd_error = %s\nresidual = %s' % (ssd_error, np.sqrt(ssd_error)))


def p1_b(debug=True, on_normalized_set=False):
    if on_normalized_set:
        pts_proj = read_points('input/pts2d-norm-pic_a.txt')
        pts_world = read_points('input/pts3d-norm.txt')
    else:
        pts_proj = read_points('input/pts2d-pic_b.txt')
        pts_world = read_points('input/pts3d.txt')
    k_sizes = range(6, 17)
    num_tests = 100
    residuals = np.zeros((num_tests, len(k_sizes)))
    (res_min, best_m) = (np.PINF, None)
    for i in range(num_tests):
        for k_i, k in zip(range(len(k_sizes)), k_sizes):
            indexes = random.sample(range(0, len(pts_world)), k)
            m = solve_for_m_svd([pts_world[i] for i in indexes], [pts_proj[i] for i in indexes])
            test_set = random.sample([i for i in range(0, len(pts_world)) if i not in indexes], 4)
            residual = calculate_residual(m, [pts_world[i] for i in test_set], [pts_proj[i] for i in test_set])
            (res_min, best_m) = (min(residual, res_min), m if residual < res_min else best_m)
            residuals[i][k_i] = residual
    if debug:
        print("Residuals (each row represents avg residual for k size sets k = %s ) :\n%s" % (k_sizes, residuals))
        print(
            "Average residuals over all random tests  for k size sets k = %s: \n%s" % (k_sizes, residuals.mean(axis=0)))
        print("Best M matrix and with residual (%s) \n%s" % (res_min, best_m))
        imshow([residuals, residuals.mean(axis=0)[np.newaxis, :][[0] * 10, :]],
               ['residuals for each k size and test', 'residuals averaged over all tests'],
               interpolation='nearest',
               shape=(1, 2),
               sup_title='Fancy way to see the avg residuals')
    return best_m, res_min


def p1_c():
    m, residual = p1_b(debug=False, on_normalized_set=True)
    cam_center = -np.matmul(np.linalg.inv(m[:, 0:3]), m[:, 3])
    print('Camera Center:\n%s' % cam_center)


def p1_svd_vs_ols():
    pts_proj = read_points('input/pts2d-norm-pic_a.txt')
    pts_world = read_points('input/pts3d-norm.txt')
    m_svd = solve_for_m_svd(pts_world, pts_proj)
    m_ols = solve_for_m_leastsq(pts_world, pts_proj)
    print('residuals with svd: %s' % calculate_residual(m_svd, pts_world, pts_proj))
    print('residuals with least squares: %s' % calculate_residual(m_ols, pts_world, pts_proj))


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
