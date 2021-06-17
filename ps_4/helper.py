import cv2 as cv
import numpy as np
import multiprocess as mp
from multiprocessing import freeze_support
from ps_hepers.helpers import non_max_suppression, xy_2_ij
from scipy.spatial import ConvexHull


def im_der(im, with_norm=False):
    gx = cv.Sobel(im, cv.CV_64F, 1, 0, ksize=3)
    gy = cv.Sobel(im, cv.CV_64F, 0, 1, ksize=3)
    if not with_norm:
        return gx, gy
    return gx, gy,


def get_gaussian_kernel(size, sigma=-1):
    if sigma < 0:
        sigma = (size - 1) / 6.0
    k = np.zeros((size, size), dtype=np.float32)
    k[size // 2, size // 2] = 1
    return cv.GaussianBlur(k, (size, size), sigma)


def compute_harris_resp(im, kernel_size=3, alpha=0.04):
    global compute_response_ij
    im = im.copy() * 1.0 / im.max()
    gx, gy = im_der(im)
    freeze_support()
    kernel_size = [kernel_size, 3][kernel_size is None]
    window_shape = (kernel_size, kernel_size)
    harris_resp = np.zeros(im.shape)
    (w_h, w_l) = map(lambda x: (x - 1) // 2, window_shape)
    w = get_gaussian_kernel(kernel_size, 0)

    def compute_response_ij(i, j):
        ix = gx[i - w_h:i + w_h + 1, j - w_l:j + w_l + 1]
        iy = gy[i - w_h:i + w_h + 1, j - w_l:j + w_l + 1]
        m_xx = np.sum(w * ix * ix)
        m_xy = np.sum(w * ix * iy)
        m_yy = np.sum(w * iy * iy)
        m = np.asarray([[m_xx, m_xy], [m_xy, m_yy]])
        return np.linalg.det(m) - alpha * (m.trace() ** 2)

    with mp.Pool(10) as pool:
        def i_slice(i):
            slice = np.zeros(im.shape[1])
            for j in range(w_l, im.shape[1] - w_l):
                slice[j] = compute_response_ij(i, j)
            return slice

        i_range = range(w_h, im.shape[0] - w_h)
        slices = pool.map(i_slice, list(i_range))
        for i, slice_i in zip(i_range, slices):
            harris_resp[i, :] = slice_i
    return harris_resp


def compute_harris_corners(img, threshold=0.01, w_shape=(5, 5), alpha=0.04):
    img = img.copy()
    harris_resp = compute_harris_resp(img, kernel_size=5, alpha=alpha)
    harris_resp[harris_resp < threshold * harris_resp.max()] = 0
    suppressed = non_max_suppression(harris_resp, w_shape)
    x_s, y_s = np.where(suppressed > 0)
    return suppressed, list(zip(x_s, y_s))


def get_sift_descriptors(im, harris_corners=None):
    if harris_corners is None:
        _, harris_corners = compute_harris_corners(im)
    gx, gy = im_der(im)
    theta_im = np.rad2deg(np.arctan2(gx, gy))
    key_points = [cv.KeyPoint(x=int(j), y=int(i), _size=10, _angle=theta_im[i, j], _octave=0) for (i, j) in
                  harris_corners]
    sift = cv.SIFT_create()
    _, descriptors = sift.compute(im, key_points)
    return key_points, descriptors


def get_matches(im_a, im_b):
    pts_a, desc_a = get_sift_descriptors(im_a)
    pts_b, desc_b = get_sift_descriptors(im_b)
    bfm = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    matches = bfm.match(desc_a, desc_b)
    matches = sorted(matches, key=lambda x: x.distance)
    return [(xy_2_ij(pts_a[match.queryIdx].pt), xy_2_ij(pts_b[match.trainIdx].pt)) for match in matches]


# def get_trans_consensus(matches):
#     pts_a = np.asarray([match[0] for match in matches])
#     pts_b = np.asarray([match[1] for match in matches])
#
#     def consensus_for_i(i):
#         trans_ij = pts_b[i] - pts_a[i]
#         consensus_set = np.linalg.norm((pts_a + trans_ij) - pts_b, axis=1) < 5
#         return consensus_set.sum(), consensus_set
#
#     (consensus_strength, cs) = (np.NINF, None)
#     for i in range(len(matches)):
#         cs_strength_i, cs_i = consensus_for_i(i)
#         (consensus_strength, cs) = (
#             max(consensus_strength, cs_strength_i), cs_i if consensus_strength < cs_strength_i else cs)
#
#     return list(np.where(cs)[0])


def ransac(matches, consensus_for_sample, sample_size, outlier_p=0.5, biased=True, retries=None):
    (consensus_strength, cs) = (np.NINF, None)
    matches_n = len(matches)
    matches = np.asarray(matches)
    probs_sum = lambda x: 3 * x * x + x * (x + 1) / 2
    biased_prob = np.arange(4 * matches_n, 3 * matches_n, -1) / probs_sum(matches_n)
    # computes the N needed for finding the best sample set with .99 probability
    if retries is None:
        max_retries = int(np.log(0.01) / np.log(1 - outlier_p ** sample_size))
        max_retries = min(max_retries, 1000)
    else:
        max_retries = retries
    for i in range(max_retries):
        rand_sample = np.random.choice(matches_n, sample_size, p=biased_prob) if biased else np.random.choice(
            matches_n, sample_size)
        cs_strength_i, cs_i = consensus_for_sample(matches, rand_sample)
        (consensus_strength, cs) = (
            max(consensus_strength, cs_strength_i), cs_i if consensus_strength < cs_strength_i else cs)
    return list(np.where(cs)[0])


def ransac_trans(matches):
    pts_a = np.asarray([match[0] for match in matches])
    pts_b = np.asarray([match[1] for match in matches])

    def consensus_for_i(_, sample):
        i = sample[0]
        trans_ij = pts_b[i] - pts_a[i]
        consensus_set = np.linalg.norm((pts_a + trans_ij) - pts_b, axis=1) < 5
        return consensus_set.sum(), consensus_set

    return ransac(matches, consensus_for_i, 1, biased=True)


def similarity_mat(pts_a, pts_b):
    l = min(pts_a.shape[0], pts_b.shape[0])
    a = np.zeros((2 * l, 4))
    b = np.zeros((2 * l,))
    for (u, v, *rest), (u_, v_, *rest), i in zip(pts_a, pts_b, range(l)):
        a[2 * i, :] = [u, -v, 1, 0]
        a[2 * i + 1, :] = [v, u, 1, 0]
        b[2 * i] = u_
        b[2 * i + 1] = v_
    m, res, _, _ = np.linalg.lstsq(a, b, rcond=None)
    a, b, c, d = m
    return np.asarray([[a, -b, c], [b, a, d]])


def affine_mat(pts_a, pts_b):
    l = min(pts_a.shape[0], pts_b.shape[0])
    a = np.zeros((2 * l, 6))
    b = np.zeros((2 * l,))
    for (u, v, *rest), (u_, v_, *rest), i in zip(pts_a, pts_b, range(l)):
        a[2 * i, :] = [u, v, 1, 0, 0, 0]
        a[2 * i + 1, :] = [0, 0, 0, u, v, 1]
        b[2 * i] = u_
        b[2 * i + 1] = v_
    m, res, _, _ = np.linalg.lstsq(a, b, rcond=None)
    return m.reshape((2, 3))


def get_normalization_mat(points):
    s_a = 1 / np.max(points)
    t_a_s = np.diag([s_a, s_a, 1])
    c_a = np.mean(points, axis=0)
    t_a_c = np.asarray([[1, 0, -c_a[0]], [0, 1, -c_a[1]], [0, 0, 1]])
    return np.matmul(t_a_s, t_a_c)


def normalize_match_points(matches):
    matches = np.asarray(matches)
    pts_a, pts_b = matches[:, 0, :], matches[:, 1, :]
    t_a = get_normalization_mat(pts_a)
    t_b = get_normalization_mat(pts_b)
    pts_a_hom = np.append(pts_a, np.ones((pts_a.shape[0], 1)), axis=1)
    pts_b_hom = np.append(pts_b, np.ones((pts_b.shape[0], 1)), axis=1)
    pts_t_a = np.matmul(t_a, pts_a_hom.T).T
    pts_t_b = np.matmul(t_b, pts_b_hom.T).T
    return np.concatenate([pts_t_a[:, np.newaxis, 0:2], pts_t_b[:, np.newaxis, 0:2]], axis=1), t_a, t_b


def ransac_transformation(matches, transformation):
    if transformation == 'similarity':
        sample_size = 2
        calc_transformation_mat = similarity_mat
    elif transformation == 'affine':
        sample_size = 3
        calc_transformation_mat = affine_mat
    else:
        raise Exception('%s is not a valid transformation' % transformation)

    n_matches, t_a, t_b = normalize_match_points(matches)
    error_tolerance = 5 * np.linalg.norm([t_a[0, 0], t_b[0, 0]])
    n_matches = np.asarray(n_matches)
    pts_a, pts_b = n_matches[:, 0, :], n_matches[:, 1, :]

    def consensus_for_sample(_, sample):
        transformation_m = calc_transformation_mat(pts_a[sample], pts_b[sample])
        pts_a_homo = np.append(pts_a, np.ones((pts_a.shape[0], 1)), axis=1)
        pts_b_est = np.matmul(transformation_m, pts_a_homo.T).T
        consensus_set = np.linalg.norm(pts_b_est - pts_b, axis=1) < error_tolerance
        return consensus_set.sum(), consensus_set

    return ransac(n_matches, consensus_for_sample, sample_size)


def warp_b_on_a(im_b, consensus_matches, transformation, normalize_points=True, consider_only_hull_points=False):
    matches = consensus_matches
    # normalize points so that least squares algo is not effected due to scale & offset
    if normalize_points:
        normalized_matches, t_a, t_b = normalize_match_points(consensus_matches)
        matches = normalized_matches
    pts_a, pts_b = matches[:, 0, :], matches[:, 1, :]

    # If true, the transformation matrix will be computed only from outer points (lying on convex hull's perimeter)
    if consider_only_hull_points:
        pts_index_on_hull = ConvexHull(pts_a).vertices
        pts_a, pts_b = pts_a[pts_index_on_hull, :], pts_b[pts_index_on_hull, :]
    if transformation == 'affine':
        transformation_m = np.concatenate([affine_mat(pts_a, pts_b), [[0, 0, 1]]], axis=0)
    elif transformation == 'similarity':
        transformation_m = np.concatenate([similarity_mat(pts_a, pts_b), [[0, 0, 1]]], axis=0)
    else:
        raise Exception('%s is not a valid transformation' % transformation)

    # adjust the transformation matrix to work with non normalized points
    # T' = inv(t_b) * T * t_a
    if normalize_points:
        transformation_m = np.matmul(np.linalg.inv(t_b), np.matmul(transformation_m, t_a))

    # converting the T matrix into (x,y) from (i,j) as opencv's warpAffine takes input in (x,y)
    a, b, c, d, e, f = transformation_m[:2, :].flatten()
    warped = cv.warpAffine(im_b, np.asarray([[e, d, f], [b, a, c]]), im_b.shape[1::-1], flags=cv.WARP_INVERSE_MAP)
    return warped
