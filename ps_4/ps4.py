import numpy as np
from ps_4.helper import im_der, compute_harris_resp, compute_harris_corners, get_sift_descriptors, get_matches, \
    ransac_trans, similarity_mat, affine_mat, ransac_transformation, warp_b_on_a
from ps_hepers.helpers import imread_from_rep, imshow, imfix_scale, imsave, non_max_suppression, overlap_boolean_image, \
    np_load, np_save, ij_2_xy, mark_points
import cv2 as cv

"""
Problem Set - 4
Problems: https://docs.google.com/document/d/1DlyziyQB163r1Lx3F4-Tanm8Oq4O9-W3X5Hpdw4QGUE/pub?embedded=true
"""


def p1_a():
    trans_a = imread_from_rep('transA', extension='.jpg', grey_scale=True)
    sim_a = imread_from_rep('simA', extension='.jpg', grey_scale=True)
    trans_gx, trans_gy = im_der(trans_a)
    trans_grad_pair = np.concatenate([trans_a, imfix_scale(trans_gx), imfix_scale(trans_gy)], axis=1)
    imshow(trans_grad_pair, cmap='gray')
    imsave(trans_grad_pair, 'output/ps4-1-a-1.png')
    sim_gx, sim_gy = im_der(sim_a)
    sim_grad_pair = np.concatenate([sim_a, imfix_scale(sim_gx), imfix_scale(sim_gy)], axis=1)
    imshow(sim_grad_pair, cmap='gray')
    imsave(sim_grad_pair, 'output/ps4-1-a-2.png')


def p1_bc():
    img_names = ['transA', 'transB', 'simA', 'simB']
    for_disp = []
    for i in range(len(img_names)):
        print('processing image %s' % (i + 1))
        img = imread_from_rep(img_names[i], extension='.jpg', grey_scale=True)
        harris_resp = compute_harris_resp(img, kernel_size=5)
        harris_corners_im, _ = compute_harris_corners(img)
        corners_marked = overlap_boolean_image(np.concatenate([img[:, :, np.newaxis]] * 3, axis=2), harris_corners_im,
                                               cross_marks=True)
        imsave(imfix_scale(harris_resp), 'output/ps4-1-b-%s.png' % (i + 1))
        imsave(imfix_scale(corners_marked), 'output/ps4-1-c-%s.png' % (i + 1))
        for_disp.append((img, harris_resp, corners_marked))
    [imshow([img, harris_resp, corners_marked], cmap='gray', shape=(2, 2)) for (img, harris_resp, corners_marked) in
     for_disp]


def p1_exp():
    """
    play with the slider to understand the effect of alpha on harris corners
    :return:
    """
    filename = 'transA'
    img = imread_from_rep(filename, extension='.jpg', grey_scale=True)
    alpha_start, alpha_end, alpha_step = 0.01, 0.1, 0.01
    alpha_range = np.arange(alpha_start, alpha_end, alpha_step)
    hc_alpha_list = np_load(len(alpha_range), 'objects/hc_alpha_%s_%s.npy' % (alpha_range, filename))

    if hc_alpha_list is None or len(hc_alpha_list) != len(alpha_range):
        hc_alpha_list = []
        for alpha in alpha_range:
            print('processing image for harris corners with alpha %.2f' % alpha)
            harris_corners_im, _ = compute_harris_corners(img, alpha=alpha)
            corners_marked = overlap_boolean_image(np.concatenate([img[:, :, np.newaxis]] * 3, axis=2),
                                                   harris_corners_im,
                                                   cross_marks=True)
            hc_alpha_list.append((alpha, corners_marked))
            imsave(corners_marked, 'observations/harris_corners_%s_alpha_%.2f.png' % (filename, alpha))
    np_save(np.asarray(hc_alpha_list, dtype=object), 'objects/hc_alpha_%s_%s.npy' % (alpha_range, filename))

    def update_exp(x, axs, sliders, buttons):
        i = np.abs(alpha_range - sliders[0].val).argmin()
        return [0], [hc_alpha_list[i][1]]

    slider_attr = [{'label': 'alpha', 'valmin': alpha_start, 'valmax': alpha_end, 'valstep': alpha_step}]
    imshow(hc_alpha_list[0][1], ['harris corners'], slider_attr=slider_attr, slider_callback=[update_exp])


def p2_a():
    img_pair_names = [('transA', 'transB'), ('simA', 'simB')]
    for (im_name_a, im_name_b), i_im in zip(img_pair_names, range(len(img_pair_names))):
        im_a = imread_from_rep(im_name_a, extension='.jpg', grey_scale=True)
        im_b = imread_from_rep(im_name_b, extension='.jpg', grey_scale=True)
        pts_a, _ = get_sift_descriptors(im_a)
        pts_b, _ = get_sift_descriptors(im_b)
        im_a = np.concatenate([im_a[:, :, np.newaxis]] * 3, axis=2)
        im_b = np.concatenate([im_b[:, :, np.newaxis]] * 3, axis=2)
        cv.drawKeypoints(im_a, pts_a, im_a, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        cv.drawKeypoints(im_b, pts_b, im_b, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        ab = np.concatenate([im_a, im_b], axis=1)
        imsave(ab, 'output/ps4-2-a-%s.png' % (i_im + 1))
        imshow(ab, cmap='gray')


def draw_matches(a, b, matches, color='random'):
    def color_i(i):
        np.random.seed(i)
        return {'random': list(np.random.random(size=3) * 256), 'weighted': (255 - i * 255.0 / len(matches), 0, 0)}.get(
            color, color)

    ab = np.concatenate([a, b], axis=1)
    ab = np.concatenate([ab[:, :, np.newaxis]] * 3, axis=2)
    for ((p1, p2), i) in reversed(list(zip(matches, range(len(matches))))):
        cv.line(ab, ij_2_xy(p1), ij_2_xy((p2[0], p2[1] + a.shape[1])), color=color_i(i), thickness=2,
                lineType=cv.LINE_AA)
        cv.circle(ab, ij_2_xy(p1), radius=5, color=color_i(i), thickness=2)
        cv.circle(ab, ij_2_xy((p2[0], p2[1] + a.shape[1])), radius=5, color=color_i(i), thickness=2)
    return ab


def p2_b():
    img_pair_names = [('transA', 'transB'), ('simA', 'simB')]
    for (im_name_a, im_name_b), i_im in zip(img_pair_names, range(len(img_pair_names))):
        a = imread_from_rep(im_name_a, extension='.jpg', grey_scale=True)
        b = imread_from_rep(im_name_b, extension='.jpg', grey_scale=True)
        matches = get_matches(a, b)
        ab = draw_matches(a, b, matches, color='weighted')
        imshow(ab, cmap='gray')
        imsave(ab, 'output/ps4-2-b-%s.png' % (i_im + 1))


def p3_a():
    img_pair_names = [('transA', 'transB')]
    for (im_name_a, im_name_b), i_im in zip(img_pair_names, range(len(img_pair_names))):
        a = imread_from_rep(im_name_a, extension='.jpg', grey_scale=True)
        b = imread_from_rep(im_name_b, extension='.jpg', grey_scale=True)
        matches = get_matches(a, b)
        matches_index = ransac_trans(matches)
        matches = [matches[i] for i in list(matches_index)]
        ab = draw_matches(a, b, matches)
        imshow(ab, cmap='gray')
        imsave(ab, 'output/ps4-3-a-%s.png' % (i_im + 1))


def p3_bcde():
    img_pair_names = [('simA', 'simB')]
    transformation_types = ['similarity', 'affine']
    calc_transformation_method = [similarity_mat, affine_mat]
    for transformation_type, pb_i in zip(transformation_types, range(len(transformation_types))):
        for (im_name_a, im_name_b), i_im in zip(img_pair_names, range(len(img_pair_names))):
            a = imread_from_rep(im_name_a, extension='.jpg', grey_scale=True)
            b = imread_from_rep(im_name_b, extension='.jpg', grey_scale=True)
            putative_matches = get_matches(a, b)
            matches_index = ransac_transformation(putative_matches, transformation_type)
            matches = np.asarray([putative_matches[i] for i in list(matches_index)])
            cal_transformation_mat = calc_transformation_method[pb_i]
            print('%s matrix:\n%s' % (transformation_type, cal_transformation_mat(matches[:, 0, :], matches[:, 1, :])))
            print('Percentage of matches for biggest consensus set: %s' % (100 * len(matches) / len(putative_matches)))
            ab = draw_matches(a, b, matches)
            imshow(ab, 'consensus set : %s' % transformation_type, cmap='gray')
            imsave(ab, 'output/ps4-3-%s-%s.png' % (chr(ord('b') + pb_i), i_im + 1))
            warped = warp_b_on_a(b, matches, transformation_type)
            blend = a // 2 + warped // 2
            imshow(blend, 'blend : %s' % transformation_type, cmap='gray')
            imsave(warped, 'output/ps4-3-%s-%s.png' % (chr(ord('d') + pb_i), 2 * i_im + 1))
            imsave(blend, 'output/ps4-3-%s-%s.png' % (chr(ord('d') + pb_i), 2 * i_im + 2))


def p3_exp():
    img_pair_names = [('simA', 'simB')]
    for (im_name_a, im_name_b), i_im in zip(img_pair_names, range(len(img_pair_names))):
        a = imread_from_rep(im_name_a, extension='.jpg', grey_scale=True)
        b = imread_from_rep(im_name_b, extension='.jpg', grey_scale=True)
        putative_matches = get_matches(a, b)
        matches_index = ransac_transformation(putative_matches, 'affine')
        matches = np.asarray([putative_matches[i] for i in list(matches_index)])
        print('Affine matrix:\n%s' % affine_mat(matches[:, 0, :], matches[:, 1, :]))
        print('Percentage of matches for biggest consensus set: %s' % (100 * len(matches) / len(putative_matches)))
        warp1 = warp_b_on_a(b, matches, 'affine', normalize_points=True, consider_only_hull_points=True)
        warp2 = warp_b_on_a(b, matches, 'affine', normalize_points=True, consider_only_hull_points=False)
        warp3 = warp_b_on_a(b, matches, 'affine', normalize_points=False, consider_only_hull_points=True)
        warp4 = warp_b_on_a(b, matches, 'affine', normalize_points=False, consider_only_hull_points=False)
        ab = draw_matches(a, b, matches)
        imshow(ab, cmap='gray')
        blend = lambda warped: warped // 2 + a // 2
        imgs = [blend(warp1), blend(warp2), blend(warp3), blend(warp4)]
        img_names = ['normalized hull only points', 'normalized points', 'non normalized hull only points',
                     'non normalized points']
        imshow(imgs, img_names, cmap='gray')
        [imsave(im, 'observations/%s.png' % name) for (im, name) in zip(imgs, img_names)]


if __name__ == '__main__':
    p1_a()
    p1_bc()
    p1_exp()
    p2_a()
    p2_b()
    p3_bcde()
    p3_exp()
