import cv2 as cv
import numpy as np
from ps_hepers.helpers import imread_from_rep, imread, imshow, imsave, is_in, imfix_scale, highlight_pos_im, \
    overlap_boolean_image

"""
Problem Set - 1
Questions at : https://docs.google.com/document/d/13CJgtDr8kIX9KIrs6BYFDF6-N7cfAyX0R54v8CWoqmQ/pub?embedded=true
"""

def testing_imshow():
    im = imread_from_rep('ps1-input0')
    im2 = imread_from_rep('lena')
    # sliders
    # 1st one sets image at slider.val to lena and  2nd one sets image at slider.val back to squares
    slider_attr = [{'label': 'Sets to Lena', 'valmin': 0, 'valmax': 10, 'valint': 5},
                   {'label': 'Sets to squares', 'valmin': 0, 'valmax': 20, 'valint': 15}]
    # buttons
    button_attr = [{'label': 'reset 1st slider'}, {'label': 'reset 2nd slider'}]

    def update_im2lena(x, axs, sliders, buttons):
        return [int(x)], [im2]

    def update_im2squares(x, axs, sliders, buttons):
        return [int(x / 2)], [im]

    def reset_a(x, axs, sliders, buttons):
        sliders[0].reset()

    def reset_b(x, axs, sliders, buttons):
        sliders[1].reset()

    imshow([im] * 8, im_title=['im - %s' % i for i in range(23)], sup_title='Testing fun', slider_attr=slider_attr,
           slider_callback=[update_im2lena, update_im2squares], button_attr=button_attr,
           button_callback=[reset_a, reset_b])


# def l_o_g(im, th):
#     """
#     Experimental edge detection using laplacian with zero crossings
#     """
#     im_log = cv.Laplacian(im, cv.CV_16S)
#     min_log = cv.morphologyEx(im_log, cv.MORPH_ERODE, np.ones((3, 3)))
#     max_log = cv.morphologyEx(im_log, cv.MORPH_DILATE, np.ones((3, 3)))
#     crossings = np.logical_or(np.logical_and(min_log < 0, im_log > 0), np.logical_and(max_log > 0, im_log < 0))
#     return imfix_scale(crossings & im)


def p1():
    # Canny Detection - https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
    # 2d explanation on Non-maximal suppression & Hysteresis Thresholding -
    # http://www.justin-liang.com/tutorials/canny/#:~:text=Non%20maximum%20suppression%20works%20by,the%20gradient%20direction%20of%20q
    im = imread_from_rep('ps1-input3 copy', grey_scale=True)
    edges = cv.Canny(im, 100, 200)
    slider_attr = [{'label': 'threshold1', 'valmin': 0, 'valmax': 300, 'valint': 100},
                   {'label': 'threshold2', 'valmin': 0, 'valmax': 300, 'valint': 200}]

    def update_threshold(x, axs, sliders, buttons):
        print(sliders[0].val, sliders[1].val)
        return [1], [cv.Canny(im, sliders[0].val, sliders[1].val)]

    imshow([im, edges], ['original', 'canny edges'], slider_attr=slider_attr,
           slider_callback=[update_threshold] * 2)


def hough_lines_acc(edges, theta_range=np.arange(-180, 180)):
    """
    custom hough lines that computes accumulator with only positive rho range to positive, while keeping the theta range
    broad. This helps avoid issues with edges at th = -90 or 89. The issue being the peaks at boundaries of acc tend to
    go inward due to noise in the accumulator
    """
    max_d = int(np.round(np.sqrt(edges.shape[0] ** 2 + edges.shape[1] ** 2)))
    acc = np.zeros((max_d, len(theta_range)))
    i_s, j_s = np.where(edges > 0)
    for i, j in zip(i_s, j_s):
        for th in theta_range:
            d = int(np.round(j * np.cos(th * np.pi / 180.0) + i * np.sin(th * np.pi / 180.0)))
            if d > 0:
                th_i = np.where(theta_range == th)[0][0]
                acc[d][th_i] = acc[d][th_i] + 1
    return acc


def draw_line_on_im(im, th_d_tuples):
    shape = im.shape
    for th_d in th_d_tuples:
        d = th_d[0]
        th = th_d[1]
        with_x = np.abs(th) >= 45.0
        for it in range(shape[1] if with_x else shape[0]):
            th_in_radians = th * np.pi / 180.0
            x = int(np.round(it if with_x else (d - it * np.sin(th_in_radians)) / np.cos(th_in_radians)))
            y = int(np.round(it if not with_x else (d - it * np.cos(th_in_radians)) / np.sin(th_in_radians)))
            if is_in(im.shape, y, x):
                im[y, x] = 1
    return im


def hough_peak_matlab_like(acc, npeaks=1, threshold=0.5, size=None):
    """
    This one works with both two and three dimensional hough accumulator spaces/bins
    """
    acc = np.copy(acc)

    def round_range(start, end, max):
        return [range(start, end), list(range(start, max)) + list(range(0, end % max))][int(end) >= int(max)]

    size = [size, np.asarray(acc.shape) // 50][size is None]
    size = [[size[i] + 1, size[i]][size[i] % 2] for i in range(len(acc.shape))]
    bounds = [(size[i] - 1) // 2 for i in range(len(acc.shape))]
    pts = []
    max_peak = np.max(acc)
    while len(pts) < npeaks:
        index = np.unravel_index(np.asarray(acc).argmax(), acc.shape)
        if len(index) == 2 and acc[index[0], index[1]] >= threshold * max_peak and acc[index[0], index[1]] > 0:
            pts.append((index[0], index[1]))
            acc[np.ix_(round_range(index[0] - bounds[0], index[0] + bounds[0] + 1, acc.shape[0]),
                       round_range(index[1] - bounds[1], index[1] + bounds[1] + 1, acc.shape[1]))] = 0
        elif len(index) == 3 and acc[index[0], index[1], index[2]] >= threshold * max_peak and acc[
            index[0], index[1], index[2]] > 0:
            pts.append((index[0], index[1], index[2]))
            acc[np.ix_(round_range(index[0] - bounds[0], index[0] + bounds[0] + 1, acc.shape[0]),
                       round_range(index[1] - bounds[1], index[1] + bounds[1] + 1, acc.shape[1]),
                       round_range(index[2] - bounds[2], index[2] + bounds[2] + 1, acc.shape[2]))] = 0
        else:
            break
    return pts


def hough_peak_custom(hough_acc, threshold=0.5):
    hough_acc_w = cv.copyMakeBorder(hough_acc, 3, 3, 3, 3, cv.BORDER_WRAP)
    blur_acc = cv.GaussianBlur(hough_acc_w, (5, 5), 0)[3:-3, 3:-3]
    th_blur_acc = blur_acc * (blur_acc > np.max(blur_acc) * threshold)
    nms_th_blur_acc = non_max_suppression(th_blur_acc)
    # imshow(nms_th_blur_acc)
    ds, ths = np.where(nms_th_blur_acc > 0)
    return [(d, th) for d, th in zip(ds, ths)]


def non_max_suppression(im, size=3):
    if size % 2 == 0:
        raise Exception('size should be odd, so the filter has a center pixel')
    uv_s = []
    for i in range(-size // 2 + 1, size // 2 + 1):
        for j in range(-size // 2 + 1, size // 2 + 1):
            uv_s.append((i, j))
    uv_s.remove((0, 0))
    im_size = im.shape
    op = np.zeros(im_size)
    for i in range(im_size[0]):
        for j in range(im_size[1]):
            op[i, j] = im[i, j] > np.max(np.asarray(
                [im[i + uv[0], j + uv[1]] if is_in(im_size, i + uv[0], j + uv[1]) else np.NINF for uv in uv_s]))
    return op


def p2_experiment():
    image_filename = 'ps1-input0'
    im = imread_from_rep(image_filename, grey_scale=True)
    # load color version to overlay the detected lines
    im_color = imread_from_rep(image_filename)
    # get edges with canny
    edges = cv.Canny(im, 100, 200)
    # compute accumulator array
    th_range = np.arange(-180, 180, 1)
    hough_acc = hough_lines_acc(edges, th_range)
    imsave(hough_acc, 'output/ps1-2-a-1.png')
    # using custom peak finding, but the method is copied so that intermediate steps can be visualized in the
    # interactive imshow figure
    hough_acc_w = cv.copyMakeBorder(hough_acc, 3, 3, 3, 3, cv.BORDER_WRAP)
    blur_acc = cv.GaussianBlur(hough_acc_w, (5, 5), 0)[3:-3, 3:-3]
    th_blur_acc = blur_acc * (blur_acc > np.max(blur_acc) * 0.90)
    nms_th_blur_acc = non_max_suppression(th_blur_acc)

    ds, ths = np.where(nms_th_blur_acc > 0)
    # convert the th indexes to theta range
    line_params = [(d, th_range[th]) for d, th in zip(ds, ths)]
    # draw lines over new image and merge it into red channel of colored input image
    line_im = draw_line_on_im(np.zeros(im.shape), line_params)
    im_color[:, :, 1] = im_color[:, :, 1] + line_im * 255.0
    # slider that controls the thresholding
    slider_attr = [{'label': 'threshold over accumulator', 'valmin': 0.1, 'valmax': .99, 'valstep': 0.01}]

    def slider_update(x, axs, sliders, buttons):
        th_blur_acc = blur_acc * (blur_acc > np.max(blur_acc) * sliders[0].val)
        # er_th_blur_acc = cv.morphologyEx(th_blur_acc, cv.MORPH_ERODE, np.ones((3, 3)))
        nms_th_blur_acc = non_max_suppression(th_blur_acc)
        # print(np.sum(nms_th_blur_acc > 0))
        ds, ths = np.where(nms_th_blur_acc > 0)
        print(ths)
        line_params = [(d, th_range[th]) for d, th in zip(ds, ths)]
        line_im = draw_line_on_im(np.zeros(im.shape), line_params)
        print(line_params)
        print(line_im.shape, im.shape, edges.shape)
        im_color_to_render = np.copy(im_color)
        im_color_to_render[line_im > 0, 0] = 255
        return [3, 4, 5, 6], [th_blur_acc, nms_th_blur_acc, line_im, im_color_to_render]

    imshow([edges, hough_acc, blur_acc, th_blur_acc, nms_th_blur_acc, line_im, im_color],
           ['edges', 'hough_acc', 'blur_acc', 'th_blur_acc', 'nms_th_blur_acc', 'line_im', 'colored'],
           interpolation='nearest', slider_attr=slider_attr, slider_callback=[slider_update])
    imsave(imfix_scale(hough_acc), 'observations/hough_acc.png')


def p2_experiment_load_houghacc():
    im = imread('observations/hough_acc.png', grey_scale=True)
    print(im.shape)
    im_w = cv.copyMakeBorder(im, 3, 3, 3, 3, cv.BORDER_WRAP)
    imblur = cv.GaussianBlur(im_w, (5, 5), 0)
    imblur = imblur[3:-3, 3:-3]
    print(imblur.shape)
    th_imblur = (imblur > np.max(imblur) * .9) * imblur
    nm_imblur = non_max_suppression(th_imblur)
    imshow([im, im_w, imfix_scale(imblur), th_imblur, nm_imblur], interpolation='nearest')


def p2_experiment_peak_finding():
    image_filename = 'ps1-input0'
    im = imread_from_rep(image_filename, grey_scale=True)
    edges = cv.Canny(im, 100, 200)
    th_range = np.arange(-180, 180, 1)
    hough_acc = hough_lines_acc(edges, th_range)
    hough_peaks_c = hough_peak_custom(hough_acc, 0.7)
    lines_p_c = [(d_th[0], th_range[d_th[1]]) for d_th in hough_peaks_c]
    hough_peaks_m = hough_peak_matlab_like(hough_acc, 6, 0.7)
    lines_p_m = [(d_th[0], th_range[d_th[1]]) for d_th in hough_peaks_m]

    slider_attr = [{'label': 'threshold over accumulator', 'valmin': 0.1, 'valmax': .99, 'valstep': 0.01},
                   {'label': 'num peaks', 'valmin': 0, 'valmax': 99, 'valstep': 1}]

    def update_exp(x, axs, sliders, buttons):
        hough_peaks_c = hough_peak_custom(hough_acc, sliders[0].val)
        lines_p_c = [(d_th[0], th_range[d_th[1]]) for d_th in hough_peaks_c]
        hough_peaks_m = hough_peak_matlab_like(hough_acc, int(sliders[1].val), sliders[0].val)
        lines_p_m = [(d_th[0], th_range[d_th[1]]) for d_th in hough_peaks_m]
        # print('updated', len(lines_p_m))
        # print(lines_p_m)
        return [0, 1], [draw_line_on_im(np.zeros(im.shape), lines_p_c), draw_line_on_im(np.zeros(im.shape), lines_p_m)]

    imshow([draw_line_on_im(np.zeros(im.shape), lines_p_c), draw_line_on_im(np.zeros(im.shape), lines_p_m)],
           ['custom hough peak', 'matlab\'s hough peak'],
           slider_attr=slider_attr, slider_callback=[update_exp] * 2)


def p3():
    im_noise = imread_from_rep('ps1-input0-noise copy')
    blur_im_noise = cv.GaussianBlur(im_noise, (5, 5), 0)
    edge_noise = cv.Canny(im_noise, 100, 200)
    edge_blur_noise = cv.Canny(blur_im_noise, 100, 200)
    # dummy init image to show before adjusting params
    zeros = np.zeros(im_noise.shape)
    th_range = np.arange(-180, 180, 1)
    noise_hough_acc = hough_lines_acc(edge_noise, th_range)
    blur_hough_acc = hough_lines_acc(edge_blur_noise, th_range)

    # interactively adjust parameters used in this method using sliders
    def update_exp(x, axs, sliders, buttons):
        hough_peaks_noise = hough_peak_matlab_like(noise_hough_acc, sliders[0].val, sliders[1].val)
        lines_p_noise = [(d_th[0], th_range[d_th[1]]) for d_th in hough_peaks_noise]
        hough_peaks_blur = hough_peak_matlab_like(blur_hough_acc, sliders[0].val, sliders[1].val)
        lines_p_blur = [(d_th[0], th_range[d_th[1]]) for d_th in hough_peaks_blur]
        return [4, 5], [draw_line_on_im(np.zeros(im_noise.shape), lines_p_noise),
                        draw_line_on_im(np.zeros(blur_im_noise.shape), lines_p_blur)]

    slider_attr = [{'label': 'num peaks', 'valmin': 0, 'valmax': 99, 'valstep': 1},
                   {'label': 'hough peak threshold', 'valmin': 0.1, 'valmax': .99}]
    imshow([im_noise, blur_im_noise, edge_noise, edge_blur_noise, zeros, zeros],
           ['im_noise', 'blur_im_noise', 'edge_noise', 'edge_blur_noise', 'noise lines', 'blur lines'],
           slider_attr=slider_attr, slider_callback=[update_exp] * 2, shape=(3, 2))


def p4():
    im_filename = 'ps1-input1'
    im = imread_from_rep(im_filename, grey_scale=True)
    im_color = imread_from_rep(im_filename)
    blur_im = cv.GaussianBlur(im, (3, 3), 0)
    imsave(blur_im, 'output/ps1-4-a-1.png')
    edge = cv.Canny(blur_im, 100, 200)
    imsave(edge, 'output/ps1-4-b-1.png')
    theta_range = np.arange(-180, 180, 1)
    hough_acc = hough_lines_acc(edge, theta_range)
    hough_peaks = hough_peak_matlab_like(hough_acc, 4, 0.6)
    acc_peaks_highlighted = highlight_pos_im(hough_acc, hough_peaks, (10, 10))
    imsave(acc_peaks_highlighted, 'output/ps1-4-c-1.png')
    lines_param = [(d_th[0], theta_range[d_th[1]]) for d_th in hough_peaks]
    line_im = draw_line_on_im(np.zeros(im.shape), lines_param)
    imsave(line_im * 255, 'output/ps1-4-c-2.png')
    im_color = overlap_boolean_image(im_color, line_im > 0)
    imsave(im_color, 'output/ps1-4-c-4.png')
    imshow([edge, acc_peaks_highlighted, line_im, im_color])


def hough_circle_acc(edge, r):
    h = np.zeros(edge.shape)
    th_res = int(np.round(360 // (2 * np.pi * r)))
    for i in range(edge.shape[0]):
        for j in range(edge.shape[1]):
            if edge[i, j] > 0:
                for th in range(0, 360, th_res):
                    u = int(np.round(r * np.cos(th * np.pi / 180.0)))
                    v = int(np.round(r * np.sin(th * np.pi / 180.0)))
                    if is_in(edge.shape, i + u, j + v):
                        h[i + u, j + v] = h[i + u, j + v] + 1
    return h


def draw_circle(im, centers, r, onCopy=True, val=1):
    if onCopy:
        im = np.copy(im)
    th_res = int(np.round(360 // (2 * np.pi * r)))
    for center in centers:
        i = center[0]
        j = center[1]
        for th in range(0, 360, th_res):
            u = int(np.round(r * np.cos(th * np.pi / 180.0)))
            v = int(np.round(r * np.sin(th * np.pi / 180.0)))
            if is_in(im.shape, i + u, j + v):
                im[i + u, j + v] = val
    return im


def p5():
    im_filename = 'ps1-input1'
    im = imread_from_rep(im_filename, grey_scale=True)
    im = cv.GaussianBlur(im, (3, 3), 0)
    imsave(im, 'output/ps1-5-a-1.png')
    im_color = imread_from_rep(im_filename)
    r = 20
    edge = cv.Canny(im, 100, 200)
    imsave(edge, 'output/ps1-5-a-2.png')
    hough_acc = imfix_scale(hough_circle_acc(edge, r))
    peaks = hough_peak_matlab_like(hough_acc, 10, 0.7)
    highlighted = highlight_pos_im(hough_acc, peaks, (10, 10))
    drawn_over_im = draw_circle(np.zeros(im.shape), centers=peaks, r=r)
    drawn_over_color = overlap_boolean_image(im_color, drawn_over_im > 0)

    slider_attr = [{'label': 'threshold over accumulator', 'valmin': 0.1, 'valmax': .99, 'valstep': 0.01},
                   {'label': 'num peaks', 'valmin': 0, 'valmax': 99, 'valstep': 1}]

    def update_exp(x, axs, sliders, buttons):
        peaks = hough_peak_matlab_like(hough_acc, sliders[1].val, sliders[0].val)
        print(peaks)
        highlighted = highlight_pos_im(hough_acc, peaks, (10, 10))
        drawn_over_im = draw_circle(np.zeros(im.shape), centers=peaks, r=r)
        drawn_over_color = overlap_boolean_image(im_color, drawn_over_im > 0)
        return [1, 2, 3], [highlighted, drawn_over_im, drawn_over_color]

    imshow([edge, highlighted, drawn_over_im, drawn_over_color],
           ['edge', 'highlighted', 'drawn_over_im', 'drawn_over_color'],
           slider_attr=slider_attr, slider_callback=[update_exp] * 2)
    imsave(drawn_over_color, 'output/ps1-5-a-3.png')


def p5_with_unknown_radii():
    im_filename, r_range = [('tyre', range(28, 40)), ('ps1-input1', range(20, 50))][1]
    im = imread_from_rep(im_filename, grey_scale=True)
    im = cv.GaussianBlur(im, (3, 3), 0)
    im_color = imread_from_rep(im_filename)
    edge = cv.Canny(im, 100, 200)
    hough_acc_3d = np.concatenate([hough_circle_acc(edge, r)[:, :, np.newaxis] for r in r_range], axis=2)
    peaks = hough_peak_matlab_like(hough_acc_3d, 10, 0.5)
    im_drawn = np.zeros(im.shape)
    [draw_circle(im_drawn, [(peak[0], peak[1])], r=r_range[peak[2]], onCopy=False) for peak in peaks]
    im_drawn = overlap_boolean_image(im_color, im_drawn > 0)
    imsave(im_drawn, 'output/ps1-5-b-1.png')
    imshow(im_drawn)


def find_parallel_lines_from_hough_peaks(peaks, th_range, allowed_dist=np.PINF):
    parallel_set = []
    th_err = 3  # allowed fluctuations in parallel lines in degrees
    # convert to bin fluctuations
    th_err = int(np.round(th_err * len(th_range) / (th_range[-1] - th_range[0])))
    for i in range(len(peaks)):
        for j in range(i + 1, len(peaks)):
            if np.abs(peaks[i][1] - peaks[j][1]) <= th_err and np.abs(peaks[i][0] - peaks[j][0]) <= allowed_dist:
                parallel_set.append((peaks[i], peaks[j]))
    return parallel_set


def p6():
    """
    be wary before funning this. This is taking hell lot of time XD
    """
    im_filename, r_range = [('ps1-input2 copy', range(3, 9)), ('ps1-input2', range(30, 38))][1]
    im = imread_from_rep(im_filename, grey_scale=True)
    im_color = imread_from_rep(im_filename)
    blur_im = cv.GaussianBlur(im, (3, 3), 0)
    edge = cv.Canny(blur_im, 100, 200)
    theta_range = np.arange(-180, 180, 1)
    hough_acc = hough_lines_acc(edge, theta_range)
    hough_peaks = hough_peak_matlab_like(hough_acc, 10, 0.6)
    parallel_peak_pairs = find_parallel_lines_from_hough_peaks(hough_peaks, th_range=theta_range, allowed_dist=40)
    pen_lines = []
    [pen_lines.append(pair[0]) or pen_lines.append(pair[1]) for pair in parallel_peak_pairs]
    acc_peaks_highlighted = highlight_pos_im(hough_acc, pen_lines, (10, 10))
    lines_param = [(d_th[0], theta_range[d_th[1]]) for d_th in pen_lines]
    line_im = draw_line_on_im(np.zeros(im.shape), lines_param)
    im_color = overlap_boolean_image(im_color, line_im > 0, color_val=(255, 255, 0))
    imsave(im_color, 'output/ps1-6-c-1.png')
    imshow([edge, acc_peaks_highlighted, im_color])


def p7():
    im_filename, r_range = [('ps1-input1', range(20, 50)), ('ps1-input2', range(30, 38))][1]
    im = imread_from_rep(im_filename, grey_scale=True)
    im_color = imread_from_rep(im_filename)
    blur_im = cv.GaussianBlur(im, (3, 3), 0)
    edge = cv.Canny(blur_im, 100, 200)
    hough_acc_3d = np.concatenate([hough_circle_acc(edge, r)[:, :, np.newaxis] for r in r_range], axis=2)
    peaks = hough_peak_matlab_like(hough_acc_3d, 10, 0.5)
    im_drawn = np.zeros(im.shape)
    [draw_circle(im_drawn, [(peak[0], peak[1])], r=r_range[peak[2]], onCopy=False) for peak in peaks]
    im_color = overlap_boolean_image(im_color, im_drawn > 0, color_val=(255, 255, 0))
    imsave(im_color, 'output/ps1-7-a-1.png')
    imshow([edge, im_color])


def p8():
    im_filename, r_range = [('ps1-input1', range(20, 50)), ('ps1-input3 copy', range(10, 15))][1]
    im = imread_from_rep(im_filename, grey_scale=True)
    im_color = imread_from_rep(im_filename)
    blur_im = cv.GaussianBlur(im, (3, 3), 0)
    edge = cv.Canny(blur_im, 120, 290)
    hough_acc_s = [cv.GaussianBlur(hough_circle_acc(edge, r), (9, 9), 0)[:, :, np.newaxis] for r in r_range]
    imshow(hough_acc_s)
    hough_acc_3d = np.concatenate(hough_acc_s, axis=2)
    peaks = hough_peak_matlab_like(hough_acc_3d, 10, 0.5, (20, 20, 3))
    im_drawn = np.zeros(im.shape)
    [draw_circle(im_drawn, [(peak[0], peak[1])], r=r_range[peak[2]], onCopy=False) for peak in peaks]
    im_color = overlap_boolean_image(im_color, im_drawn > 0, color_val=(255, 255, 0))
    imsave(im_color, 'output/ps1-8-c-1.png')
    imshow([edge, im_color])


""" Run just the methods you need, else you will go through a lot of interactive canvas popups """
# testing_imshow()
# p1()
# p2_experiment()
# p2_experiment_load_houghacc()
# p2_experiment_peak_finding()
# p3()
# p4()
# p5()
# p5_with_unknown_radii()
# p6()
# p7()
# p8()
