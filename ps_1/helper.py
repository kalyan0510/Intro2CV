import numpy as np
import cv2 as cv

from ps_hepers.helpers import is_in


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
    threshold_val = acc.min() * (1 - threshold) + acc.max() * threshold
    while len(pts) < npeaks:
        index = np.unravel_index(np.asarray(acc).argmax(), acc.shape)
        if len(index) == 2 and acc[index[0], index[1]] >= threshold_val and acc[index[0], index[1]] > 0:
            pts.append((index[0], index[1]))
            acc[np.ix_(round_range(index[0] - bounds[0], index[0] + bounds[0] + 1, acc.shape[0]),
                       round_range(index[1] - bounds[1], index[1] + bounds[1] + 1, acc.shape[1]))] = 0
        elif len(index) == 3 and acc[index[0], index[1], index[2]] >= threshold_val and acc[
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


# def l_o_g(im, th):
#     """
#     Experimental edge detection using laplacian with zero crossings
#     """
#     im_log = cv.Laplacian(im, cv.CV_16S)
#     min_log = cv.morphologyEx(im_log, cv.MORPH_ERODE, np.ones((3, 3)))
#     max_log = cv.morphologyEx(im_log, cv.MORPH_DILATE, np.ones((3, 3)))
#     crossings = np.logical_or(np.logical_and(min_log < 0, im_log > 0), np.logical_and(max_log > 0, im_log < 0))
#     return imfix_scale(crossings & im)


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


def draw_circle(im, centers, r, on_copy=True, val=1):
    if on_copy:
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
