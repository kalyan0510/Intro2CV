import cv2 as cv
import numpy as np

from ps_1.helper import hough_lines_acc, draw_line_on_im, hough_peak_custom, hough_peak_matlab_like, \
    non_max_suppression, hough_circle_acc, draw_circle, find_parallel_lines_from_hough_peaks
from ps_hepers.helpers import imread_from_rep, imread, imshow, imsave, is_in, imfix_scale, highlight_pos_im, \
    overlap_boolean_image, mark_points, np_load, np_save

"""
Problem Set - 1
Problems at : https://docs.google.com/document/d/13CJgtDr8kIX9KIrs6BYFDF6-N7cfAyX0R54v8CWoqmQ/pub?embedded=true
"""


def testing_imshow():
    """
    A test method for imshow() in ps_hepers.helpers
    Use the interactive imshow() to display one of two input images in the 8 image holders
    This method is not related to PS-1
    """
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


def p1():
    # Canny Detection - https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
    # 2d explanation on Non-maximal suppression & Hysteresis Thresholding -
    # http://www.justin-liang.com/tutorials/canny/#:~:text=Non%20maximum%20suppression%20works%20by,the%20gradient%20direction%20of%20q
    im = imread_from_rep('ps1-input0', grey_scale=True)
    edges = cv.Canny(im, 100, 200)
    slider_attr = [{'label': 'threshold1', 'valmin': 0, 'valmax': 300, 'valint': 100},
                   {'label': 'threshold2', 'valmin': 0, 'valmax': 300, 'valint': 200}]

    def update_threshold(x, axs, sliders, buttons):
        print(sliders[0].val, sliders[1].val)
        return [1], [cv.Canny(im, sliders[0].val, sliders[1].val)]

    imshow([im, edges], ['original', 'canny edges'], slider_attr=slider_attr, slider_callback=[update_threshold] * 2)
    imsave(edges, 'output/ps1-1-a-1.png')


def p2():
    image_filename = 'ps1-input0'
    im = imread_from_rep(image_filename, grey_scale=True)
    # a
    # load color version to overlay the detected lines
    im_color = imread_from_rep(image_filename)
    # get edges with canny
    edges = cv.Canny(im, 100, 200)
    # compute accumulator array
    th_range = np.arange(-180, 180, 1)
    hough_acc = hough_lines_acc(edges, th_range)
    imsave(imfix_scale(hough_acc), 'output/ps1-2-a-1.png')
    # b
    # using custom peak finding, but the method is copied so that intermediate steps can be visualized in the
    # interactive imshow figure
    hough_acc_w = cv.copyMakeBorder(hough_acc, 3, 3, 3, 3, cv.BORDER_WRAP)
    blur_acc = cv.GaussianBlur(hough_acc_w, (5, 5), 0)[3:-3, 3:-3]
    th_blur_acc = blur_acc * (blur_acc > np.max(blur_acc) * 0.90)
    nms_th_blur_acc = non_max_suppression(th_blur_acc)
    peak_highlighted_acc = overlap_boolean_image(imfix_scale(hough_acc), nms_th_blur_acc, cross_marks=True)
    imsave(peak_highlighted_acc, 'output/ps1-2-b-1.png')
    # c
    ds, ths = np.where(nms_th_blur_acc > 0)
    # convert the th indexes to theta range
    line_params = [(d, th_range[th]) for d, th in zip(ds, ths)]
    # draw lines over new image and merge it into red channel of colored input image
    line_im = draw_line_on_im(np.zeros(im.shape), line_params)
    im_color[:, :, 1] = im_color[:, :, 1] + line_im * 255.0
    im_color_to_render = np.copy(im_color)
    im_color_to_render[line_im > 0, :] = 0
    im_color_to_render[line_im > 0, 0] = 255
    imsave(im_color_to_render, 'output/ps1-2-c-1.png')
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
        im_color_to_render[line_im > 0, :] = 0
        im_color_to_render[line_im > 0, 0] = 255
        return [3, 4, 5, 6], [th_blur_acc, overlap_boolean_image(nms_th_blur_acc, nms_th_blur_acc, cross_marks=True),
                              line_im, im_color_to_render]

    imshow([edges, hough_acc, blur_acc, th_blur_acc, peak_highlighted_acc, line_im, im_color_to_render],
           ['edges', 'hough_acc', 'blur_acc', 'th_blur_acc', 'nms_th_blur_acc', 'line_im', 'colored'],
           interpolation='nearest', slider_attr=slider_attr, slider_callback=[slider_update])
    imsave(imfix_scale(hough_acc), 'observations/hough_acc.png')
    # d
    """
    What parameters did you use for finding lines in this image?
    The parameters are as follows:
    Canny thresholds: 100, 200
    Hough accumulator sizes: theta is [-180, 180] and d is [0, #pixelsondiagonal]
    Hough peaks method: Simple non maximal suppression with window size 3 and threshold 90% of max 
    """


def p2_experiment_load_houghacc():
    im = imread('observations/hough_acc.png', grey_scale=True)
    im_w = cv.copyMakeBorder(im, 3, 3, 3, 3, cv.BORDER_WRAP)
    imblur = cv.GaussianBlur(im_w, (5, 5), 0)
    imblur = imblur[3:-3, 3:-3]
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
        return [0, 1], [draw_line_on_im(np.zeros(im.shape), lines_p_c), draw_line_on_im(np.zeros(im.shape), lines_p_m)]

    imshow([draw_line_on_im(np.zeros(im.shape), lines_p_c), draw_line_on_im(np.zeros(im.shape), lines_p_m)],
           ['custom hough peak', 'matlab\'s hough peak'], slider_attr=slider_attr, slider_callback=[update_exp] * 2)


def p3():
    image_filename = 'ps1-input0-noise'
    im_noise = imread_from_rep(image_filename)
    im_color = imread_from_rep(image_filename)
    # a
    blur_im_noise = cv.GaussianBlur(im_noise, (7, 7), 4)
    imsave(blur_im_noise, 'output/ps1-3-a-1.png')
    # b
    edge_noise = cv.Canny(im_noise, 100, 200)
    edge_blur_noise = cv.Canny(blur_im_noise, 100, 200)
    imsave(edge_noise, 'output/ps1-3-b-1.png')
    imsave(edge_blur_noise, 'output/ps1-3-b-2.png')
    # c
    # dummy init image to show before adjusting params
    zeros = np.zeros(im_noise.shape)
    th_range = np.arange(-180, 180, 1)
    # noise_hough_acc = hough_lines_acc(edge_noise, th_range)
    blur_hough_acc = hough_lines_acc(edge_blur_noise, th_range)
    hough_peaks_blur = hough_peak_matlab_like(blur_hough_acc, 10, 0.28, np.asarray(blur_hough_acc.shape) // 10)
    lines_p_blur = [(d_th[0], th_range[d_th[1]]) for d_th in hough_peaks_blur]
    lines_drawn = draw_line_on_im(np.zeros(blur_im_noise.shape), lines_p_blur)
    imsave(mark_points(blur_hough_acc, hough_peaks_blur), 'output/ps1-3-c-1.png')
    im_color[lines_drawn[:, :, 0] > 0, :] = [255, 0, 0]
    imsave(im_color, 'output/ps1-3-c-2.png')

    # interactively adjust parameters used in this method using sliders
    def update_exp(x, axs, sliders, buttons):
        hough_peaks_blur = hough_peak_matlab_like(blur_hough_acc, sliders[0].val, sliders[1].val,
                                                  np.asarray(blur_hough_acc.shape) // 10)
        lines_p_blur = [(d_th[0], th_range[d_th[1]]) for d_th in hough_peaks_blur]
        return [4], [draw_line_on_im(np.zeros(blur_im_noise.shape), lines_p_blur)]

    slider_attr = [{'label': 'num peaks', 'valmin': 0, 'valmax': 99, 'valstep': 1},
                   {'label': 'hough peak threshold', 'valmin': 0.1, 'valmax': .99}]
    imshow([im_noise, blur_im_noise, edge_noise, edge_blur_noise, zeros],
           ['im_noise', 'blur_im_noise', 'edge_noise', 'edge_blur_noise', 'noise lines', 'blur lines'], shape=(3, 2),
           slider_attr=slider_attr, slider_callback=[update_exp] * 2)
    """
    What you had to do to get the best result you could?
    Just adjusting the blur params (window size and sigma) to smoothen all the noise does the job
    """


def p4():
    im_filename = 'ps1-input1'
    im = imread_from_rep(im_filename, grey_scale=True)
    im_color = imread_from_rep(im_filename)
    # a
    blur_im = cv.GaussianBlur(im, (3, 3), 0)
    imsave(blur_im, 'output/ps1-4-a-1.png')
    # b
    edge = cv.Canny(blur_im, 100, 200)
    imsave(edge, 'output/ps1-4-b-1.png')
    # c
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


def p5_a():
    im_filename = 'ps1-input1'
    im = imread_from_rep(im_filename, grey_scale=True)
    im = cv.GaussianBlur(im, (3, 3), 0)
    imsave(im, 'output/ps1-5-a-1.png')
    im_color = imread_from_rep(im_filename)
    r = 20
    edge = cv.Canny(im, 290, 290)
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
        highlighted = highlight_pos_im(hough_acc, peaks, (10, 10))
        drawn_over_im = draw_circle(np.zeros(im.shape), centers=peaks, r=r)
        drawn_over_color = overlap_boolean_image(im_color, drawn_over_im > 0)
        return [1, 2, 3], [highlighted, drawn_over_im, drawn_over_color]

    imshow([edge, highlighted, drawn_over_im, drawn_over_color],
           ['edge', 'highlighted', 'drawn_over_im', 'drawn_over_color'],
           slider_attr=slider_attr, slider_callback=[update_exp] * 2)
    imsave(drawn_over_color, 'output/ps1-5-a-3.png')


def p5_b():
    """
    be wary before running this without the files ('observations/hough_acc_coins_%s_%s.npy' % (im_filename, r_range)).
    This is taking hell lot of time
    """
    im_filename, r_range = [('tyre', range(28, 40)), ('ps1-input1', range(20, 50))][1]
    im = imread_from_rep(im_filename, grey_scale=True)
    im = cv.GaussianBlur(im, (3, 3), 0)
    im_color = imread_from_rep(im_filename)
    edge = cv.Canny(im, 290, 290)
    pre_computed_acc_path = 'observations/hough_acc_coins_%s_%s.npy' % (im_filename, r_range)
    hough_acc_3d = np_load(1, pre_computed_acc_path)
    if hough_acc_3d is None:
        hough_acc_3d = np.concatenate([hough_circle_acc(edge, r)[:, :, np.newaxis] for r in r_range], axis=2)
        np_save([hough_acc_3d], pre_computed_acc_path)
    print(hough_acc_3d.shape)
    peaks = hough_peak_matlab_like(hough_acc_3d, 14, 0.4, (50, 50, 20))
    im_drawn = np.zeros(im.shape)
    [draw_circle(im_drawn, [(peak[0], peak[1])], r=r_range[peak[2]], on_copy=False) for peak in peaks]
    im_drawn = overlap_boolean_image(im_color, im_drawn > 0)
    imsave(im_drawn, 'output/ps1-5-b-1.png')
    imshow(im_drawn)
    """
    What you had to do to find circles?
    1. Adjust the canny edge thresholds to remove any noisy edges
    2. customize shape of hough peak detector, so no spurious circles are detected
    """


def p6():
    """
    be wary before running this with out the file 'observations/hough_line_acc_%s_%s.npy' % (im_filename, r_range)
    """
    # a
    im_filename, r_range = [('ps1-input2 copy', range(3, 9)), ('ps1-input2', range(30, 38))][1]
    im = imread_from_rep(im_filename, grey_scale=True)
    im_color = imread_from_rep(im_filename)
    blur_im = cv.GaussianBlur(im, (3, 3), 0)
    edge = cv.Canny(blur_im, 100, 200)
    theta_range = np.arange(-180, 180, 1)
    pre_computed_acc_path = 'observations/hough_line_acc_%s_%s.npy' % (im_filename, r_range)
    hough_acc = np_load(1, pre_computed_acc_path)
    if hough_acc is None:
        hough_acc = hough_lines_acc(edge, theta_range)
        np_save([hough_acc], pre_computed_acc_path)
    hough_peaks = hough_peak_matlab_like(hough_acc, 10, 0.6)
    lines_p_m = [(d_th[0], theta_range[d_th[1]]) for d_th in hough_peaks]
    line_im = draw_line_on_im(np.zeros(im.shape), lines_p_m)
    imsave(overlap_boolean_image(im_color, line_im > 0, color_val=(255, 255, 0)), 'output/ps1-6-a-1.png')
    # b
    """
    Likely the last step found lines that are not the boundaries of the pens. What are the problems present?
    
    Hough lines detect all the edges that run straight. They need not be of the pens. So, we should try finding lines
    that are parallel to each other and are near by each other.
    This logic might not work in production (real life) but will work in this example as all the pens in the image are
    not nearby and parallel to each other.  
    """
    # c
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
    # a
    im_filename, r_range = [('ps1-input1', range(20, 50)), ('ps1-input2', range(30, 38))][1]
    im = imread_from_rep(im_filename)[:, :, 1]
    im_color = imread_from_rep(im_filename)
    blur_im = cv.GaussianBlur(im, (5, 5), 3)
    edge = cv.Canny(blur_im, 100, 150)
    pre_computed_acc_path = 'observations/hough_circle_acc_%s_%s.npy' % (im_filename, r_range)
    hough_acc_3d = np_load(1, pre_computed_acc_path)
    if hough_acc_3d is None:
        hough_acc_3d = np.concatenate([hough_circle_acc(edge, r)[:, :, np.newaxis] for r in r_range], axis=2)
        np_save([hough_acc_3d], pre_computed_acc_path)
    hough_acc_3d = (hough_acc_3d / hough_acc_3d.max()) * 255
    peaks = hough_peak_matlab_like(hough_acc_3d, 11, 0.3, (50, 50, 8))
    im_drawn = np.zeros(im.shape)
    [draw_circle(im_drawn, [(peak[0], peak[1])], r=r_range[peak[2]], on_copy=False) for peak in peaks]
    im_color = overlap_boolean_image(im_color, im_drawn > 0, color_val=(255, 255, 0))
    imsave(im_color, 'output/ps1-7-a-1.png')
    imshow([edge, im_color])
    # b
    """
    Are there any false alarms? How would/did you get rid of them?
    Yes. 
    One way is to eliminate false votes caused by clusters of spurious edge pixels. This can be done by n-correlating 
    hough accumulator with normalized gaussian filter.
    This will eliminate dense accumulator regions which are not distributed towards a center.  
    """


def p8():
    im_filename, r_range = [('ps1-input3', range(30, 40))][0]
    # a
    im = imread_from_rep(im_filename, grey_scale=True)
    im_color = imread_from_rep(im_filename)
    src = np.array([
        [0, 0],
        [im.shape[1] - 1, 0],
        [im.shape[1] - 1, im.shape[0] - 1],
        [0, im.shape[0] - 1]], dtype="float32")
    dst = np.array([
        [120, 33],
        [542, 22],
        [682, 280],
        [0, 280]], dtype="float32")
    im = cv.warpPerspective(im, cv.getPerspectiveTransform(dst, src), (im.shape[1], im.shape[0]))
    imshow(im)
    blur_im = cv.GaussianBlur(im, (3, 3), 0)
    edge = cv.Canny(blur_im, 120, 230)
    imshow(edge)
    pre_computed_acc_path = 'observations/hough_circle_acc_%s_%s.npy' % (im_filename, r_range)
    hough_acc_3d = np_load(1, pre_computed_acc_path)
    if hough_acc_3d is None:
        hough_acc_s = [cv.GaussianBlur(hough_circle_acc(edge, r), (9, 9), 0)[:, :, np.newaxis] for r in r_range]
        imshow(hough_acc_s)
        hough_acc_3d = np.concatenate(hough_acc_s, axis=2)
        np_save([hough_acc_3d], pre_computed_acc_path)
    peaks = hough_peak_matlab_like(hough_acc_3d, 15, 0.1, (100, 100, 10))
    im_drawn = np.zeros(im.shape)
    [draw_circle(im_drawn, [(peak[0], peak[1])], r=r_range[peak[2]], on_copy=False) for peak in peaks]
    im_color = cv.warpPerspective(im_color, cv.getPerspectiveTransform(dst, src), (im.shape[1], im.shape[0]))
    im_color = overlap_boolean_image(im_color, im_drawn > 0, color_val=(255, 255, 0))
    im_color = cv.warpPerspective(im_color, cv.getPerspectiveTransform(src, dst), (im.shape[1], im.shape[0]))
    imsave(im_color, 'output/ps1-8-c-1.png')
    imshow([edge, im_color])
    # b
    """
    What might you do to fix the circle problem?
    Warping using known homography is a hack I did to find the circles very aligned. In real usage, if we do not know 
    the homography transformation, we might have to find it.
    
    Another way of fixing the circle problem is to do fuzzy voting on the accumulator array. That is, we add a vote
     for all accumulator cells containing a vote. 
     That is we can blur the accumulator with a flat or gaussian kernel to distribute the votes and then we find the 
     centroid of dense distributions (which might look like fuzzy ellipses). From the shape of the distribution, we 
     derive the warping of coins and detect their contours.          
    """


if __name__ == '__main__':
    """ Run just the methods that are needed, else the session will go through a lot of interactive canvas popups """
    # testing_imshow()
    p1()
    # p2()
    # p2_experiment_load_houghacc()
    # p2_experiment_peak_finding()
    # p3()
    # p4()
    # p5_a()
    # p5_b()
    # p6()
    # p7()
    # p8()
    # p8_exp()
