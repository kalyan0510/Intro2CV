import cv2 as cv
import numpy
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


def imread(filename, channels=None, grey_scale=None):
    # setting defaults
    channels = [channels, (0, 1, 2)][channels is None]
    grey_scale = [grey_scale, False][grey_scale is None]
    # cv imread
    img = cv.imread(filename)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if grey_scale else img[:, :, channels] if len(
        img.shape) == 3 else np.concatenate([img[:, :, np.newaxis]] * 3)


def imread_from_rep(image_name, **kwargs):
    channels = kwargs.get('channels', None)
    grey_scale = kwargs.get('grey_scale', None)
    in_cur_dir = Path("input/" + image_name + '.png')
    if in_cur_dir.is_file():
        path = in_cur_dir
    else:
        path = Path("../images/" + image_name + '.png')
    if not path.is_file():
        raise Exception('no image found at file path %s' % path)
    return imread(str(path.resolve()), channels, grey_scale)


def imsave(img, file_name):
    cv.imwrite(file_name, img)


def imfix_scale(img):
    return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(int)


def is_in(im_sz, i, j):
    return 0 <= i < im_sz[0] and 0 <= j < im_sz[1]


def imshow(im, im_title=None, shape=None, interpolation='bilinear', sup_title='Figure(s)', slider_attr=None,
           slider_callback=None,
           button_attr=None, button_callback=None):
    if not type(im) == list:
        im = [im]
    # determine squarish shape to arrange all images
    shape = [shape, (int(np.sqrt(len(im))), int(np.ceil(len(im) / int(np.sqrt(len(im))))))][shape is None]
    im_title = [im_title, 'Image'][im_title is None]
    fig, axs = plt.subplots(shape[0], shape[1])
    if not type(axs) == list:
        axs = numpy.asarray([axs])
    axs = axs.flatten()
    # make ticks and axes(axis) disappear
    for ax in axs:
        ax.set_axis_off()
    # plot each image in its axes
    for i in range(len(im)):
        axs[i].imshow(im[i], interpolation=interpolation)
        axs_title = '%s $%sx%s$  $mn=%s$  $mx=%s$ ' % (
            im_title if not type(im_title) == list else im_title[i], im[i].shape[0], im[i].shape[1], np.min(im[i]),
            np.max(im[i]))
        axs[i].set_title(axs_title, fontweight='bold')
        axs[i].set_axis_off()
    # create widgets to interact with images
    num_sliders = 0 if slider_attr is None else len(slider_attr)
    num_buttons = 0 if button_attr is None else len(button_attr)

    widget_width = 0.05
    fig.subplots_adjust(bottom=widget_width * (num_buttons + num_sliders + 1))

    def create_slider(i):
        slider_ax = fig.add_axes([0.2, widget_width * (num_sliders - i), 0.65, 0.03], facecolor='grey')

        slider = Slider(slider_ax, slider_attr[i].get('label', '%s' % i), slider_attr[i].get('valmin', 0),
                        slider_attr[i].get('valmax', 1),
                        slider_attr[i].get('valint', slider_attr[i].get('valmin', 0)), color='#6baeff')
        slider.on_changed(lambda x: update_images(x, slider_callback[i]))
        return slider

    def create_button(i):
        button_ax = fig.add_axes([0.75, widget_width * (num_sliders) + widget_width * (num_buttons - i), 0.1, 0.03],
                                 facecolor='grey')
        button = Button(button_ax, button_attr[i].get('label', '%s' % i), color='0.99', hovercolor='0.575')
        button.on_clicked(lambda x: update_images(x, button_callback[i]))
        return button

    # create sliders and store them in memory
    sliders = list(map(create_slider, range(num_sliders)))
    # create buttons and store them in memory
    buttons = list(map(create_button, range(num_buttons)))

    # method that is called when a slider or button is touched. This method in turn
    # calls the callbacks to get the updated images and put them in the plot
    def update_images(x, callback):
        updates = callback(x, axs, sliders, buttons)
        if updates is not None and type(updates) == tuple and len(updates) > 0:
            updated_i_s = updates[0]
            updated_im_s = updates[1]
            for u_i, u_im in zip(updated_i_s, updated_im_s):
                if u_i < len(im):
                    axs[u_i].imshow(u_im, interpolation=interpolation)

    # set main title
    fig.canvas.manager.set_window_title(sup_title)
    plt.suptitle(sup_title)
    # bigger viewing area
    fig.set_size_inches(2 * fig.get_size_inches())
    plt.show()


def testing_imshow():
    im = imread_from_rep('ps1-input0')
    im2 = imread_from_rep('lena')

    slider_attr = [{'label': 'Sets to Lena', 'valmin': 0, 'valmax': 10, 'valint': 5},
                   {'label': 'Sets to squares', 'valmin': 0, 'valmax': 20, 'valint': 15}]
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
#     im_log = cv.Laplacian(im, cv.CV_16S)
#     min_log = cv.morphologyEx(im_log, cv.MORPH_ERODE, np.ones((3, 3)))
#     max_log = cv.morphologyEx(im_log, cv.MORPH_DILATE, np.ones((3, 3)))
#     crossings = np.logical_or(np.logical_and(min_log < 0, im_log > 0), np.logical_and(max_log > 0, im_log < 0))
#     return imfix_scale(crossings & im)


def p1():
    # Canny Detection - https://docs.opencv.org/master/da/d22/tutorial_py_canny.html
    # 2d explanation on Non-maximal suppression & Hysteresis Thresholding -
    # http://www.justin-liang.com/tutorials/canny/#:~:text=Non%20maximum%20suppression%20works%20by,the%20gradient%20direction%20of%20q
    im = imread_from_rep('ps1-input0-noise', grey_scale=True)
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


def hough_peak_like_matlab(acc, npeaks=1, threshold=0.5, size=None):
    acc = np.copy(acc)

    def round_range(start, end, max):
        return [range(start, end), list(range(start, max)) + list(range(0, end % max))][int(end) >= int(max)]

    size = [size, np.asarray(acc.shape) // 50][size is None]
    size = ([size[0] + 1, size[0]][size[0] % 2], [size[1] + 1, size[1]][size[1] % 2])
    i_bound = (size[0] - 1) // 2
    j_bound = (size[1] - 1) // 2
    pts = []
    max_peak = np.max(acc)
    while len(pts) <= npeaks:
        i, j = np.unravel_index(np.asarray(acc).argmax(), acc.shape)
        if acc[i, j] >= threshold * max_peak and acc[i, j] > 0:
            pts.append((i, j))
            acc[np.ix_(round_range(i - i_bound, i + i_bound + 1, acc.shape[0]),
                       round_range(j - j_bound, j + j_bound + 1, acc.shape[1]))] = 0
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
    hough_peaks_m = hough_peak_like_matlab(hough_acc, 6, 0.7)
    lines_p_m = [(d_th[0], th_range[d_th[1]]) for d_th in hough_peaks_m]

    slider_attr = [{'label': 'threshold over accumulator', 'valmin': 0.1, 'valmax': .99, 'valstep': 0.01},
                   {'label': 'num peaks', 'valmin': 0, 'valmax': 99, 'valstep': 1}]

    def update_exp(x, axs, sliders, buttons):
        hough_peaks_c = hough_peak_custom(hough_acc, sliders[0].val)
        lines_p_c = [(d_th[0], th_range[d_th[1]]) for d_th in hough_peaks_c]
        hough_peaks_m = hough_peak_like_matlab(hough_acc, int(sliders[1].val), sliders[0].val)
        lines_p_m = [(d_th[0], th_range[d_th[1]]) for d_th in hough_peaks_m]
        # print('updated', len(lines_p_m))
        # print(lines_p_m)
        return [0, 1], [draw_line_on_im(np.zeros(im.shape), lines_p_c), draw_line_on_im(np.zeros(im.shape), lines_p_m)]

    imshow([draw_line_on_im(np.zeros(im.shape), lines_p_c), draw_line_on_im(np.zeros(im.shape), lines_p_m)],
           ['custom hough peak', 'matlab\'s hough peak'],
           slider_attr=slider_attr, slider_callback=[update_exp] * 2)


testing_imshow()
p1()
p2_experiment()
p2_experiment_load_houghacc()
p2_experiment_peak_finding()
