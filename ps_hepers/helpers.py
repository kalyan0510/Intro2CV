import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def imread(filename, channels=None, grey_scale=None):
    """
    Simple imread that uses cv.imread to read and return an uint8 array
    :param filename: name of file
    :param channels: channels to return eg.: [0,1] to return image with two channels R & G
    :param grey_scale: method returns grey scale image if this is true
    :return: np loaded image
    """
    # setting defaults
    channels = [channels, (0, 1, 2)][channels is None]
    grey_scale = [grey_scale, False][grey_scale is None]
    # cv imread
    img = cv.imread(filename)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if grey_scale \
        else cv.cvtColor(img, cv.COLOR_BGR2RGB)[:, :, channels] if len(img.shape) == 3 \
        else np.concatenate([img[:, :, np.newaxis]] * 3, axis=2)


def imread_from_rep(image_name, **kwargs):
    """
    Abstraction over above imread method. This searches for given file in two dirs
    Firstly in the current dir's 'input' folder and then the global 'images' dir
    :param image_name: name of file without extension name (ext defaults to .png)
    :param kwargs: other args that needs to be passed to imread
    :return: np loaded image
    """
    channels = kwargs.get('channels', None)
    grey_scale = kwargs.get('grey_scale', None)
    ext = kwargs.get('extension', '.png')
    in_cur_dir = Path("input/" + image_name + ext)
    if in_cur_dir.is_file():
        path = in_cur_dir
    else:
        path = Path("../images/" + image_name + ext)
    if not path.is_file():
        raise Exception('no image found at file path %s' % path)
    return imread(str(path.resolve()), channels, grey_scale)


def imsave(img, path_name):
    """
    Saves the image as file at mentioned path
    :param img: np array 2d or 3d(with 3 channels)
    :param path_name: path of file
    :return: None
    """
    img_float32 = np.float32(img)
    cv.imwrite(path_name, img_float32 if len(img.shape) == 2 else cv.cvtColor(img_float32, cv.COLOR_RGB2BGR))


def imfix_scale(img):
    """
    Fixes the range of each pixels to 0 and 255
    :param img: image whose scale is not in [0,255]
    :return: pixel scaled image
    """
    return ((img - np.min(img)) / (np.max(img) - np.min(img)) * 255).astype(int)


def is_in(im_shape, i, j):
    """
    Helper method that return true iff (i,j) is inside if image with shape im_sz
    :param im_shape: shape of im
    :param i: i
    :param j: j
    :return: true iff (i,j) is inside if image
    """
    return 0 <= i < im_shape[0] and 0 <= j < im_shape[1]


def imshow(im, im_title=None, shape=None, interpolation='bilinear', cmap=None, sup_title='Figure(s)', slider_attr=None,
           slider_callback=None,
           button_attr=None, button_callback=None):
    """
    My favorite method that abstracts all the matplotlib code when building an interactive canvas that have as many
    sliders and buttons which can alter what is shown on the plots
    This method opens a new window that displays image(s) using matplotlib. Optionally the methods accepts params that
    define interactive sliders and buttons in the canvas
    :param im: single image or list of images (greyscale or multi channel)
    :param im_title: single string or list of strings defining the title for each image in $im
    :param shape: shape of subplots eg.: (3,1) makes fig to have 3 subplots one above the other
    :param interpolation: interpolation logic eg.: 'nearest', 'bicubic', 'bilinear', ..
    :param cmap: color map of the plot
    :param sup_title: main title of the entire figure
    :param slider_attr: a list of set of slider attributes. one set for each slider that defines valmax, valmin, valint
    :param slider_callback: a list of callback methods each accepting the params {event, axs, sliders, buttons} which
    define the state of figure/canvas
    :param button_attr: a list of set of button attributes. one for each slider that defines {'label'}
    :param button_callback: a list of callback methods each accepting the params {event, axs, sliders, buttons} which
    define the state of figure/canvas
    :return: None
    """
    # TODO : This method has become bulky. Work on simplification.
    if not type(im) == list:
        im = [im]
    # determine squarish shape to arrange all images
    shape = [shape, (int(np.sqrt(len(im))), int(np.ceil(len(im) / int(np.sqrt(len(im))))))][shape is None]
    im_title = [im_title, 'Image'][im_title is None]
    fig, axs = plt.subplots(shape[0], shape[1])
    if not type(axs) == list:
        axs = np.asarray([axs])
    axs = axs.flatten()
    # make ticks and axes(axis) disappear
    for ax in axs:
        ax.set_axis_off()
    # plot each image in its axes
    for i in range(len(im)):
        if cmap is not None:
            axs[i].imshow(im[i], interpolation=interpolation, cmap=cmap)
        else:
            axs[i].imshow(im[i], interpolation=interpolation)
        axs_title = '%s $%sx%s$\n$mn=%.3f$  $mx=%.3f$ ' % (
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
        button.on_clicked(lambda event: update_images(event, button_callback[i]))
        return button

    # create sliders and store them in memory
    sliders = list(map(create_slider, range(num_sliders)))
    # create buttons and store them in memory
    buttons = list(map(create_button, range(num_buttons)))

    # method that is called when a slider or button is touched. This method in turn
    # calls the callbacks to get the updated images and put them in the plot
    def update_images(event, callback):
        updates = callback(event, axs, sliders, buttons)
        if updates is not None and type(updates) == tuple and len(updates) > 0:
            updated_i_s = updates[0]
            updated_im_s = updates[1]
            for u_i, u_im in zip(updated_i_s, updated_im_s):
                if u_i < len(im):
                    if cmap is not None:
                        axs[u_i].imshow(u_im, interpolation=interpolation, cmap=cmap)
                    else:
                        axs[u_i].imshow(u_im, interpolation=interpolation)

    # set main title
    fig.canvas.manager.set_window_title(sup_title)
    plt.suptitle(sup_title)
    # bigger viewing area
    fig.set_size_inches(2 * fig.get_size_inches())
    plt.show()


def highlight_pos_im(im, points, size, highlight_val=255):
    """
    Highlights an image with rectangular boxes having centers defined by the input param 'points'
    :param im: input image
    :param points: list of tubles that define the center to highlight
    :param size: shape/size of the rectangular box
    :param highlight_val: value put over the pixels of rectangle (defaults to 255/white)
    :return: image with rectangular grids
    """
    im = np.copy(im)
    if not (type(points) == list or type(points) == np.ndarray and len(points.shape) == 1):
        points = [points]

    def round_range(start, end, max):
        return [range(start, end), list(range(start, max)) + list(range(0, end % max))][int(end) >= int(max)]

    for pos in points:
        im[round_range(pos[0] - size[0] // 2, pos[0] + size[0] // 2 + 1, im.shape[0]), (pos[1] - size[1] // 2) %
           im.shape[1]] = highlight_val
        im[round_range(pos[0] - size[0] // 2, pos[0] + size[0] // 2 + 1, im.shape[0]), (pos[1] + size[1] // 2) %
           im.shape[1]] = highlight_val
        im[(pos[0] - size[0] // 2) % im.shape[0], round_range(pos[1] - size[1] // 2, pos[1] + size[1] // 2 + 1,
                                                              im.shape[1])] = highlight_val
        im[(pos[0] + size[0] // 2) % im.shape[0], round_range(pos[1] - size[1] // 2, pos[1] + size[1] // 2 + 1,
                                                              im.shape[1])] = highlight_val
    return im


def overlap_boolean_image(im, boolean_im, val=255, color_val=(255, 0, 0), cross_marks=False):
    """
    The boolean array is projected into the input image (grey scale or color)
    :param im: image (grey scale or color)
    :param boolean_im: boolean image
    :param val: greyscale value used for projection onto image
    :param color_val: color value used for projection onto image
    :param cross_marks: if Ture, marks pts on boolean im with a cross
    :return: overlapped image
    """
    im = np.copy(im)
    if cross_marks:
        boolean_im = cv.filter2D(boolean_im, -1, np.diag(np.ones(7)) + np.diag(np.ones(7))[::-1])
    if len(im.shape) == 2:
        im[boolean_im > 0] = val
    elif len(im.shape) == 3:
        im[boolean_im > 0, 0:3] = color_val
    return im


def mark_points(im, points, color_val=(255, 0, 0)):
    ij_s = np.asarray(points)
    i_s = ij_s[:, 0]
    j_s = ij_s[:, 1]
    z = np.zeros((im.shape[0], im.shape[1]))
    z[i_s, j_s] = 1
    return overlap_boolean_image(im, z, cross_marks=True, color_val=color_val)


def round_range(start, end, len_max):
    if int(end) > int(len_max):
        return list(range(start, len_max)) + list(range(0, end % len_max))
    return range(start, end)


def reflect_border_range(start, end, len_max):
    if start >= 0 and end < len_max:
        return range(start, end)


def get_window_ix(shape, pos, w_shape):
    (w_h, w_l) = map(lambda x: (x - 1) // 2, w_shape)
    return np.ix_(round_range(pos[0] - w_h, pos[0] + w_h + 1, shape[0]),
                  round_range(pos[1] - w_l, pos[1] + w_l + 1, shape[1]))


def np_save(numpy_objects, file):
    with open(file, 'wb') as f:
        for numpy_object in numpy_objects:
            np.save(f, numpy_object)


def np_load(num_objects, file):
    objects = []
    try:
        with open(file, 'rb') as f:
            for i in range(num_objects):
                objects.append(np.load(f, allow_pickle=True))
        return objects[0] if num_objects == 1 else objects
    except Exception as e:
        print(e)
        return None


def add_gaussian_noise(im, sigma=1):
    im = im + np.random.normal(0, sigma, im.shape)
    im[im > 255] = 255
    return im.astype('uint8')


def read_points(filepath):
    pts = []
    with open(filepath) as f:
        for line in f:
            pts.append(tuple([float(ele) for ele in line.split()]))
    return pts


def non_max_suppression(im, window_shape=None):
    im = im.copy()
    if window_shape is None:
        window_shape = ((im.shape[0] // 40) * 2 + 1, (im.shape[1] // 40) * 2 + 1)
    if window_shape[0] % 2 == 0 or window_shape[1] % 2 == 0:
        raise Exception('size should be odd, so the filter has a center pixel')
    w_h = (window_shape[0] - 1) // 2
    w_l = (window_shape[1] - 1) // 2
    op = np.zeros(im.shape)
    for i in range(w_h, im.shape[0] - w_h):
        for j in range(w_l, im.shape[1] - w_l):
            t = im[i, j]
            im[i, j] = np.NINF
            op[i, j] = 1.0 if t > im[i - w_h:i + w_h + 1, j - w_l:j + w_l + 1].max() else 0.0
            im[i, j] = t
    return op


def xy_2_ij(pt_xy):
    return int(pt_xy[1]), int(pt_xy[0])


def ij_2_xy(pt_ij):
    return pt_ij[1], pt_ij[0]


def stitch_images(images, axis=1):
    return np.concatenate(images, axis=axis)


def get_frames_from_video(video_path, f_range=None, resize_to=None):
    """
    Not a great but just a handy util to get a bunch of frames from a video
    """
    video = cv.VideoCapture(video_path)
    frames_total = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    print('total frames: %s' % frames_total)
    frames = []
    if f_range is None:
        frames = [video.read()[1] for i in range(frames_total)]
    else:
        for i in f_range:
            video.set(cv.CAP_PROP_POS_FRAMES, i)
            res, frame = video.read()
            frames.append(frame)
    frames = [cv.cvtColor(frame, cv.COLOR_BGR2RGB) for frame in frames]
    if resize_to is not None:
        return [cv.resize(f, dsize=resize_to, interpolation=cv.INTER_LINEAR) for f in frames]
    return frames


def im_hist(im, bins_per_channel=10, val_range=(0, 255), normed=True):
    if len(im.shape) == 2:
        hist = np.histogram(im, bins=bins_per_channel, range=val_range)
    else:
        hist = np.concatenate(
            [np.histogram(im[:, :, i], bins=bins_per_channel, range=val_range)[0][np.newaxis, :] for i in [0, 1, 2]],
            axis=0)
    return hist.astype(np.float64) / hist.sum() if normed else hist.astype(np.float64)
