import cv2 as cv
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def imread(filename, channels=None, grey_scale=None):
    """
    Simple imread that uses cv.imread
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
    cv.imwrite(path_name, img if len(img.shape) == 2 else cv.cvtColor(img, cv.COLOR_RGB2BGR))


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


def imshow(im, im_title=None, shape=None, interpolation='bilinear', sup_title='Figure(s)', slider_attr=None,
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
    :param sup_title: main title of the entire figure
    :param slider_attr: a list of set of slider attributes. one set for each slider that defines valmax, valmin, valint
    :param slider_callback: a list of callback methods each accepting the params {event, axs, sliders, buttons} which
    define the state of figure/canvas
    :param button_attr: a list of set of button attributes. one for each slider that defines {'label'}
    :param button_callback: a list of callback methods each accepting the params {event, axs, sliders, buttons} which
    define the state of figure/canvas
    :return: None
    """
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


def overlap_boolean_image(im, boolean_im, val=255, color_val=(255, 0, 0)):
    """
    The boolean array is projected into the input image (grey scale or color)
    :param im: image (grey scale or color)
    :param boolean_im: boolean image
    :param val: greyscale value used for projection onto image
    :param color_val: color value used for projection onto image
    :return: overlapped image
    """
    im = np.copy(im)
    if len(im.shape) == 2:
        im[boolean_im > 0] = val
    elif len(im.shape) == 3:
        im[boolean_im > 0, 0:3] = color_val
    return im