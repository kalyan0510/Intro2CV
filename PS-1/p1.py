import cv2 as cv
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
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY) if grey_scale else img[:, :, channels]


def imread_from_rep(image_name, **kwargs):
    channels = kwargs.get('channels', None)
    grey_scale = kwargs.get('grey_scale', None)
    in_cur_dir = Path("input/" + image_name + '.png')
    if in_cur_dir.is_file():
        path = in_cur_dir
    else:
        path = Path("../images/" + image_name + '.png')
    return imread(str(path.resolve()), channels, grey_scale)


def imsave(img, file_name):
    cv.imwrite(file_name, img)


def imfix_scale(img):
    return (img - np.min(img)) / (np.max(img) - np.min(img)) * 256


def imshow(im, im_title='Image', shape=None, sup_title='Figure(s)', slider_attr=None, slider_callback=None,
           button_attr=None, button_callback=None):
    if not type(im) == list:
        im = [im]
    # determine squarish shape to arrange all images
    shape = [shape, (int(np.sqrt(len(im))), int(np.ceil(len(im) / int(np.sqrt(len(im))))))][shape is None]
    fig, axs = plt.subplots(shape[0], shape[1])
    axs = axs.flatten()
    # make ticks and axes(axis) disappear
    for ax in axs:
        ax.set_axis_off()
    # plot each image in its axes
    for i in range(len(im)):
        axs[i].imshow(im[i], interpolation='bilinear')
        axs[i].set_title(im_title if not type(im_title) == list else im_title[i], fontweight='bold')
        axs[i].set_axis_off()
    # create widgets to interact with images
    num_sliders = 0 if slider_attr is None else len(slider_attr)
    num_buttons = 0 if button_attr is None else len(button_attr)

    widget_width = 0.05
    fig.subplots_adjust(bottom=widget_width * (num_buttons + num_sliders + 1))

    def create_slider(i):
        slider_ax = fig.add_axes([0.2, widget_width * (num_sliders - i), 0.65, 0.03], facecolor='grey')
        slider = Slider(slider_ax, slider_attr[i]['label'], slider_attr[i]['valmin'], slider_attr[i]['valmax'],
                        slider_attr[i]['valint'], color='#6baeff')
        slider.on_changed(lambda x: update_images(x, slider_callback[i]))
        return slider

    def create_button(i):
        button_ax = fig.add_axes([0.75, widget_width * (num_sliders) + widget_width * (num_buttons - i), 0.1, 0.03],
                                 facecolor='grey')
        button = Button(button_ax, button_attr[i]['label'], color='0.99', hovercolor='0.575')
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
                    axs[u_i].imshow(u_im, interpolation='bilinear')

    # set main title
    fig.canvas.manager.set_window_title(sup_title)
    plt.suptitle(sup_title)
    # bigger viewing area
    fig.set_size_inches(2*fig.get_size_inches())
    plt.show()


def p0():
    im = imread_from_rep('ps1-input0')
    im2 = imread_from_rep('lena')

    slider_attr = [{'label': 'A', 'valmin': 0, 'valmax': 10, 'valint': 5},
                   {'label': 'B', 'valmin': 0, 'valmax': 20, 'valint': 15}]
    button_attr = [{'label': 'resetA'}, {'label': 'resetB'}]

    def update_im2lena(x, axs, sliders, buttons):
        return [int(x)], [im2]

    def update_im2squares(x, axs, sliders, buttons):
        return [int(x / 2)], [im]

    def reset_a(x, axs, sliders, buttons):
        sliders[0].reset()

    def reset_b(x, axs, sliders, buttons):
        sliders[1].reset()

    imshow([im] * 10, im_title=['im - %s' % i for i in range(23)], sup_title='Testing fun', slider_attr=slider_attr,
           slider_callback=[update_im2lena, update_im2squares], button_attr=button_attr, button_callback=[reset_a, reset_b])


p0()
