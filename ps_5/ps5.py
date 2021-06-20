import numpy as np
from ps_5.helper import lucas_kanade, add_flow_over_im, reduce, gaussian_pyramid, expand, laplacian_pyramid
from ps_hepers.helpers import imread_from_rep, imshow, np_load, imsave, np_save, imfix_scale


def p1_a():
    a = imread_from_rep('TestSeq/Shift0', grey_scale=True)
    b = imread_from_rep('TestSeq/ShiftR2', grey_scale=True)
    c = imread_from_rep('TestSeq/ShiftR5U5', grey_scale=True)
    d = imread_from_rep('TestSeq/ShiftR10', grey_scale=True)
    e = imread_from_rep('TestSeq/ShiftR20', grey_scale=True)
    f = imread_from_rep('TestSeq/ShiftR40', grey_scale=True)

    #a
    flow1 = lucas_kanade(a, b, (15, 15))
    a_flow1 = add_flow_over_im(a, flow1)
    flow2 = lucas_kanade(a, c, (47, 47))
    a_flow2 = add_flow_over_im(a, flow2)
    imsave(a_flow1, 'output/ps5-1-a-1.png')
    imsave(a_flow2, 'output/ps5-1-a-2.png')
    imshow([a_flow1, a_flow2], ['right shift', 'top right shift'], cmap='gray')

    #b
    flow_ims = [add_flow_over_im(a, lucas_kanade(a, x, (47, 47))) for x in [d, e, f]]
    imshow(flow_ims,['r10', 'r20', 'r40'], shape=(3,1))
    [imsave(im, 'output/ps5-1-b-%s.png' % (i+1)) for i, im in zip(range(len(flow_ims)), flow_ims)]


def p1_exp():
    im_names = ['TestSeq/Shift0', 'TestSeq/ShiftR5U5']
    w_start = 5
    w_end = 81
    window_range = np.arange(w_start, w_end, 2)
    exp_name = ','.join([i.replace('/', '-') for i in im_names])
    w_range_as_str = 'w%s-%s' % (w_start, w_end)
    a = imread_from_rep(im_names[0], grey_scale=True)
    b = imread_from_rep(im_names[1], grey_scale=True)
    flow_window_list = np_load(len(window_range), 'objects/flow_%s_%s.npy' % (w_range_as_str, exp_name))
    if flow_window_list is None or len(flow_window_list) != len(window_range):
        flow_window_list = []
        for w in window_range:
            print('processing images for motion flow with window size %s' % w)
            flow1 = add_flow_over_im(a, lucas_kanade(a, b, (w, w)))
            flow_window_list.append((w, flow1))
            imsave(flow1, 'observations/flow_%s_w_size_%s.png' % (exp_name, w))
    np_save(np.asarray(flow_window_list, dtype=object), 'objects/flow_%s_%s.npy' % (w_range_as_str, exp_name))

    def update_exp(x, axs, sliders, buttons):
        i = np.abs(window_range - sliders[0].val).argmin()
        return [0], [flow_window_list[i][1]]

    slider_attr = [{'label': 'Window Size', 'valmin': w_start, 'valmax': w_end, 'valstep': 2}]
    imshow(flow_window_list[0][1], [' Detected Optical Flow'], slider_attr=slider_attr, slider_callback=[update_exp])


def p2():
    im = imread_from_rep('DataSeq1/yos_img_01', extension='.jpg')
    g_py = gaussian_pyramid(im, 4)
    imshow(g_py)
    [imsave(g_py[i], 'output/ps5-2-a-%s.png' % (i+1)) for i in range(len(g_py))]
    l_py = laplacian_pyramid(im, 4)
    imshow([imfix_scale(i) for i in l_py])
    [imsave(g_py[i], 'output/ps5-2-b-%s.png' % (i+1)) for i in range(len(g_py))]


if __name__ == '__main__':
    # p1_a()
    # p1_exp()
    p2()
