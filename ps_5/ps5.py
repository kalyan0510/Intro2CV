import numpy as np
from ps_5.helper import lucas_kanade, add_flow_over_im, reduce, gaussian_pyramid, expand, laplacian_pyramid, remap, \
    hierarchical_lk, hierarchical_laplacian_lk
from ps_hepers.helpers import imread_from_rep, imshow, np_load, imsave, np_save, imfix_scale, stitch_images, \
    get_frames_from_video


def p1_a():
    a = imread_from_rep('TestSeq/Shift0', grey_scale=True)
    b = imread_from_rep('TestSeq/ShiftR2', grey_scale=True)
    c = imread_from_rep('TestSeq/ShiftR5U5', grey_scale=True)
    d = imread_from_rep('TestSeq/ShiftR10', grey_scale=True)
    e = imread_from_rep('TestSeq/ShiftR20', grey_scale=True)
    f = imread_from_rep('TestSeq/ShiftR40', grey_scale=True)

    # a
    flow1 = lucas_kanade(a, b, (15, 15))
    a_flow1 = add_flow_over_im(a, flow1)
    flow2 = lucas_kanade(a, c, (47, 47))
    a_flow2 = add_flow_over_im(a, flow2)
    imsave(a_flow1, 'output/ps5-1-a-1.png')
    imsave(a_flow2, 'output/ps5-1-a-2.png')
    imshow([a_flow1, a_flow2], ['right shift', 'top right shift'], cmap='gray')

    # b
    flow_ims = [add_flow_over_im(a, lucas_kanade(a, x, (47, 47))) for x in [d, e, f]]
    imshow(flow_ims, ['r10', 'r20', 'r40'], shape=(3, 1))
    [imsave(im, 'output/ps5-1-b-%s.png' % (i + 1)) for i, im in zip(range(len(flow_ims)), flow_ims)]


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
    [imsave(g_py[i], 'output/ps5-2-a-%s.png' % (i + 1)) for i in range(len(g_py))]
    l_py = laplacian_pyramid(im, 4)
    imshow([imfix_scale(i) for i in l_py])
    [imsave(g_py[i], 'output/ps5-2-b-%s.png' % (i + 1)) for i in range(len(g_py))]


def p3_exp():
    dataset_params = [('DataSeq2/%s', '.png', [0, 1, 2]), ('DataSeq1/yos_img_0%s', '.jpg', [1, 2, 3])]
    for prm in dataset_params:
        imgs = [imread_from_rep(prm[0] % i, extension=prm[1]) for i in prm[2]]
        gpy_s = [gaussian_pyramid(imgs[i], 4) for i in range(len(imgs))]
        [imshow([add_flow_over_im(a, lucas_kanade(a, b, (21, 21))) for a, b in zip(gpy_s[j], gpy_s[j + 1])],
                ['level %s' % i for i in range(len(gpy_s[j]))]) for j in range(len(imgs) - 1)]


def p3():
    dataset_params = [('DataSeq1/yos_img_0%s', '.jpg', [1, 2, 3], 1, True),
                      ('DataSeq2/%s', '.png', [0, 1, 2], 2, False)]
    for (ds_path, ext, frame_seq, fit_level, upscale), d in zip(dataset_params, range(len(dataset_params))):
        imgs = [imread_from_rep(ds_path % i, extension=ext) for i in frame_seq]
        gpy_s = [gaussian_pyramid(img, 4, up_scaled=upscale) for img in imgs]
        flows = [lucas_kanade(gpy_s[i][fit_level], gpy_s[i + 1][fit_level], (27, 27)) for i in range(len(imgs) - 1)]
        arrows = [add_flow_over_im(gpy_s[i][fit_level], flows[i]) for i in range(len(imgs) - 1)]
        remaps = [remap(gpy_s[i + 1][fit_level], -flows[i]) for i in range(len(imgs) - 1)]
        diffs = [imfix_scale(gpy_s[i][fit_level].astype(np.float32) - remaps[i].astype(np.float32)) for i in
                 range(len(imgs) - 1)]
        imshow(arrows, ['frame %s' % i for i in range(len(imgs))], sup_title='flow arrows')
        imshow(remaps, ['frame %s' % i for i in range(len(imgs))], sup_title='remaps')
        imsave(stitch_images(arrows), 'output/ps5-3-a-%s.png' % (2 * d + 1))
        imsave(stitch_images(imfix_scale(diffs)), 'output/ps5-3-a-%s.png' % (2 * d + 2))
        for i in range(len(remaps)):
            imsave(gpy_s[i][fit_level], 'observations/3-%s%s.png' % (d, i))
            imsave(remaps[i], 'observations/3-%s%sr.png' % (d, i))


def p4_a():
    a = imread_from_rep('TestSeq/Shift0')
    b = imread_from_rep('TestSeq/ShiftR2')
    c = imread_from_rep('TestSeq/ShiftR5U5')
    d = imread_from_rep('TestSeq/ShiftR10')
    e = imread_from_rep('TestSeq/ShiftR20')
    f = imread_from_rep('TestSeq/ShiftR40')
    targets = [b, c, d, e, f]
    flows = [hierarchical_lk(a, x) for x in targets]
    flow_ims = [add_flow_over_im(a, flow) for flow in flows]
    imshow([stitch_images([np.abs(flow[:, :, 0]), np.abs(flow[:, :, 1])], axis=0) for flow in flows],
           ['r2', 'r5u5', 'r10', 'r20', 'r40'], cmap='gray', sup_title='displacement images')
    imshow(flow_ims, ['r2', 'r5u5', 'r10', 'r20', 'r40'], sup_title='flow arrows')
    remaps = [remap(a, flow) for flow in flows]
    imshow([stitch_images([remap_i, target], axis=0) for (remap_i, target) in zip(remaps, targets[:-1])],
           ['r2', 'r5u5', 'r10', 'r20', 'r40'], sup_title='remap vs actual')
    imshow([imfix_scale(remap_i.astype(np.float32) - target.astype(np.float32)) for (remap_i, target) in
            zip(remaps, targets[:-1])], sup_title='differences')
    imsave(stitch_images(flow_ims), 'output/ps5-4-a-1.png')
    imsave(stitch_images([imfix_scale(remap_i.astype(np.float32) - target.astype(np.float32)) for (remap_i, target) in
                          zip(remaps, targets[:-1])]), 'output/ps5-4-a-2.png')


def p4_bc():
    dataset_params = [
        ('DataSeq1/yos_img_0%s', '.jpg', [1, 2, 3]),
        ('DataSeq2/%s', '.png', [0, 1, 2]),
    ]
    for (ds_path, ext, frame_seq), d in zip(dataset_params, range(len(dataset_params))):
        imgs = [imread_from_rep(ds_path % i, extension=ext) for i in frame_seq]
        f_range = range(len(imgs))
        flows = [hierarchical_lk(imgs[i], imgs[i + 1]) for i in f_range[:-1]]
        arrows = [add_flow_over_im(imgs[i], flows[i]) for i in f_range[:-1]]
        remaps = [remap(imgs[i], flows[i]) for i in f_range[:-1]]
        diffs = [imgs[i + 1].astype(np.float32) - remaps[i].astype(np.float32) for i in
                 f_range[:-1]]
        imshow([stitch_images(imfix_scale([np.abs(flow[:, :, 0]), np.abs(flow[:, :, 1])]), axis=0) for flow in flows],
               range(len(flows)), sup_title='displacement images')
        imshow(arrows, ['frame %s' % i for i in range(len(imgs))], sup_title='flow arrows')
        imshow(remaps, ['frame %s' % i for i in range(len(imgs))], sup_title='remaps')
        imshow([imfix_scale(diff) for diff in diffs], ['frame %s' % i for i in range(len(imgs))],
               sup_title='differences')
        imsave(stitch_images(arrows), 'output/ps5-4-%s-1.png' % chr(ord('b') + d))
        imsave(stitch_images(imfix_scale(diffs)), 'output/ps5-4-%s-2.png' % chr(ord('b') + d))
        for i in f_range[:-1]:
            imsave(imgs[i + 1], 'observations/4-%s%s.png' % (d, i))
            imsave(remaps[i], 'observations/4-%s%sr.png' % (d, i))


def on_video():
    imgs = get_frames_from_video('input/car/car.mp4', f_range=range(20, 40))
    f_range = range(len(imgs))
    [imsave(arrow, 'output/car/img-%s.png' % i) for arrow, i in zip(imgs, f_range)]
    flows = [hierarchical_lk(imgs[i], imgs[i + 1]) for i in f_range[:-1]]
    [imsave(imfix_scale(np.concatenate([flow, 0 * flow[:, :, 0][:, :, np.newaxis]], axis=2)),
            'output/car/flow-%s.png' % i) for flow, i in zip(flows, f_range[:-1])]
    arrows = [add_flow_over_im(imgs[i], flows[i] / 200.0, gap=15) for i in f_range[:-1]]
    [imsave(arrow, 'output/car/%s.png' % i) for arrow, i in zip(arrows, f_range[:-1])]


def p5():
    dataset_params = [('Juggle/%s', '.png', [0, 1, 2])]
    for (ds_path, ext, frame_seq), d in zip(dataset_params, range(len(dataset_params))):
        imgs = [imread_from_rep(ds_path % i, extension=ext) for i in frame_seq]
        f_range = range(len(imgs))
        flows = [hierarchical_laplacian_lk(imgs[i], imgs[i + 1]) for i in f_range[:-1]]
        arrows = [add_flow_over_im(imgs[i], flows[i]) for i in f_range[:-1]]
        remaps = [remap(imgs[i], flows[i]) for i in f_range[:-1]]
        diffs = [imgs[i + 1].astype(np.float32) - remaps[i].astype(np.float32) for i in
                 f_range[:-1]]
        imshow([stitch_images(([(flow[:, :, 0]), (flow[:, :, 1])]), axis=0) for flow in flows],
               list(range(len(flows))), sup_title='displacement images')
        imshow(arrows, ['frame %s' % i for i in range(len(imgs))], sup_title='flow arrows')
        imshow(remaps, ['frame %s' % i for i in range(len(imgs))], sup_title='remaps')
        imshow([imfix_scale(diff) for diff in diffs], ['frame %s' % i for i in range(len(imgs))],
               sup_title='differences')
        imsave(stitch_images(arrows), 'output/ps5-5-%s-1.png' % chr(ord('a') + d))
        imsave(stitch_images(imfix_scale(diffs)), 'output/ps5-5-%s-2.png' % chr(ord('a') + d))


if __name__ == '__main__':
    print('Running p1_a')
    p1_a()
    print('Running p1_exp')
    p1_exp()
    print('Running p2')
    p2()
    print('Running p3_exp')
    p3_exp()
    print('Running p3')
    p3()
    print('Running p4_a')
    p4_a()
    print('Running p4_bc')
    p4_bc()
    print('Running on_video')
    on_video()
    print('Running p5')
    p5()
