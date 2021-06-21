import numpy as np
import cv2 as cv
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt


def get_pixel_derivatives(im_t0, im_t1):
    t0, t1 = im_t0.astype(np.float32), im_t1.astype(np.float32)
    ix, iy = np.zeros(t0.shape, dtype=np.float32), np.zeros(t0.shape, dtype=np.float32)
    ix[1:-1, 1:-1, ...] = t0[1:-1, 2:, ...] - t0[1:-1, :-2, ...]
    iy[1:-1, 1:-1, ...] = t0[2:, 1:-1, ...] - t0[:-2, 1:-1, ...]
    return ix, iy, (t0 - t1).astype(np.float32)


def lucas_kanade(im0, im1, k_shape=(5, 5)):
    ix, iy, it = get_pixel_derivatives(im0, im1)
    grad_prods = [ix * ix, ix * iy, iy * iy, ix * it, iy * it]
    grad_prods = [np.sum(grad_prod, axis=2) for grad_prod in grad_prods] if len(im0.shape) == 3 else grad_prods
    wgps = [cv.GaussianBlur(grad_prod, k_shape, 0) for grad_prod in grad_prods]
    wgps = np.concatenate([wgp[:, :, np.newaxis] for wgp in wgps], axis=2)
    m_flow = np.zeros(im0.shape[:2] + (2,), dtype=np.float32)
    (w_h, w_l) = map(lambda x: (x - 1) // 2, k_shape)
    for i in range(w_h, im0.shape[0] - w_h):
        for j in range(w_l, im0.shape[1] - w_l):
            a = [[wgps[i, j, 0], wgps[i, j, 1]], [wgps[i, j, 1], wgps[i, j, 2]]]
            b = [[-wgps[i, j, 3]], [-wgps[i, j, 4]]]
            uv, _, rank, _ = np.linalg.lstsq(a, b, rcond=0.1)
            m_flow[i, j, 0] = uv[0] if rank == 2 else 0
            m_flow[i, j, 1] = uv[1] if rank == 2 else 0
    return -m_flow


def get_flow_arrows(flow, gap=None, show_arrows_on_plt=False):
    gap = [gap, flow.shape[0] // 30][gap is None]
    x = np.arange(0, flow.shape[1], 1)
    y = np.arange(0, flow.shape[0], 1)
    x, y = np.meshgrid(x, y)
    fig = plt.figure()
    axs = fig.add_axes([0, 0, 1, 1], frameon=False)
    fig.patch.set_alpha(0)
    axs.patch.set_alpha(0)
    canvas = FigureCanvas(fig)
    plt.quiver(x[::gap, ::gap], y[::-gap, ::-gap], flow[::gap, ::gap, 0], -flow[::gap, ::gap, 1], color='red')
    axs.axis('off')
    axs.margins(0)
    canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    if not show_arrows_on_plt:
        plt.close(fig)
    return cv.resize(image_from_plot, dsize=flow.shape[1::-1])


def add_flow_over_im(im, flow, show_arrows_on_plt=False):
    im = im.copy()
    if len(im.shape) == 2:
        im = cv.cvtColor(im, cv.COLOR_GRAY2RGB)
    arrows_arbg = get_flow_arrows(flow, show_arrows_on_plt=show_arrows_on_plt)
    alpha = arrows_arbg[:, :, 0]
    for i in range(3):
        im[:, :, i] = (alpha / 255.0) * arrows_arbg[:, :, 1 + i] + ((255.0 - alpha) / 255.0) * im[:, :, i]
    return im


def reduce(im):
    im_blur = cv.GaussianBlur(im, (5, 5), 0)
    return im_blur[::2, ::2, ...]


def expand(im, dst_size):
    return cv.GaussianBlur(cv.resize(im, dst_size[1::-1], interpolation=cv.INTER_NEAREST), (5, 5), 0)


def gaussian_pyramid(im, max_l=np.PINF, up_scaled=False):
    levels = min(int(np.log2(min(im.shape[:2])) + 1), max_l)
    g_pyr = [im]
    subscale = lambda i: expand(reduce(i), i.shape) if up_scaled else reduce(i)
    [g_pyr.append(subscale(g_pyr[-1])) for i in range(levels - 1)]
    return g_pyr


def laplacian_pyramid(im, max_l=np.PINF):
    levels = min(int(np.log2(min(im.shape[:2])) + 1), max_l)
    l_pyr = [im]
    for i in range(levels - 1):
        tmp = reduce(l_pyr[-1])
        l_pyr[-1] = l_pyr[-1].astype(np.int32) - expand(tmp, l_pyr[-1].shape).astype(np.int32)
        l_pyr.append(tmp)
    return l_pyr


def remap(a, flow):
    x = flow[:, :, 1] + np.arange(a.shape[1]).astype(np.float32)
    y = flow[:, :, 0] + np.arange(a.shape[0])[:, np.newaxis].astype(np.float32)
    return cv.remap(a, x, y, interpolation=cv.INTER_LINEAR)
