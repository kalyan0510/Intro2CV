import numpy as np

from ps_6.helper import track_patch
from ps_hepers.helpers import get_frames_from_video, imshow


def read_box(path):
    with open(path) as f:
        x, y = f.readline().split()
        w, h = f.readline().split()
    return np.asarray([[x, y], [w, h]]).astype(np.float32)


def p1_a():
    frames = get_frames_from_video('input/pres_debate.avi', resize_to=(512, 288))
    patch_box = (read_box('input/pres_debate.txt') * 2 / 5).astype(np.int32)
    track_patch(frames, patch_box, frames_to_save=[28, 88, 144], output_format='output/ps6-1-a-%s.png')


def p1_e():
    frames = get_frames_from_video('input/noisy_debate.avi', resize_to=(512, 288))
    patch_box = (read_box('input/noisy_debate.txt') * 2 / 5).astype(np.int32)
    track_patch(frames, patch_box, frames_to_save=[28, 88, 144], output_format='output/ps6-1-e-%s.png')


if __name__ == '__main__':
    print('Running p1_a')
    p1_a()
    # print('Running p1_e')
    # p1_e()
