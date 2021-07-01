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
    track_patch(frames, patch_box, frames_to_save=[28, 88, 144], output_format='output/ps6-1-a-%s.png', sigma_mse=10,
                dynamics_std=5.0, num_particles=100, prior='gaussian', similarity_method='mse')


def p1_e():
    frames = get_frames_from_video('input/noisy_debate.avi', resize_to=(512, 288))
    patch_box = (read_box('input/noisy_debate.txt') * 2 / 5).astype(np.int32)
    track_patch(frames, patch_box, frames_to_save=[14, 32, 46], output_format='output/ps6-1-e-%s.png', sigma_mse=10.0,
                dynamics_std=5.0, num_particles=50, state_est='top_quarter')


def p1_exp():
    print('tracking Obama')
    frames = get_frames_from_video('input/pres_debate.avi', resize_to=(512, 288))
    patch_box = (read_box('input/pres_debate_exp.txt') * 2 / 5).astype(np.int32)
    track_patch(frames, patch_box, frames_to_save=[28, 88, 144], output_format='output/ps6-1-exp-%s.png',
                sigma_mse=10.0, dynamics_std=10.0, num_particles=200, state_est='top_70_percentile')


def p1_exp_l():
    print('tracking lally')
    frames = get_frames_from_video('input/lally_jumping.mp4', resize_to=(320, 176))
    patch_box = (read_box('input/lally_jumping.txt') / 2.0).astype(np.int32)
    track_patch(frames, patch_box, frames_to_save=[28, 88, 144], output_format='output/ps6-1-exp_l-%s.png',
                sigma_mse=10.0, dynamics_std=5.0, num_particles=300, state_est='top_70_percentile', prior='gaussian',
                t_update_factor=0.01)


def p2_a():
    frames = get_frames_from_video('input/pres_debate.avi')
    patch_box = (read_box('input/romneys_hand.txt')).astype(np.int32)
    track_patch(frames, patch_box, frames_to_save=[15, 50, 140], output_format='output/ps6-2-a-%s.png', sigma_mse=10,
                dynamics_std=20.0, num_particles=300, prior='gaussian', t_update_factor=0.05)


def p2_b():
    frames = get_frames_from_video('input/noisy_debate.avi')
    patch_box = (read_box('input/romneys_hand.txt')).astype(np.int32)
    # below params just does the job to track his hand until mid of the video. A noise tolerant template update logic
    # might help the tracker to maintain the target
    track_patch(frames, patch_box, frames_to_save=[15, 50, 140], output_format='output/ps6-2-b-%s.png', sigma_mse=2.5,
                dynamics_std=30.0, num_particles=500, prior='gaussian', t_update_factor=0.3, state_est='top_quarter')


def p3_a():
    frames = get_frames_from_video('input/pres_debate.avi', resize_to=(512, 288))
    patch_box = (read_box('input/pres_debate.txt') * 2 / 5).astype(np.int32)
    track_patch(frames, patch_box, frames_to_save=[28, 88, 144], output_format='output/ps6-3-a-%s.png', sigma_mse=0.05,
                dynamics_std=5.0, num_particles=100, prior='gaussian', similarity_method='histcmp', t_update_factor=0)


def p3_b():
    """
    Histogram comparison is very sensitive to background colors. So, when Romney's hand moves down the change in the
    background from blue to red creates confusion for the particles and keeps them on the blue background which was in
    the template. A fix for this is to adjust the patch size to not include any background
    """
    frames = get_frames_from_video('input/pres_debate.avi')
    patch_box = (read_box('input/romneys_hand_no_background.txt')).astype(np.int32)
    track_patch(frames, patch_box, frames_to_save=[15, 50, 140], output_format='output/ps6-3-b-%s.png', sigma_mse=0.05,
                dynamics_std=20.0, num_particles=200, prior='gaussian', t_update_factor=0.0,
                similarity_method='histcmp')


def p4():
    frames = get_frames_from_video('input/pedestrians.avi')
    patch_box = (read_box('input/pedestrians_new.txt')).astype(np.int32)
    track_patch(frames, patch_box, track_velocities=True, frames_to_save=[40, 100, 240],
                output_format='output/ps6-4-a-%s.png', sigma_mse=10, dynamics_std=5.0, num_particles=1500,
                prior='uniform', similarity_method='mseblur')


if __name__ == '__main__':
    # p1_a()
    # p1_e()
    # p1_exp()
    # p1_exp_l()
    # p2_a()
    # p2_b()
    # p3_a()
    # p3_b()
    p4()
