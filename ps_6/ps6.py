import numpy as np

from ps_6.helper import track_patch
from ps_hepers.helpers import get_frames_from_video, imshow

"""
Problem Set - 6
Problems: https://docs.google.com/document/d/1ZGdXBjLgr9U-6wdIgcEyRmDHckJFlyhnSWXK8OsduS4/pub?embedded=true
"""


def read_box(path):
    with open(path) as f:
        x, y = f.readline().split()
        w, h = f.readline().split()
    return np.asarray([[x, y], [w, h]]).astype(np.float32)


def p1_a():
    frames = get_frames_from_video('input/pres_debate.avi', resize_to=(512, 288))
    patch_box = (read_box('input/pres_debate.txt') * 2 / 5).astype(np.int32)
    track_patch(frames, patch_box, frames_to_save=[28, 88, 144], output_format='output/ps6-1-a-%s.png', sigma_mse=10,
                dynamics_std=5.0, num_particles=30, prior='gaussian', similarity_method='mse')
    """
    # b
    Picking up the correct window size for tracking can be hard. 
    With bigger windows the templates hold enough information about the target and thus will have less ambiguity at 
    localization. But this comes with a downside of noise from background. With bigger windows, if the background 
    changes, depending on the distance method, tracking becomes difficult. 
    
    With smaller windows we might be able to avoid disturbances in background changes but we might not have entire 
    information in the widow to track the target. For example, if the window only has a patch of skin but the target is
    the head, the tracker will be lost among the other identical patches of skin on the head. 
    
    # c
    σMSE defines the sensitivity of score to the distance method (mse). σMSE should be set to a value that corresponds 
    to relevance of target distance detection.
    
    With higher σMSE's, scoring does not happen correctly and particle weights are distributed evenly, making the 
    irrelevant particles to also get re-sampled, resulting in poor particle tracking performance. 
    
    With lowers σMSE, the scores tend to be smaller and variations between them lesser. Resulting in poor particle 
    tracking performance.
    
    # d
    Roughly only 30 particles, at least, were required for tracking Romney's face successfully till the end.
    """


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
    track_patch(frames, patch_box, frames_to_save=[15, 50, 140], output_format='output/ps6-2-b-%s.png', sigma_mse=2.5,
                dynamics_std=30.0, num_particles=500, prior='gaussian', t_update_factor=0.3, state_est='top_quarter')
    """
    above params just does the job to track his hand until mid of the video. A noise tolerant template update logic
    might help the tracker to maintain the target
    """


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
    """
    Roughly 1000 particles were required to track the blond-woman. In this model, I had used particle velocities and 
    window sizes in the particle states. 
    The dynamics included position to be updated according to the velocity.
    With multiple dimensions in the particle state (i.e, 5 - 2 for position, 2 for velocity, 1 for window size), more 
    particles were required in maintaining/tracking or scanning the target.  
    """


if __name__ == '__main__':
    p1_a()
    # p1_e()
    # p1_exp()
    # p1_exp_l()
    # p2_a()
    # p2_b()
    # p3_a()
    # p3_b()
    # p4()
