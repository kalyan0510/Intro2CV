import cv2 as cv
import numpy as np
from ps_7.helper import frame_diff, get_action_observations_with_labels, get_video_path, train_and_test
from ps_7.mhi_reader import MHIReader
from ps_hepers.helpers import get_frames_from_video, imshow, imsave, imfix_scale


def p1_a_exp():
    frames = get_frames_from_video(get_video_path(1, 1, 1), f_range=[9, 10])
    t0, t1 = frames[0], frames[1]
    t0, t1 = [cv.GaussianBlur(frame, (5, 5), 1) for frame in [t0, t1]]
    imshow(t0)
    imshow(t1)

    def update_exp(x, axs, sliders, buttons):
        return [0], [frame_diff(t0, t1, sliders[0].val)]

    slider_attr = [{'label': 'threshold', 'valmin': 0, 'valmax': 255, 'valstep': 1}]
    imshow(t0, 'Frame diff', slider_attr=slider_attr, slider_callback=[update_exp], cmap='gray')


def p1_a():
    frames = get_frames_from_video(get_video_path(1, 1, 1))
    frames = [cv.GaussianBlur(frame, (5, 5), 1) for frame in frames]
    f_diff_to_save = [10, 20, 30]
    for t, i in zip(f_diff_to_save, range(len(f_diff_to_save))):
        diff = frame_diff(frames[t - 1], frames[t], threshold=10)
        imshow(diff, cmap='gray')
        imsave(imfix_scale(diff.astype(np.float32)), 'output/ps7-1-a-%s.png' % (i + 1))


def p1_b():
    data_config = [
        {'filepath': get_video_path(1, 2, 1), 'tau': 25, 'end': 45},
        {'filepath': get_video_path(2, 2, 1), 'tau': 35, 'end': 40},
        {'filepath': get_video_path(3, 2, 1), 'tau': 45, 'end': 50}
    ]
    for data, d_i in zip(data_config, range(len(data_config))):
        frames = get_frames_from_video(data['filepath'])
        mhi_reader = MHIReader(motion_extractor=frame_diff, init_frame=frames[0], tau=data['tau'])
        for i in range(1, len(frames)):
            mhi_reader.update(frames[i])
            if cv.waitKey(40) & 0xFF == ord('q'):
                break
            cv.imshow("MHI", (mhi_reader.get_mhi() * (255 // mhi_reader.tau)).astype(np.uint8))
            if i == data['end'] - 1:
                imsave(imfix_scale(mhi_reader.get_mhi()), 'output/ps7-1-b-%s.png' % (d_i + 1))


def observe_mhi():
    """
    Saves an MHi for each frame in all the trials.
    For the provided 27 trials, this provides 2238 MHIs. The observation is that with tau=40, one can start recognizing
    the actions from frame ~30
    """
    for action in [1, 2, 3]:
        for person in [1, 2, 3]:
            for trial in [1, 2, 3]:
                print('saving mhis for a%s p%s t%s' % (action, person, trial))
                frames = get_frames_from_video(get_video_path(action, person, trial))
                mhi_reader = MHIReader(motion_extractor=frame_diff, init_frame=frames[0], tau=40)
                for i in range(1, len(frames)):
                    mhi_reader.update(frames[i])
                    imsave(imfix_scale(mhi_reader.get_mhi()),
                           'observations/a%sp%st%sf%s.png' % (action, person, trial, i))


def p2_a():
    mhi_feat_data, action_labels = get_action_observations_with_labels()
    cf_mat = train_and_test(mhi_feat_data, action_labels, mhi_feat_data, action_labels)
    print(cf_mat)


def compute_cf_mat_with_test_person(person):
    mhi_feat_train_data, action_labels_train = get_action_observations_with_labels(list({1, 2, 3} - {person}))
    mhi_feat_test_data, action_labels_test = get_action_observations_with_labels([person])
    cf_mat = train_and_test(mhi_feat_train_data, action_labels_train, mhi_feat_test_data, action_labels_test)
    return cf_mat


def p2_b():
    cf_mats = [compute_cf_mat_with_test_person(person) for person in [1, 2, 3]]
    print(cf_mats)
    print(np.sum(cf_mats, axis=0) / 3.0)


if __name__ == '__main__':
    # p1_a_exp()
    # p1_a()
    # p1_b()
    # observe_mhi()
    # p2_a()
    p2_b()
