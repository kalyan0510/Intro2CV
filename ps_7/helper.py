import numpy as np
import cv2 as cv

from ps_7.compute_moments import compute_mhi_feature_vec
from ps_7.mhi_reader import MHIReader
from ps_hepers.helpers import get_frames_from_video


def get_video_path(action, person, trial):
    return 'input/PS7A%sP%sT%s.avi' % (action, person, trial)


def to_gray_float(im):
    return (cv.cvtColor(im, cv.COLOR_RGB2GRAY) if len(im.shape) > 2 else im).astype(np.float32)


def frame_diff(im_t0, im_t1, threshold=30):
    b = np.abs(to_gray_float(im_t1) - to_gray_float(im_t0)) > threshold
    return cv.morphologyEx(b.astype(np.uint8), cv.MORPH_OPEN,
                           cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))).astype(np.bool)


def get_action_observations_with_labels(persons=None, tau=40, start_f=30, calc_feature_with_mei=True):
    if persons is None:
        persons = [1, 2, 3]
    mhi_feat_vectors = []
    action_labels = []
    for action in [1, 2, 3]:
        for person in persons:
            for trial in [1, 2, 3]:
                print('calculating mhi feature vectors for a%s p%s t%s' % (action, person, trial))
                frames = get_frames_from_video(get_video_path(action, person, trial))
                mhi_reader = MHIReader(motion_extractor=frame_diff, init_frame=frames[0], tau=tau)
                for i in range(1, len(frames)):
                    mhi_reader.update(frames[i])
                    if i >= start_f:
                        mhi_feat_vectors.append(
                            compute_mhi_feature_vec(mhi_reader.get_mhi(normed=True), with_mei=calc_feature_with_mei))
                        action_labels.append(action)
    return np.asarray(mhi_feat_vectors, dtype=np.float32), np.asarray(action_labels, dtype=np.float32)


def train_and_test(x_train, y_train, x_test, y_test):
    cf_mat = np.zeros((3, 3))
    knn = cv.ml.KNearest_create()
    knn.train(x_train, cv.ml.ROW_SAMPLE, y_train)
    ret, results, neighbours, dist = knn.findNearest(x_test, 3)
    for pred, actual in zip(results, y_test):
        cf_mat[int(pred) - 1, int(actual) - 1] += 1
    return cf_mat / cf_mat.sum()
