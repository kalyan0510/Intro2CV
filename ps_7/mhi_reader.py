import numpy as np
import cv2 as cv


class MHIReader:
    def __init__(self, motion_extractor, init_frame=None, frame_shape=None, tau=25, blur_size=(5, 5),
                 frame_diff_threshold=20):
        self.tau = tau
        self.blur_size = blur_size
        self.frame_diff_threshold = frame_diff_threshold
        self.motion_extractor = motion_extractor
        if init_frame is not None:
            self.frame_shape = init_frame.shape
            self.prev = self.pre_process(init_frame)
        else:
            self.frame_shape = frame_shape
            self.prev = None
        self.mhi = np.zeros(self.frame_shape[:2], dtype=np.uint8)

    def pre_process(self, frame):
        return cv.GaussianBlur(frame, self.blur_size, 0)

    def update(self, im_t):
        current = self.pre_process(im_t)
        if self.prev is not None:
            b_motion = self.motion_extractor(self.prev, current)
            self.mhi[self.mhi == 0] = 1
            self.mhi -= 1
            self.mhi[b_motion] = self.tau
        self.prev = current

    def get_mhi(self, normed=False):
        if (not normed) or self.mhi.max() == 0:
            return self.mhi.astype(np.float32)
        else:
            return self.mhi.astype(np.float32) / self.mhi.max()
