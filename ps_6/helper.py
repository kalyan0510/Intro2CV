import cv2 as cv

from ps_6.template_tracking_motion_pf import TemplateTrackingMotionPF
from ps_6.template_tracking_pf import TemplateTrackingPF
from ps_hepers.helpers import imsave
import time


def track_patch(frames, track_box, frames_to_save, output_format, track_velocities=False, **kwargs):
    tracker_pf = TemplateTrackingMotionPF(first_frame=frames[0], track_box=track_box.flatten(), **kwargs) \
        if track_velocities else TemplateTrackingPF(first_frame=frames[0], track_box=track_box.flatten(), **kwargs)
    for i in range(len(frames)):
        t = time.time()
        frame = frames[i]
        print('Running frame %s' % i)
        tracker_pf.update(frame)
        tracker_pf.visualize_frame(frame)
        wait_time = max(1, int(33 - time.time() + t))
        if cv.waitKey(wait_time) & 0xFF == ord('q'):
            break
        cv.imshow('tracker', cv.cvtColor(frame, cv.COLOR_RGB2BGR))
        if i + 1 in frames_to_save:
            imsave(frame, output_format % (frames_to_save.index(i + 1) + 1))

