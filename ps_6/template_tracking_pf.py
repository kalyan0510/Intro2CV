import numpy as np
import cv2 as cv

from ps_6.particle_filter import ParticleFilter


class TemplateTrackingPF:
    """
    This is an implementation of image template tracker. This is implemented over the particle filter implementation.
    """

    def __init__(self, first_frame=None, track_box=None, template=None, state_range=None, dynamics_std=10,
                 sigma_mse=0.5, frame_shape=None, num_particles=100, similarity_method='mse',
                 state_est='top_70_percentile', prior='gaussian', t_update_factor=0):
        if first_frame is not None:
            if track_box is not None:
                x, y, w, h = track_box
                self.template = first_frame[y:y + h, x:x + w, ...].copy()
            else:
                raise Exception("Either template or tracking window box has to be provided as init arg")
            self.frame_shape = first_frame.shape
            self.state_range = np.asarray([np.arange(x) for x in first_frame.shape[:2]])
        self.template = ([template, self.template][template is None]).astype(np.float32)
        self.validate_template_w(self.template.shape)
        self.window_hw = np.asarray(self.template.shape[:2]) // 2
        self.frame_shape = [frame_shape, self.frame_shape][frame_shape is None]
        self.state_range = [state_range, self.state_range][state_range is None]
        self.dynamics_std = dynamics_std
        self.sigma_mse = sigma_mse
        self.num_particles = num_particles
        self.t_update_factor = t_update_factor
        self.particle_filter = ParticleFilter(self.state_range, dynamics_model_fn=self.dynamics_model,
                                              score_particle_fn=self.get_similarity_fn(similarity_method),
                                              clip_particles_fn=self.clip_particles_states, num_particles=num_particles,
                                              state_est=state_est, prior_st=self.prior_fn(prior, track_box))

    def prior_fn(self, prior, track_box):
        x, y, w, h = track_box
        if prior == 'gaussian':
            return self.clip_particles_states(
                np.asarray([y + h // 2, x + w // 2]) + np.random.normal(0, np.linalg.norm(track_box[2:]),
                                                                        (self.num_particles, 2)))
        elif prior == 'random':
            i_s = np.random.choice(np.arange(self.frame_shape[0]), self.num_particles)
            j_s = np.random.choice(np.arange(self.frame_shape[1]), self.num_particles)
            return np.concatenate([i_s[:, np.newaxis], j_s[:, np.newaxis]], axis=1).astype(np.float32)
        elif prior == 'uniform':
            i_s = np.random.uniform(0, self.frame_shape[0], self.num_particles)
            j_s = np.random.uniform(0, self.frame_shape[1], self.num_particles)
            return np.concatenate([i_s[:, np.newaxis], j_s[:, np.newaxis]], axis=1).astype(np.float32)

    def dynamics_model(self, particle_states, action):
        particle_states += np.random.normal(0, self.dynamics_std, particle_states.shape)
        return particle_states

    def get_similarity_fn(self, similarity_method):
        if similarity_method == 'mse':
            return self.score_particle_mse
        elif similarity_method == 'ncorr':
            return self.score_particle_ncorr
        else:
            raise Exception("no such simmilarity method %s" % similarity_method)

    def score_particle_mse(self, particle_states, observation):
        candidate_patches = [observation[int(pos[0]) - self.window_hw[0]:int(pos[0]) + self.window_hw[0] + 1,
                             int(pos[1]) - self.window_hw[1]:int(pos[1]) + self.window_hw[1] + 1, ...].astype(
            np.float32) for pos in particle_states]
        return [np.exp(-np.square(np.subtract(patch, self.template)).mean() / (
                2 * self.sigma_mse ** 2)) for patch in
                candidate_patches]

    def score_particle_ncorr(self, particle_states, observation):
        candidate_patches = [observation[int(pos[0]) - self.window_hw[0]:int(pos[0]) + self.window_hw[0] + 1,
                             int(pos[1]) - self.window_hw[1]:int(pos[1]) + self.window_hw[1] + 1, ...].astype(
            np.float32) for pos in particle_states]
        return [1+np.mean((self.template - self.template.mean()) * (p - p.mean())) / (np.std(p) * np.std(self.template))
                for p in candidate_patches]

    def clip_particles_states(self, particle_states):
        """
        uncomment this snippet to cut off the particles falling outside instead of clipping to nearest bound
        return particle_states[
            ((self.window_hw[0] <= particle_states[:, 0]) * (
                        particle_states[:, 0] <= (self.frame_shape[0] - self.window_hw[0]))) * (
                        (self.window_hw[1] <= particle_states[:, 1]) * (
                            particle_states[:, 1] <= (self.frame_shape[1] - self.window_hw[1]))), ...]
        """
        particle_states[:, 0] = np.clip(particle_states[:, 0], self.window_hw[0],
                                        self.frame_shape[0] - self.window_hw[0] - 1)
        particle_states[:, 1] = np.clip(particle_states[:, 1], self.window_hw[1],
                                        self.frame_shape[1] - self.window_hw[1] - 1)
        return particle_states

    def update(self, frame):
        self.particle_filter.update(frame, None)
        if self.t_update_factor != 0:
            self.update_template(frame)

    def update_template(self, frame):
        pos = self.particle_filter.get_state()[:2]
        best = frame[int(pos[0]) - self.window_hw[0]:int(pos[0]) + self.window_hw[0] + 1,
               int(pos[1]) - self.window_hw[1]:int(pos[1]) + self.window_hw[1] + 1, ...]
        self.template = best * self.t_update_factor + (1 - self.t_update_factor) * self.template

    def visualize_frame(self, frame):
        self.draw_particles(frame)
        self.draw_window(frame)
        self.draw_std_circle(frame)
        self.overlay_template(frame)

    def overlay_template(self, frame):
        frame[:self.template.shape[0], :self.template.shape[1], ...] = self.template

    def draw_particles(self, frame):
        [cv.circle(frame, p[1::-1].astype(np.int32), radius=max(min(int(w * self.num_particles), 5), 1),
                   color=(0, 255, 0)) for p, w in
         zip(self.particle_filter.get_particle_states(), self.particle_filter.get_particle_weights())]

    def draw_window(self, frame):
        center_xy = self.particle_filter.get_state()[1::-1]
        cv.rectangle(frame, (center_xy - self.window_hw[1::-1]).astype(np.int32),
                     (center_xy + self.window_hw[1::-1]).astype(np.int32), color=(0, 0, 255), thickness=1)

    def draw_std_circle(self, frame):
        center_xy = self.particle_filter.get_state()[1::-1]
        particle_pos_xy = self.particle_filter.get_particle_states()[:, 1::-1]
        r = (np.linalg.norm(particle_pos_xy - center_xy, axis=1) * self.particle_filter.get_particle_weights()).sum()
        cv.circle(frame, center_xy.astype(np.int32), radius=int(r), color=(255, 0, 0), thickness=1)

    @staticmethod
    def validate_template_w(shape):
        if not shape[0] % 2 and shape[1] % 2:
            raise Exception('template shape cannot have even sizes')
