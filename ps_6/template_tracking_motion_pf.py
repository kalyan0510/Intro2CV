import numpy as np
import cv2 as cv

from ps_6.particle_filter import ParticleFilter
from ps_hepers.helpers import im_hist, imshow


class TemplateTrackingMotionPF:
    """
    This is an implementation of image template tracker. This is implemented over the particle filter implementation.
    The state of each particle is a vector of size 5 made of (i,j) co-ordinates, scale of window, velocity (vi,vj)
    TODO- Merge this class with TemplateTrackingPF to cut off duplicate logic
    """

    def __init__(self, first_frame=None, track_box=None, template=None, state_range=None, dynamics_std=10,
                 sigma_mse=0.5, frame_shape=None, num_particles=100, similarity_method='mse',
                 state_est='top_70_percentile', prior='uniform', t_update_factor=0):
        if first_frame is not None:
            if track_box is not None:
                x, y, w, h = track_box
                self.template = first_frame[y:y + h, x:x + w, ...].copy()
            else:
                raise Exception("Either template or tracking window box has to be provided as init arg")
            self.frame_shape = first_frame.shape
            self.state_range = np.asarray(
                [np.arange(first_frame.shape[0]), np.arange(first_frame.shape[0]), np.arange(-0.),
                 np.arange(first_frame.shape[0]), np.arange(first_frame.shape[0])])
        self.template = ([template, self.template][template is None]).astype(np.float32)
        self.validate_template_w(self.template.shape)
        self.window_hw = np.asarray(self.template.shape[:2]) // 2
        self.frame_shape = [frame_shape, self.frame_shape][frame_shape is None]
        self.dynamics_std = dynamics_std
        self.sigma_mse = sigma_mse
        self.num_particles = num_particles
        self.t_update_factor = t_update_factor
        self.particle_filter = ParticleFilter(dynamics_model_fn=self.dynamics_model,
                                              score_particle_fn=self.get_similarity_fn(similarity_method),
                                              clip_particles_fn=self.clip_particles_states, num_particles=num_particles,
                                              state_est=state_est, prior_st=self.prior_fn(prior))

    def prior_fn(self, prior):
        if prior == 'uniform':
            # i_s = np.random.choice(np.arange(self.frame_shape[0]), self.num_particles)
            # j_s = np.random.choice(np.arange(self.frame_shape[1]), self.num_particles)
            i_s = np.random.uniform(0, self.frame_shape[0], self.num_particles)
            j_s = np.random.uniform(0, self.frame_shape[1], self.num_particles)
            scales_s = np.random.uniform(np.linspace(0.8, 1.2, self.num_particles), self.num_particles)
            vi_s = np.random.uniform(np.linspace(-5, 5, self.num_particles), self.num_particles)
            vj_s = np.random.uniform(np.linspace(-5, 5, self.num_particles), self.num_particles)
            return np.concatenate([i[:, np.newaxis] for i in [i_s, j_s, scales_s, vi_s, vj_s]], axis=1).astype(
                np.float32)

    def get_similarity_fn(self, similarity_method):
        if similarity_method == 'mse':
            return self.score_particle_mse
        if similarity_method == 'mseblur':
            return self.score_particle_mse_blur
        elif similarity_method == 'ncorr':
            return self.score_particle_ncorr
        elif similarity_method == 'corr':
            return self.score_corr
        elif similarity_method == 'histcmp':
            return self.score_particle_hist_chi_sq
        else:
            raise Exception("no such similarity method %s" % similarity_method)

    def dynamics_model(self, particle_states, action):
        # Xt = Xt-1 + Vt-1
        particle_states[:, :2] += particle_states[:, 3:]
        # add noise to window sizes
        particle_states[:, 2] += np.random.normal(0, 0.1, particle_states.shape[0])
        # add noise to particle velocities
        particle_states[:, 3:] += np.random.normal(0, self.dynamics_std, (particle_states.shape[0], 2))
        return particle_states

    def get_candidate_patches(self, particle_states, observation):
        candidates = []
        for state in particle_states:
            pos = state[:2].round().astype(np.int32)
            win_hw = (self.window_hw * state[2]).astype(np.int32)
            patch = observation[(pos[0] - win_hw[0]):(pos[0] + win_hw[0] + 1),
                    (pos[1] - win_hw[1]):(pos[1] + win_hw[1] + 1),
                    ...].astype(np.float32)
            candidates.append(cv.resize(patch, self.template.shape[1::-1]))
        return candidates

    def score_particle_mse(self, particle_states, observation):
        scores = [np.exp(-np.square(np.subtract(patch, self.template)).mean() / (
                2 * self.sigma_mse ** 2)) for patch in
                  self.get_candidate_patches(particle_states, observation)]
        return scores

    def score_particle_mse_blur(self, particle_states, observation):
        x = cv.GaussianBlur(self.template, (3, 3), 0)
        scores = [np.square(np.subtract(cv.GaussianBlur(patch, (3, 3), 0), x)).mean() for patch in
                  self.get_candidate_patches(particle_states, observation)]
        scores = np.asarray([np.exp(-score / (2 * self.sigma_mse ** 2)) for score in scores])
        """
        This below code line (scores[scores < 1] = 0.001) is a big saviour at least to track the 'blond-haired woman'. 
        
        What this does is, it keeps the less confident particles from deviating the tracker.
        
        Image distance methods like MSE can be thought of a probabilistic methods. That means, a black-white squared 
        checker board can be half as similar as a white square board to the target 'white square board'.
        While this is fundamental for any particle filter method to work, this can be problematic if the particle 
        weights are adjusted from in-confident scores.
        
        For example: When we are tracking the 'blond-haired woman' from the template, the surviving tracking particles
        are usually distributed right at the center of template. Because the particles at the center score high and gain
        higher weights and thus survive in a sampling step.
        
        But when an occlusion (a man in the black coat) comes, the (relatively white) target is blocked and the
        particles lying out side (on white walls) may gain higher scores and survive the sampling step, while the 
        particles on (black) occlusion do not. This makes the particles to shift along with the occlusion and lose the
        track on target. 
        One way to deal with this is by giving same scores to all the less-scoring particles.
        
        The values 1000000 and 0.01 and just some sweet spots I have found while tracking the blond-haired woman. 
        """
        scores = scores * 1000000
        scores[scores < 1] = 0.001
        return scores

    def score_corr(self, particle_states, observation):
        t = (self.template - self.template.mean())
        scores = np.asarray([(t * (p - p.mean())).mean()
                             for p in self.get_candidate_patches(particle_states, observation)])
        scores[scores < 700] = 1
        return scores

    def score_particle_ncorr(self, particle_states, observation):
        score = [
            np.exp(np.pi / 3 - (np.pi - np.arccos(np.mean((self.template - self.template.mean()) * (p - p.mean())) / (
                    np.std(p) * np.std(self.template)))) / 2 * self.sigma_mse ** 2)
            for p in self.get_candidate_patches(particle_states, observation)]
        return score

    def score_particle_hist_chi_sq(self, particle_states, observation):
        candidate_patches = self.get_candidate_patches(particle_states, observation)
        t_hist = im_hist(self.template, normed=True)

        def chi_sq(h1, h2):
            x = h1 + h2
            return np.divide((h1 - h2) ** 2, x, where=x != 0).sum() * 0.5

        hists = [im_hist(p, normed=True) for p in candidate_patches]
        return [np.exp(-chi_sq(h, t_hist) / (2 * 0.068 ** 2)) for h in hists]

    def clip_particles_states(self, particle_states):
        """
        uncomment this snippet to cut off the particles falling outside instead of clipping to nearest bound
        return particle_states[
            ((self.window_hw[0] <= particle_states[:, 0]) * (
                        particle_states[:, 0] <= (self.frame_shape[0] - self.window_hw[0]))) * (
                        (self.window_hw[1] <= particle_states[:, 1]) * (
                            particle_states[:, 1] <= (self.frame_shape[1] - self.window_hw[1]))), ...]
        """
        # clip the scales
        particle_states[:, 2] = np.clip(particle_states[:, 2], 0.3, 2.0)
        # use clipped scales to clip the positions
        win_h = (self.window_hw[0] * particle_states[:, 2]).round().astype(np.int32)
        win_w = (self.window_hw[1] * particle_states[:, 2]).round().astype(np.int32)
        particle_states[win_h > self.frame_shape[0] // 2, 2] = 1.0
        particle_states[win_w > self.frame_shape[1] // 2, 2] = 1.0
        win_h = (self.window_hw[0] * particle_states[:, 2]).round().astype(np.int32)
        win_w = (self.window_hw[1] * particle_states[:, 2]).round().astype(np.int32)
        particle_states[:, 0] = np.clip(particle_states[:, 0], win_h,
                                        self.frame_shape[0] - win_h - 1)
        particle_states[:, 1] = np.clip(particle_states[:, 1], win_w,
                                        self.frame_shape[1] - win_w - 1)
        # clip particle velocities
        particle_states[:, 3] = np.clip(particle_states[:, 3], -5, 5)
        particle_states[:, 4] = np.clip(particle_states[:, 4], -5, 5)
        return particle_states

    def update(self, frame):
        self.particle_filter.update(frame, None)
        if self.t_update_factor != 0:
            self.update_template(frame)
        # return self.show_dmap(frame)

    def show_dmap(self, frame):
        """
        Method that helps see the scores of all the patches on the image
        """
        di = np.zeros(frame.shape)
        x = cv.GaussianBlur(self.template, (3, 3), 0)
        for i in range(self.window_hw[0], frame.shape[0] - self.window_hw[0]):
            for j in range(self.window_hw[1], frame.shape[1] - self.window_hw[1]):
                p = frame[(i - self.window_hw[0]):(i + self.window_hw[0] + 1),
                    (j - self.window_hw[1]):(j + self.window_hw[1] + 1),
                    ...].astype(np.float32)
                di[i, j] = np.exp(
                    -np.square(np.subtract(cv.GaussianBlur(p, (3, 3), 0), x)).mean() / (2 * self.sigma_mse ** 2))
        return di / di.max()

    def update_template(self, frame):
        particle_pos_xy = self.particle_filter.get_particle_states()[:, 1::-1]
        center_xy = particle_pos_xy.mean(axis=0)
        r = (np.linalg.norm(particle_pos_xy - center_xy, axis=1) * self.particle_filter.get_particle_weights()).sum()
        state = self.particle_filter.get_state()
        pos = state[:2].round().astype(np.int32)
        win_hw = (self.window_hw * state[2]).astype(np.int32)
        mn_i = int(pos[0] - win_hw[0])
        mx_i = int(pos[0] + win_hw[0] + 1)
        mn_j = int(pos[1] - win_hw[1])
        mx_j = int(pos[1] + win_hw[1] + 1)
        best = frame[mn_i:mx_i, mn_j:mx_j, ...]
        """
        below line is again something added to make sure the updates do not happen when scores are less confident.
        """
        tf = self.t_update_factor * ((4 if r < 6 else 1) if r < 15 else 0)
        self.template = cv.resize(best, self.template.shape[1::-1]) * tf + (
                1 - tf) * self.template

    def visualize_frame(self, frame):
        self.draw_particles(frame)
        self.draw_window(frame)
        self.draw_std_circle(frame)
        self.overlay_template(frame, 0.5)

    def overlay_template(self, frame, size=0.1):
        shape = (np.asarray(self.template.shape[:2]) * size).astype(np.int32)
        frame[:shape[0], :shape[1], ...] = cv.resize(self.template, shape[1::-1])

    def draw_particles(self, frame):
        [cv.circle(frame, p[1::-1].astype(np.int32), radius=int(max(min((w * self.num_particles), 5), 1)),
                   color=(0, 255, 0)) for p, w in
         zip(self.particle_filter.get_particle_states(), self.particle_filter.get_particle_weights())]

    def draw_window(self, frame):
        state = self.particle_filter.get_state()
        center_xy = state[1::-1]
        win_hw = (self.window_hw * state[2]).astype(np.int32)
        cv.rectangle(frame, (center_xy - win_hw[1::-1]).astype(np.int32),
                     (center_xy + win_hw[1::-1]).astype(np.int32), color=(0, 0, 255), thickness=1)

    def draw_std_circle(self, frame):
        particle_pos_xy = self.particle_filter.get_particle_states()[:, 1::-1]
        center_xy = particle_pos_xy.mean(axis=0)
        r = (np.linalg.norm(particle_pos_xy - center_xy, axis=1) * self.particle_filter.get_particle_weights()).sum()
        r = 1 if r is np.NaN else int(r)
        cv.circle(frame, center_xy.astype(np.int32), radius=r, color=(255, 0, 0), thickness=1)

    @staticmethod
    def validate_template_w(shape):
        if not shape[0] % 2 and shape[1] % 2:
            raise Exception('template shape cannot have even sizes')
