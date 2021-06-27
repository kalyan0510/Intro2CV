import random

import numpy as np


class ParticleFilter:
    """
    Particle filter implementation needs a state to be maintained. So, we need to create a state-ful python class that
    maintains the particles and updates their states according to the new measurements/control-inputs.
    """

    def __init__(self, state_range, dynamics_model_fn, score_particle_fn, diffuse_fn=lambda x: x,
                 clip_particles_fn=lambda x: x, num_particles=100):
        self.state_range = state_range
        self.num_particles = num_particles
        self.particle_states = np.asarray(
            [random.sample(list(range_), num_particles) for range_ in self.state_range]).T.astype(np.float32)
        self.particle_weights = np.ones(num_particles) / num_particles
        self.dynamics_model = dynamics_model_fn
        self.score_particle_fn = score_particle_fn
        self.diffuse_fn = diffuse_fn
        self.clip_particles_fn = clip_particles_fn
        self.state = self.particle_states[0]

    def update(self, observation, action):
        """
        A good visualization of what update method does
         https://www.codeproject.com/KB/recipes/865934/introduction-4.PNG
        :param observation: sensor observation (Zt)
        :param action: action input to the system (Ut)
        :return:
        """
        self.sample()
        # p(Xt|Xt-1,Ut)
        self.particle_states = self.dynamics_model(self.particle_states, action)
        self.particle_states = self.diffuse_fn(self.particle_states)
        self.particle_states = self.clip_particles_fn(self.particle_states)
        # p(Zt|Xt)
        self.observe(observation)
        self.update_state()
        return self.state

    def sample(self):
        j = np.random.choice(np.arange(self.num_particles), self.num_particles, True, p=self.particle_weights)
        self.particle_states = self.particle_states[j]
        # weights are not set to constants as they are reset in the observation step

    def observe(self, observation):
        scores = np.asarray(self.score_particle_fn(self.particle_states, observation))
        self.particle_weights = scores / scores.sum()

    def update_state(self):
        self.state = (self.particle_states * self.particle_weights.reshape((-1, 1))).sum(axis=0)

    def get_particle_states(self):
        return self.particle_states

    def get_particle_weights(self):
        return self.particle_weights

    def get_state(self):
        return self.state
