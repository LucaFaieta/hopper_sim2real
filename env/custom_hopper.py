"""Implementation of the Hopper environment supporting
domain randomization optimization."""
import csv
from math import sqrt
import pdb
from copy import deepcopy

import numpy as np
import gym
from gym import utils
from .mujoco_env import MujocoEnv
from scipy.stats import truncnorm

class CustomHopper(MujocoEnv, utils.EzPickle):
    def __init__(self, domain=None):
        MujocoEnv.__init__(self, 4)
        utils.EzPickle.__init__(self)
        self.domain = domain
        self.M = deepcopy(self.sim.model.body_mass[2:])
        self.S = np.array([1.0, 1.0, 1.0])
        self.MG = np.radians(30.0)
        self.SG = np.sqrt(30)
        self.muG = np.radians(30.0)
        self.sigmaG = np.sqrt(30)
        self.mu = np.array([])
        self.s = np.array([])
        self.original_masses = np.copy(self.sim.model.body_mass[1:])    # Default link masses

        if domain != 'target':  # Source environment has an imprecise torso mass (1kg shift)
            self.sim.model.body_mass[1] -= 1.0
            if domain == "source":
                self.set_random_parameters()
            if domain == "inclined":
                self.set_inclinations()
                #self.change_gravity()

    def get_M(self):
      return self.M

    def get_S(self):
      return self.S

    def print_dictionaries(self):
      print(type(self.M))
      print(f"M element is {type(self.M[0])}")
      print(type(self.S))
      print(f"S element is {type(self.S[0])}")
      print(type(self.mu))
      print(f"mu element is {type(self.mu[0])}")
      print(type(self.s))
      print(f"s element is {type(self.s[0])}")


    def move_distribution(self):
          for i in range(len(self.M)): ### for each parameter, set new mu and sigma
            self.M[i] = self.mu[i]
            self.S[i] = self.s[i]
            self.MG = self.muG
            self.SG = self.sigmaG


    def sampling_distribution(self):
        self.mu = np.array([])
        self.s = np.array([])
        for i in range(len(self.M)):
          self.mu = np.append(self.mu, truncnorm(a = 0, b = 2*self.original_masses[i], scale = 1, loc = self.M[i]).rvs(size=1)[0]) 
          self.s = np.append(self.s, truncnorm(a = 0, b = (2*self.S[i]), scale = 1, loc = self.S[i]).rvs(size=1)[0])
        self.muG = truncnorm(a = 0,b = 2*self.MG,scale = np.sqrt(3), loc = self.MG).rvs(size = 1)[0]
        self.sigmaG = truncnorm(a = 0,b = 2*self.SG, scale = np.sqrt(3), loc = self.SG).rvs(size = 1)[0]



    
    def adr_sample_parameters(self):    
        """Sample parameters"""
        masses = []
        thigh = truncnorm(a = 0,b = 2*self.original_masses[1],scale = self.s[0], loc =self.mu[0]).rvs(size = 1)[0]
        leg = truncnorm(a = 0,b = 2*self.original_masses[2],scale = self.s[1], loc = self.mu[1]).rvs(size = 1)[0]
        foot = truncnorm(a = 0,b = 2*self.original_masses[3],scale = self.s[2], loc = self.mu[2]).rvs(size = 1)[0]
        
        masses.append(thigh)
        masses.append(leg)
        masses.append(foot)
        
        '''
        print(thigh)
        print(leg)
        print(foot)
        '''
        return masses


    def set_random_parameters(self):
        """Set random masses"""
        self.set_parameters(self.sample_parameters())

    def sample_parameters(self):
        """Sample parameters"""
        masses = []
        thigh = truncnorm(a = 0,b = 2*self.original_masses[1],scale = np.sqrt(3), loc = self.original_masses[1]).rvs(size = 1)[0]
        leg = truncnorm(a = 0,b = 2*self.original_masses[2],scale = np.sqrt(3), loc = self.original_masses[2]).rvs(size = 1)[0]
        foot = truncnorm(a = 0,b = 2*self.original_masses[3],scale = np.sqrt(3), loc = self.original_masses[3]).rvs(size = 1)[0]
        masses.append(thigh)
        masses.append(leg)
        masses.append(foot)
        
        '''print(thigh)
        print(leg)
        print(foot)'''
        
        return masses

    def get_parameters(self):
        """Get value of mass for each link"""
        masses = np.array( self.sim.model.body_mass[1:] )
        return masses

    def set_inclinations(self):
        
        angle = truncnorm(a = 0,b = 2*self.MG ,scale = self.sigmaG, loc = self.muG).rvs(size = 1)[0]
        inclination_angle = np.radians(angle)  # Adjust the inclination
        # Set the gravity vector to represent the inclined environment
        gravity_magnitude = 9.81  # Adjust g as needed
        gravity_vector = np.array([0.0, 0.0, -gravity_magnitude])
        gravity_rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(inclination_angle), -np.sin(inclination_angle)],
            [0, np.sin(inclination_angle), np.cos(inclination_angle)],
        ])
        gravity_vector_inclined = gravity_rotation_matrix.dot(gravity_vector)

        self.model.opt.gravity[:] = gravity_vector_inclined

    def change_gravity(self):
        self.model.opt.gravity[2] = truncnorm(a = -5,b = 3*9.81/2 , loc = 9.81,scale = 2.3).rvs(size = 1)[0]

    def set_parameters(self, task : list):
        """Set each hopper link's mass to a new value"""
        self.sim.model.body_mass[2] = task[0]
        self.sim.model.body_mass[3] = task[1]
        self.sim.model.body_mass[4] = task[len(task)-1]


    def step(self, a):
        """Step the simulation to the next timestep

        Parameters
        ----------
        a : ndarray,
            action to be taken at the current timestep
        """
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()

        return ob, reward, done, {}

    def _get_obs(self):
        """Get current state"""
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat
        ])

    def reset_model(self):
        """Reset the environment to a random initial state"""
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        if self.domain == "source":
            self.set_random_parameters()
        if self.domain == "inclined":
            self.set_inclinations()
        if self.domain == "automatic":            
            self.sampling_distribution()
            self.adr_sample_parameters()
        if self.domain == "autoinclined":
            self.set_inclinations()            
            self.sampling_distribution()
            self.adr_sample_parameters()

        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20



"""
    Registered environments
"""
gym.envs.register(
        id="CustomHopper-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
)

gym.envs.register(
        id="CustomHopper-source-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "source"}
)

gym.envs.register(
        id="CustomHopper-target-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "target"}
)

gym.envs.register(
        id="CustomHopper-source-inclined-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "inclined"}
)

gym.envs.register(
        id="CustomHopper-source-automatic-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "automatic"}
)

gym.envs.register(
        id="CustomHopper-source-autoinclined-v0",
        entry_point="%s:CustomHopper" % __name__,
        max_episode_steps=500,
        kwargs={"domain": "autoinclined"}
)

