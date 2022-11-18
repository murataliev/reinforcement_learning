import numpy as np

from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

class UR5eEnv(MujocoEnv, utils.EzPickle):
    
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 25
    }

    
    def __init__(self, **kwargs):
        
        utils.EzPickle.__init__(self, **kwargs)  
        
        observation_space = Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(30,), 
            dtype=np.float64
        )
        
        MujocoEnv.__init__(
            self, 
            '/home/akbar/projects/mujoco_menagerie/universal_robots_ur5e/scene.xml', 
            20,
            observation_space=observation_space, 
            **kwargs
        )
            
    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()
        observation = np.concatenate((position, velocity))
        return observation
    
    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        # vec = self.get_body_com("wrist_3_link") - self.get_body_com("target")
        # reward = -np.linalg.norm(vec)
        reward = 1
        terminated = False
        info = self.render()

        return observation, reward, terminated, False, info
    
    def reset_model(self):

        qpos = self.init_qpos + self.np_random.uniform(
            low=0.1, 
            high=0.2, 
            size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=0.1, 
            high=0.2, 
            size=self.model.nv
        )

        self.set_state(qpos, qvel)
        return self._get_obs() 
    

    def viewer_setup(self):
        assert self.viewer is not None
        self.viewer.cam.trackbodyid = 0