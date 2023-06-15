import sys
# import gymnasium as gym
import gym
# sys.modules["gym"] = gym
import numpy as np
import f_pNew
import torch
from gym.envs.registration import register



class buildingEnv(gym.Env):

    def __init__(self, dims, interval, goal, data):
        bound = 1e8#np.inf
        self.observation_space = gym.spaces.Box(low=-bound, high=bound, shape=(dims[0], dims[1]))
        self.action_space = gym.spaces.Box(low=-bound, high=bound, shape=(dims[0]* dims[2],))

        rooms = np.load(f'preprocessing_output\\merged_rooms_list.npy')
        self.model = f_pNew.Model(80, [19, 20, 20, 1], rooms)
        self.model.load_state_dict(torch.load(f'runtime data\\model{interval}_save.pt'))
        self.model = self.model.to(torch.device('cpu'))

        self.data = data
        self.dims = self.data.shape
        self.interval = interval
        self.disturbs = dims[1]

        self.current_vals = None
        self.goal = goal
        self.start_loc = 0
        self.step_num = 0


    def _get_obs(self):
        return self.current_vals

    def _get_info(self):
        return self.goal

    def reset(self, seed=None, options=None):
        # super().reset()
        # get piece of data to be first observation
        # self.start_loc = int(np.random.randint(low=0, high=(self.dims[0] - 1440 / self.interval), size=(1, 1)))
        # self.start_loc = int(np.random.randint(low=0, high=4, size=(1, 1)))
        self.start_loc = 0
        self.step_num = 0
        observation = self.data[self.start_loc, :, 0:self.disturbs]
        self.current_vals = observation
        return self.current_vals

    def step(self, action):

        # choose set of input decisions
        action = np.reshape(action, (80,9))
        # combine with observation
        complete_features = torch.cat((torch.from_numpy(self._get_obs()), torch.from_numpy(action)), dim=1)
        # predict new temperature
        new_temp = self.model.predict(complete_features[:, 1], complete_features).detach().numpy()
        # print(np.average(new_temp))
        # get reward
        error = np.linalg.norm(self.goal- new_temp)
        reward = -error
        # make next observation
        self.step_num += 1
        new_obs = np.copy(self.data[self.start_loc + self.step_num, :, :])
        # new_obs[:, 1] = new_temp
        new_obs[:, 1] = self.goal
        self.current_vals = new_obs[:, 0:self.disturbs]
        # check for end
        max_time = self.interval * 0
        if self.step_num * self.interval >= max_time:
            terminated = True
        else:
            terminated = False

        info = {}
        return self.current_vals, reward, terminated, info


if __name__ == '__main__':
    temp_goal = 70
    interval = 30
    control = 9
    num_days = 100000
    data = np.load(f'preprocessing_output/{interval}T/features_rooms_testing_{interval}T.npy').astype(np.float32)
    dims = [data.shape[1], data.shape[2] - control, control]

    env = buildingEnv(dims, interval, temp_goal, data)
    pass
