import os
import time

import gym
import numpy as np
import matplotlib.pyplot as plt
import torch

import f_pNew

from stable_baselines3 import PPO
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback

class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq:
    :param log_dir: Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, "best_model")
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:

          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), "timesteps")
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose >= 1:
                print(f"Num timesteps: {self.num_timesteps}")
                print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose >= 1:
                    print(f"Saving new best model to {self.save_path}")
                  self.model.save(self.save_path)

        return True


def create_model(temp_goal, interval, data, dims, model_file):

    env = gym.make('BuildingEnv-v0', dims = dims, interval=interval, goal = temp_goal, data= data)

    log_dir = "tmp/"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

    model = PPO('MlpPolicy', env, verbose=0, learning_rate=.01)
    timesteps = 5e6
    model.learn(total_timesteps= timesteps, callback = callback)
    plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "Building Env")
    plt.show()

    model.save(model_file)
    return model

def test_model(temp_goal, interval, data, dims, model_file):
    env = gym.make('BuildingEnv-v0', dims = dims, interval=interval, goal = temp_goal, data= data)
    model = PPO.load(model_file, env=env)

if __name__ == '__main__':
    temp_goal = 70
    interval = 30
    control = 9
    num_days = 100000
    data_train = np.load(f'preprocessing_output/{interval}T/features_rooms_training_{interval}T.npy').astype(np.float32)
    dims = [data_train.shape[1], data_train.shape[2] - control, control]
    # model_file = 'simulation_data\\model_batch_10'
    model_file = 'simulation_data\\trash'

    start = time.time()
    model = create_model(temp_goal, interval, data_train, dims, model_file)
    del model
    print('total time taken: ', time.time()-start)

    # data_test = np.load(f'preprocessing_output/{interval}T/features_rooms_testing_{interval}T.npy').astype(np.float32)
    # test_model(temp_goal, interval, data_test, dims, model_file)
    pass

