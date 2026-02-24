# //----------------------------//
# // Gymnasium Vectorized Environment Wrapper
# // Compatible with RaisimGymVecEnv interface
# //----------------------------//

import numpy as np
import gymnasium as gym
from gymnasium.vector import SyncVectorEnv


class GymnasiumVecEnv:
    """
    A wrapper around gymnasium environments that mimics the RaisimGymVecEnv interface.
    This allows seamless replacement of raisim environments with gymnasium environments.
    """

    def __init__(self, env_id, num_envs, normalize_ob=True, seed=0, normalize_rew=True, clip_obs=10.):
        """
        Initialize the vectorized gymnasium environment.
        
        Args:
            env_id: Gymnasium environment ID (e.g., 'BipedalWalker-v3', 'Humanoid-v4')
            num_envs: Number of parallel environments
            normalize_ob: Whether to normalize observations
            seed: Random seed
            normalize_rew: Whether to normalize rewards
            clip_obs: Clipping value for normalized observations
        """
        self.env_id = env_id
        self.num_envs = num_envs
        self.normalize_ob = normalize_ob
        self.normalize_rew = normalize_rew
        self.clip_obs = clip_obs
        self._seed = seed
        
        # Create vectorized environment
        def make_env():
            def _init():
                env = gym.make(env_id)
                env.reset(seed=seed)
                return env
            return _init
        
        # Create parallel environments
        env_fns = [make_env() for _ in range(num_envs)]
        self.env = SyncVectorEnv(env_fns)
        
        # Get observation and action dimensions from a single environment
        single_env = gym.make(env_id)
        self.num_obs = single_env.observation_space.shape[0]
        
        # Check for continuous action space (required for PPO)
        if isinstance(single_env.action_space, gym.spaces.Box):
            self.num_acts = single_env.action_space.shape[0]
        else:
            raise ValueError(
                f"Environment '{env_id}' has a '{type(single_env.action_space).__name__}' action space. "
                "GymnasiumVecEnv requires continuous (Box) action spaces for PPO. "
                "Please use environments like: Humanoid-v4, BipedalWalker-v3, Walker2d-v4, etc."
            )
        single_env.close()
        
        # Initialize observation storage
        self._observation = np.zeros([self.num_envs, self.num_obs], dtype=np.float32)
        self.obs_rms = RunningMeanStd(shape=[self.num_envs, self.num_obs])
        
        # Initialize reward and done tracking
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._done = np.zeros(self.num_envs, dtype=np.bool_)
        self.rewards = [[] for _ in range(self.num_envs)]
        
        # Additional info tracking (for compatibility)
        self.displacements = np.zeros([self.num_envs, 4], dtype=np.float32)
        self.reward_info = np.zeros([self.num_envs, 16], dtype=np.float32)
        
        # Track episode rewards for info
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        
        # Last observation for value calculation at episode end
        self._last_obs = None

    def seed(self, seed=None):
        """Set the seed for the environment."""
        if seed is not None:
            self._seed = seed
        self.env.reset(seed=seed)

    def set_command(self, command):
        """Placeholder for command setting (not applicable to standard gymnasium envs)."""
        pass

    def turn_on_visualization(self):
        """Placeholder for visualization."""
        pass

    def turn_off_visualization(self):
        """Placeholder for visualization."""
        pass

    def start_video_recording(self, file_name):
        """Placeholder for video recording."""
        pass

    def stop_video_recording(self):
        """Placeholder for video recording."""
        pass

    def step(self, action):
        """
        Execute one step in all parallel environments.
        
        Args:
            action: Action array of shape [num_envs, num_acts]
            
        Returns:
            rewards: Array of rewards for each env
            dones: Array of done flags for each env
        """
        # Clip actions to valid range
        action = np.clip(action, self.env.single_action_space.low, self.env.single_action_space.high)
        
        # Step all environments
        obs, rewards, terminated, truncated, infos = self.env.step(action)
        
        # Store observations
        self._observation = obs.astype(np.float32)
        self._last_obs = obs.copy()
        
        # Handle done flags (terminated or truncated)
        dones = np.logical_or(terminated, truncated).astype(np.bool_)
        
        # Store rewards and track episode metrics
        self._reward = rewards.astype(np.float32)
        for i in range(self.num_envs):
            self.rewards[i].append(rewards[i])
            self._episode_returns[i] += rewards[i]
            self._episode_lengths[i] += 1
        
        # Reset environment tracking for done environments
        for i in np.where(dones)[0]:
            self._episode_returns[i] = 0
            self._episode_lengths[i] = 0
        
        return self._reward.copy(), dones.copy()

    def load_scaling(self, dir_name, iteration, policy_type=None, num_g1=None):
        """Load observation normalization statistics from files."""
        try:
            mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
            var_file_name = dir_name + "/var" + str(iteration) + ".csv"
            loaded_mean = np.loadtxt(mean_file_name, dtype=np.float32)
            loaded_var = np.loadtxt(var_file_name, dtype=np.float32)
            self.obs_rms.mean = loaded_mean
            self.obs_rms.var = loaded_var
        except FileNotFoundError:
            print(f"Scaling files not found at {dir_name}. Using default normalization.")

    def save_scaling(self, dir_name, iteration):
        """Save observation normalization statistics to files."""
        mean_file_name = dir_name + "/mean" + str(iteration) + ".csv"
        var_file_name = dir_name + "/var" + str(iteration) + ".csv"
        np.savetxt(mean_file_name, self.obs_rms.mean)
        np.savetxt(var_file_name, self.obs_rms.var)

    def observe(self, update_mean=True):
        """
        Get current observations with optional normalization.
        
        Args:
            update_mean: Whether to update running mean/std
            
        Returns:
            Normalized or raw observations
        """
        if self.normalize_ob:
            if update_mean:
                self.obs_rms.update(self._observation)
            return self._normalize_observation(self._observation)
        else:
            return self._observation.copy()

    def reset(self):
        """Reset all environments."""
        self._reward = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self._observation, _ = self.env.reset()
        self._observation = self._observation.astype(np.float32)
        self._last_obs = self._observation.copy()

    def _normalize_observation(self, obs):
        """Normalize observations using running mean and std."""
        if self.normalize_ob:
            return np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + 1e-8),
                -self.clip_obs,
                self.clip_obs
            )
        else:
            return obs

    def get_dis(self):
        """Get displacement info (placeholder for gymnasium)."""
        return self.displacements.copy()
    
    def get_reward_info(self):
        """Get detailed reward info (placeholder for gymnasium)."""
        return self.reward_info.copy()

    def reset_and_update_info(self):
        """Reset and return episode info."""
        info = self.reset()
        return info, self._update_epi_info()

    def _update_epi_info(self):
        """Generate episode info for each environment."""
        info = [{} for _ in range(self.num_envs)]
        for i in range(self.num_envs):
            eprew = sum(self.rewards[i])
            eplen = len(self.rewards[i])
            epinfo = {"r": eprew, "l": eplen}
            info[i]['episode'] = epinfo
            self.rewards[i].clear()
        return info

    def close(self):
        """Close the environments."""
        self.env.close()

    def curriculum_callback(self):
        """Placeholder for curriculum callback."""
        pass

    def set_itr_number(self, itr_number):
        """Placeholder for iteration number setting."""
        pass

    @property
    def extra_info_names(self):
        """Return extra info names."""
        return []


class RunningMeanStd(object):
    """Running mean and std calculation for observations."""
    
    def __init__(self, epsilon=1e-4, shape=()):
        """
        Calculate running mean and std of a data stream.
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        Args:
            epsilon: Helps with arithmetic issues
            shape: Shape of the data stream's output
        """
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon

    def update(self, arr):
        """Update running statistics with new batch of observations."""
        batch_mean = np.mean(arr, axis=0)
        batch_var = np.var(arr, axis=0)
        batch_count = arr.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Update from batch moments."""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + np.square(delta) * (self.count * batch_count / (self.count + batch_count))
        new_var = m_2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count
