import numpy as np

from pysc2.env.environment import StepType


# TODO: implement multienv, with methods reset(), step(actions), len, envs[i] (return env i)
# see sc2aibot


def compute_returns_advantages(rewards, dones, values, next_values, discount):
  """Compute returns and advantages from received rewards and value estimates.

  Args:
    rewards: array of shape [n_env, n_steps] containing received rewards.
    dones: array of shape [n_env, n_steps] indicating whether an episode is
      finished after a time step.
    values: array of shape [n_env, n_steps] containing estimated values.
    next_values: array of shape [n_env] containing estimated values after the
      last step for each environment.
    discount: scalar discount for future rewards.

  Returns:
    returns: array of shape [n_env, n_steps]
    advs: array of shape [n_env, n_steps]
  """
  returns = np.zeros([rewards.shape[0] + 1, rewards.shape[1]])
  advs = np.zeros_like(rewards)

  returns[-1, :] = next_values
  for t in reversed(range(rewards.shape[0])):
    future_rewards = discount * returns[t + 1, :] * (1 - dones[t, :])
    returns[t, :] = rewards[t, :] + future_rewards
    advs[t, :] = returns[t, :] - values[t, :]
  return returns[:-1, :], advs


class A2CRunner():
  def __init__(self, agent, envs, is_training=True, n_steps, discount):
    """
    Args:
      agent: A2CAgent instance.
      envs:
      is_training:
      n_steps: number of agent steps for collecting rollouts.
      discount: reward discount.
    """
    self.agent = agent
    self.envs = envs
    self.is_training = is_training
    self.n_steps = n_steps
    self.discount = discount
    self.preproc = Preprocessor(self.envs[0].observation_spec)

  def reset(self):
    obs_raw = self.envs.reset()
    self.last_obs = self.preproc.preprocess_obs(obs_raw)

  def run_batch():

    def flatten_first_dims(x):
      new_shape = [x.shape[0] * x.shape[1]] + x.shape[2:]
      return x.reshape(*new_shape)

    values = np.zeros((len(self.envs), self.n_steps), dtype=np.float32)
    rewards = np.zeros((len(self.envs), self.n_steps), dtype=np.float32)
    dones = np.zeros((len(self.envs), self.n_steps), dtype=np.float32)
    all_obs = []

    last_obs = self.latest_obs

    for n in range(self.n_steps):
      actions, value_estimate = self.agent.step(last_obs)

      values[:, n] = value_estimate
      obs.append(last_obs)

      obs_raw = envs.step(actions) # TODO
      last_obs = self.preproc.preprocess_obs(obs_raw) # TODO this should return a ndarray of obs
      rewards[:, n] = [t.reward for t in obs_raw]
      dones[:, n] = [t.step_type is StepType.LAST for t in obs_raw]

      #for t in obs_raw:
      #   if t.last():
      #       self._handle_episode_end(t) # TODO (see github sc2aibot)

    next_values = self.agent.get_value(last_obs)

    returns, advs = compute_returns_advantages(
        rewards, dones, values, next_values, self.discount)

    actions = ... # TODO sc2 actions to agent action format

    obs = (flatten_first_dims(x) for x in latest_obs)
    actions = flatten_first_dims(actions) # TODO dict shape
    returns = flatten_first_dims(returns)
    advs = flatten_first_dims(advs)
    actions = flatten_first_dims(advs)

    self.agent.train(obs, actions, all_returns, all_advs)
