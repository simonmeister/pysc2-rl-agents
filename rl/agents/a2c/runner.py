import numpy as np

from pysc2.env.environment import StepType
from pysc2.lib import actions

from rl.pre_processing import is_spatial_action


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


def actions_to_pysc2(actions, size):
  """Convert agent action representation to FunctionCall representation."""
  fn_id, arg_ids = actions
  actions_list = []
  for n in range(fn_id.shape[0]):
    a_0 = fn_id[n]
    a_l = []
    for arg_type in actions.FUNCTIONS._func_list[a_0].args:
      arg_id = arg_ids[arg_type][n]
      if is_spatial_action[arg_type]:
        arg = [arg_id % width, arg_id // height] # TODO verify dim order (x, y) is correct
      else:
        arg = [arg_id]
      a_l.append(arg)
    action = actions.FunctionCall(a_0, a_l)
    actions_list.append(action)
  return actions_list


class A2CRunner():
  def __init__(self,
               agent,
               envs,
               is_training=True,
               n_steps=8,
               discount=0.99):
    """
    Args:
      agent: A2CAgent instance.
      envs: multienv instance.
      is_training: whether to train the agent.
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
    obs_raw = self.envs.reset() # TODO return list or ndarray of obs_raw from envs.step?
    self.last_obs = self.preproc.preprocess_obs(obs_raw)

  def run_batch():

    def flatten_first_dims(x):
      new_shape = [x.shape[0] * x.shape[1]] + x.shape[2:]
      return x.reshape(*new_shape)

    values = np.zeros((len(self.envs), self.n_steps), dtype=np.float32)
    rewards = np.zeros((len(self.envs), self.n_steps), dtype=np.float32)
    dones = np.zeros((len(self.envs), self.n_steps), dtype=np.float32)
    all_obs = []
    all_actions = []

    last_obs = self.latest_obs

    for n in range(self.n_steps):
      actions, value_estimate = self.agent.step(last_obs)
      size = last_obs['screen'].shape[1:3]

      values[:, n] = value_estimate
      all_obs.append(last_obs)
      all_actions.append(actions)

      obs_raw = envs.step(actions_to_pysc2(actions, size)) # TODO return list or ndarray of obs_raw from envs.step?
      last_obs = self.preproc.preprocess_obs(obs_raw) # TODO preprocess_obs should process batches of obs
      rewards[:, n] = [t.reward for t in obs_raw]
      dones[:, n] = [t.step_type is StepType.LAST for t in obs_raw]

    next_values = self.agent.get_value(last_obs)

    returns, advs = compute_returns_advantages(
        rewards, dones, values, next_values, self.discount)

    all_actions = ... # TODO accumulate all_actions into action structure of int32 ndarrays
    all_obs = ... # TODO accumulate all_obs into dict of single ndarrays

    obs = {k: flatten_first_dims(v) for k, v in obs.items()}
    returns = flatten_first_dims(returns)
    advs = flatten_first_dims(advs)

    self.agent.train(obs, actions, returns, advs)
