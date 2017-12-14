import numpy as np

from pysc2.env.environment import StepType
from pysc2.lib.actions import FunctionCall, FUNCTIONS

from rl.pre_processing import is_spatial_action, concat_ndarray_dicts


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
    for arg_type in FUNCTIONS._func_list[a_0].args:
      arg_id = arg_ids[arg_type][n]
      if is_spatial_action[arg_type]:
        arg = [arg_id % width, arg_id // height] # TODO verify dim order (x, y) is correct
      else:
        arg = [arg_id]
      a_l.append(arg)
    action = FunctionCall(a_0, a_l)
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
      envs: SubprocVecEnv instance.
      is_training: whether to train the agent.
      n_steps: number of agent steps for collecting rollouts.
      discount: reward discount.
    """
    self.agent = agent
    self.envs = envs
    self.is_training = is_training
    self.n_steps = n_steps
    self.discount = discount
    self.preproc = Preprocessor(self.envs.observation_spec()[0])

  def reset(self):
    obs_raw = self.envs.reset()
    self.last_obs = self.preproc.preprocess_obs(obs_raw)

  def run_batch():

    def flatten_first_dims(x):
      new_shape = [x.shape[0] * x.shape[1]] + x.shape[2:]
      return x.reshape(*new_shape)

    def flatten_first_dims_dict(x):
      return {k: flatten_first_dims(v) for k, v in x.items()}

    def concat_and_flatten_actions(lst, axis=0):
      fn_id_list, arg_dict_list = zip(*lst)
      fn_id = np.concatenate(fn_id_list, axis=axis)
      fn_id = flatten_first_dims(fn_id)
      arg_ids = concat_ndarray_dicts(arg_dict_list, axis=axis)
      arg_ids = flatten_first_dims_dict(arg_ids)
      return (fn_id, arg_ids)

    shapes = (self.envs.n_envs, self.n_steps)
    values = np.zeros(shapes, dtype=np.float32)
    rewards = np.zeros(shapes, dtype=np.float32)
    dones = np.zeros(shapes, dtype=np.float32)
    all_obs = []
    all_actions = []

    last_obs = self.latest_obs

    for n in range(self.n_steps):
      actions, value_estimate = self.agent.step(last_obs)
      size = last_obs['screen'].shape[1:3]

      values[:, n] = value_estimate
      all_obs.append(last_obs)
      all_actions.append(actions)

      obs_raw = envs.step(actions_to_pysc2(actions, size))
      last_obs = self.preproc.preprocess_obs(obs_raw)
      rewards[:, n] = [t.reward for t in obs_raw]
      dones[:, n] = [t.step_type is StepType.LAST for t in obs_raw]

    next_values = self.agent.get_value(last_obs)

    returns, advs = compute_returns_advantages(
        rewards, dones, values, next_values, self.discount)

    actions = concat_and_flatten_actions(all_actions)
    obs = flatten_first_dims_dict(concat_ndarray_dicts(all_obs))
    returns = flatten_first_dims(returns)
    advs = flatten_first_dims(advs)

    self.agent.train(obs, actions, returns, advs)
