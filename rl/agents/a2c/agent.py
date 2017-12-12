import tensorflow as tf


# TODO the code here is still incomplete (many snippets do not fit together yet)


def mask_invalid_actions(batch_valid_actions, fn_pi):
  fn_pi *= batch_valid_actions
  fn_pi /= tf.reduce_sum(fn_pi, axis=1, keep_dims=True)
  return fn_pi


# TODO implement sample (see baselines/a2c/utils.py)


def sample_independent(batch_valid_actions, policy, size):
  fn_pi, arg_pis = policy
  fn_pi = mask_invalid_actions(batch_valid_actions, fn_pi)
  fn_samples = sample(fn_pi)

  arg_samples = dict()
  for arg_type, arg_out in arg_pis.items():
    is_spatial, arg_pi = arg_out
    sampled_arg = sample(arg_pi)
    if is_spatial:
      height, width = size
      arg_sample = [sampled_arg % width, sampled_arg // height]
    else:
      arg_sample = [sampled_arg]
    arg_samples[arg_type] = arg_sample

  return fn_samples, arg_samples

#  valid_actions_list = [obs.observation['available_actions'] for obs in obs_list]
#  batch_valid_actions = np.stack(valid_actions_list, axis=0) # TODO should be an argument

def step(batch_valid_actions, obs, policy, value):
  fn_samples, arg_samples = sample_independent(batch_valid_actions, policy)
  arg_samples_np, fn_samples_np, value_np = sess.run(
      [fn_samples, arg_samples, value], feed_dict=get_feed_dict(obs)) # TODO get_feed_dict


  # TODO return action format as needed by compute_total_log_probs and create FunctionCall objects
  # as postprocessing?
  actions_list = []
  for n in range(samples_np.shape[0]):
    a_0 = fn_samples_np[n]
    a_l = []
    for arg_type in actions.FUNCTIONS._func_list[a_0].args:
      a_l.append(arg_samples_np[arg_type])
    action = actions.FunctionCall(a_0, a_l)
    actions_list.append(action)

  return actions_list, values


class A2CAgent():
  def __init__(self):
    pass

  def build(self):
    """Create tensorflow graph for A2C agent."""
    # FullyConv()
    self.policy = ...
    self.value = ...
    pass

  def train(self, obs, actions, returns, advs):
    """
    Args:
      obs: tuple with preprocessed observation arrays, with num_batch elements
        in the first dimensions.
      actions: see `compute_total_log_probs`
      returns: array of shape [num_batch]
      advs: array of shape [num_batch]
    """
    policy = ... # TODO sess.run with obs to get policy dict (see compute_total_log_probs input)

  def step(self, batch_valid_actions, obs):
    """
    Returns:
      actions: `compute_total_log_probs`
      values: array of shape [num_batch] containing value estimates.
    """
    policy = ... 
    value = ...
    step(batch_valid_actions, obs, policy, value)

  def get_value(self, obs):
    pass



# TODO assemble "actions" argument for compute_total_log_probs

def compute_total_log_probs(batch_valid_actions, policy, actions):
  """Compute action log probabilities given predicted policies and selected
  actions.

  Args:
    batch_valid_actions: one-hot (in last dimenson) tensor of shape
      [num_batch, NUM_FUNCTIONS].
    policy: [fn_pi, {arg_0: arg_0_pi, ..., arg_n: arg_n_pi}]], where
      each value is a tensor of shape [num_batch, num_params] representing
      probability distributions over the function ids or over discrete
      argument values.
    actions: [fn_ids, {arg_0: arg_0_ids, ..., arg_n: arg_n_ids}], where
      each value is a tensor of shape [num_batch] representing the selected
      argument or actions ids. The argument id will be -1 if the argument is
      not available for a specific (state, action) pair.
  """
  def compute_log_probs(probs, labels):
    return tf.log(tf.gather(probs, labels))

  fn_id, arg_ids = actions
  fn_pi, arg_pis = policy
  fn_pi = mask_invalid_actions(batch_valid_actions, fn_pi)
  fn_log_prob = compute_log_probs(fn_pi, fn_id)

  # TODO logging for each arg_type
  total = fn_log_prob
  for arg_type in arg_type in actions.TYPES:
    arg_id = arg_ids[arg_type]
    arg_pi = arg_pis[arg_type]
    arg_log_prob = compute_log_probs(arg_pi, tf.maximum(arg_id, 1e-10))
    # Mask argument log prob if the argument is not relevant
    arg_log_prob *= (arg_id != -1)
    total += arg_log_prob

  return total

def compute_a2c_loss():
  pass


def update():
  pass
