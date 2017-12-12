import tensorflow as tf


# TODO the code here is still incomplete (many snippets do not fit together yet)


def mask_invalid_actions(batch_valid_actions, fn_pi):
  fn_pi *= batch_valid_actions
  fn_pi /= tf.reduce_sum(fn_pi, axis=1, keep_dims=True)
  return fn_pi


# TODO implement sample (see baselines/a2c/utils.py)


def sample_independent(batch_valid_actions, policy, size): # TODO size
  """Sample function ids and arguments from a predicted policy."""
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


class A2CAgent():
  def __init__(self, sess, network_cls=FullyConv):
    self.sess = sess
    self.network_cls = network_cls

  def build(self):
    """Create tensorflow graph for A2C agent."""
    screen = tf.placeholder(tf.float32, [], 'input_screen') # TODO static shapes
    minimap = tf.placeholder(tf.float32, [], 'input_minimap')
    flat = tf.placeholder(tf.float32, [], 'input_flat')
    self.screen = screen
    self.minimap = minimap
    self.flat = flat
    self.policy, self.value = self.network_cls().build(
        screen, minimap, flat)

  def get_obs_feed(self, obs):
    return {self.screen: obs[0],
            self.minimap: obs[1],
            self.flat: obs[2]}

  def train(self, batch_valid_actions, obs, actions, returns, advs):
    """
    Args:
      obs: tuple with preprocessed observation arrays, with num_batch elements
        in the first dimensions.
      actions: see `compute_total_log_probs`
      returns: array of shape [num_batch]
      advs: array of shape [num_batch]
    """
    policy, value = self.sess.run(
        [self.policy, self.value],
        feed_dict=self.get_obs_feed(obs))
    log_probs = compute_total_log_probs(batch_valid_actions, policy, value)
    # TODO compute loss

  def step(self, batch_valid_actions, obs):
    """
    Args:
      batch_valid_actions: one-hot array of shape [num_batch, NUM_FUNCTIONS].
      obs: tuple with preprocessed observation arrays, with num_batch elements
        in the first dimensions.

    Returns:
      actions: arrays (see `compute_total_log_probs`)
      values: array of shape [num_batch] containing value estimates.
    """
    fn_samples, arg_samples = sample_independent(batch_valid_actions, self.policy)
    arg_samples_np, fn_samples_np, value_np = sess.run(
        [fn_samples, arg_samples, self.value],
        feed_dict=self.get_obs_feed(obs))

    actions = (fn_samples_np, arg_samples_np)
    return actions, values

  def get_value(self, obs):
    return self.sess.run(
        [self.value],
        feed_dict=self.get_obs_feed(obs))


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
