import tensorflow as tf
from tensorflow.contrib import distributions


class A2CAgent():
  def __init__(self,
               sess,
               network_cls=FullyConv,
               value_loss_weight=0.5,
               entropy_weight=0.01):
    self.sess = sess
    self.network_cls = network_cls
    self.value_loss_weight = value_loss_weight

  def build(self, static_shape_channels):
    """Create tensorflow graph for A2C agent.

    Args:
      static_shape_channels: dict with keys
        {screen, minimap, flat, valid_actions}.
    """
    ch = static_shape_channels
    screen = tf.placeholder(tf.float32, [None, None, None, ch['screen']],
                            'input_screen')
    minimap = tf.placeholder(tf.float32, [None, None, None, ch['minimap']],
                             'input_minimap')
    flat = tf.placeholder(tf.float32, [None, ch['flat']],
                          'input_flat')
    valid_actions = tf.placeholder(tf.float32, [None, ch['valid_actions_channels'],
                                   'input_valid_actions')
    advs = tf.placeholder(tf.float32, [None], 'advs')
    returns = tf.placeholder(tf.float32, [None], 'returns')
    self.screen = screen
    self.minimap = minimap
    self.flat = flat
    self.advs = advs
    self.returns = returns
    self.valid_actions = valid_actions

    policy, value = self.network_cls().build(
        screen, minimap, flat)
    self.policy = policy
    self.value = value

    fn_id = tf.placeholder(tf.int32, [None, ], 'fn_id')
    arg_ids = {
        k: tf.placeholder(tf.int32, [None], 'arg_{}_id'.format(k))
        for k in policy[1].keys()}
    actions = (fn_id, arg_ids)
    self.actions = actions

    log_probs = compute_policy_log_probs(valid_actions, policy, actions)

    policy_loss = -tf.reduce_mean(advs * log_probs)
    value_loss = tf.reduce_mean(tf.square(returns - values) / 2)
    entropy = tf.reduce_mean(compute_policy_entropy(policy))

    loss = (policy_loss
            + value_loss * value_loss_weight
            - entropy * entropy_weight)

    opt = tf.train.RMSPropOptimizer(learning_rate=2e-4)
    self.train_op = opt.minimize(loss)

    self.samples = sample_actions(valid_actions, policy, size) # TODO size = (height, width) of screen/minimap

  def get_obs_feed(self, obs):
    return {self.screen: obs['screen'],
            self.minimap: obs['minimap'],
            self.flat: obs['flat'],
            self.valid_actions: obs['valid_actions'}

  def get_actions_feed(self, actions):
    feed_dict = {self.actions[0]: actions[0]}
    feed_dict.update({v: actions[k] for v in self.actions[1]})
    return feed_dict

  def train(self, obs, actions, returns, advs):
    """
    Args:
      obs: dict of preprocessed observation arrays, with num_batch elements
        in the first dimensions.
      actions: see `compute_total_log_probs`
      returns: array of shape [num_batch]
      advs: array of shape [num_batch]
    """
    feed_dict = self.get_obs_feed(obs)
    feed_dict.update(self.get_actions_feed(actions))
    feed_dict.update({
        self.returns: returns,
        self.advs: advs})

    ops = [self.train_op]

    # TODO add summary ops

    self.sess.run(ops, feed_dict=feed_dict)

  def step(self, obs):
    """
    Args:
      obs: dict of preprocessed observation arrays, with num_batch elements
        in the first dimensions.

    Returns:
      actions: arrays (see `compute_total_log_probs`)
      values: array of shape [num_batch] containing value estimates.
    """
    feed_dict = self.get_obs_feed(obs)
    return self.sess.run([self.samples, self.value], feed_dict=feed_dict)

  def get_value(self, obs):
    return self.sess.run(
        self.value,
        feed_dict=self.get_obs_feed(obs))


def mask_invalid_actions(valid_actions, fn_pi):
  fn_pi *= valid_actions
  fn_pi /= tf.reduce_sum(fn_pi, axis=1, keep_dims=True)
  return fn_pi



def compute_policy_entropy(policy):

  def compute_entropy(probs):
    dist = distributions.Categorical(probs=probs)
    return dist.entropy()

  # TODO is it correct to assume additive entropy here?
  # TODO should we compute the entropy only for the applicable arguments? (see compute_policy_log_probs)
  fn_pi, arg_pis = policy
  entropy = compute_entropy(fn_pi)

  for arg_pi in arg_pis.values():
    entropy += compute_entropy(arg_pi)

  return entropy


def sample_actions(valid_actions, policy):
  """Sample function ids and arguments from a predicted policy."""

  def sample(probs):
    dist = distributions.Categorical(probs=probs)
    return dist.sample()

  fn_pi, arg_pis = policy
  fn_pi = mask_invalid_actions(valid_actions, fn_pi)
  fn_samples = sample(fn_pi)

  arg_samples = dict()
  for arg_type, arg_pi in arg_pis.items():
    arg_samples[arg_type] = sample(arg_pi)

  return fn_samples, arg_samples


def compute_policy_log_probs(valid_actions, policy, actions):
  """Compute action log probabilities given predicted policies and selected
  actions.

  Args:
    valid_actions: one-hot (in last dimenson) tensor of shape
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
  fn_pi = mask_invalid_actions(valid_actions, fn_pi)
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
