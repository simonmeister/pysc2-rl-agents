import os

import tensorflow as tf

# See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/distributions/categorical.py
from tensorflow.contrib.distributions import Categorical

from pysc2.lib.actions import TYPES as ACTION_TYPES

from rl.networks.fully_conv import FullyConv


class A2CAgent():
  """A2C agent.

  Run build(...) first, then init() or load(...).
  """
  def __init__(self,
               sess,
               network_cls=FullyConv,
               value_loss_weight=0.5,
               entropy_weight=1e-3,
               learning_rate=1e-4):
    self.sess = sess
    self.network_cls = network_cls
    self.value_loss_weight = value_loss_weight
    self.entropy_weight = entropy_weight
    self.learning_rate = learning_rate
    self.train_step = 0

  def build(self, static_shape_channels, resolution, scope, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
      self._build(static_shape_channels, resolution)
      variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)
      self.saver = tf.train.Saver(variables)
      self.init_op = tf.variables_initializer(variables)
      train_summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
      self.train_summary_op = tf.summary.merge(train_summaries)

  def _build(self, static_shape_channels, resolution):
    """Create tensorflow graph for A2C agent.

    Args:
      static_shape_channels: dict with keys
        {screen, minimap, flat, available_actions}.
      resolution: Integer resolution of screen and minimap.
    """
    ch = static_shape_channels
    res = resolution
    screen = tf.placeholder(tf.float32, [None, res, res, ch['screen']],
                            'input_screen')
    minimap = tf.placeholder(tf.float32, [None, res, res, ch['minimap']],
                             'input_minimap')
    flat = tf.placeholder(tf.float32, [None, ch['flat']],
                          'input_flat')
    available_actions = tf.placeholder(tf.float32, [None, ch['available_actions']],
                                       'input_available_actions')
    advs = tf.placeholder(tf.float32, [None], 'advs')
    returns = tf.placeholder(tf.float32, [None], 'returns')
    self.screen = screen
    self.minimap = minimap
    self.flat = flat
    self.advs = advs
    self.returns = returns
    self.available_actions = available_actions

    policy, value = self.network_cls().build(
        screen, minimap, flat)
    self.policy = policy
    self.value = value

    fn_id = tf.placeholder(tf.int32, [None, ], 'fn_id')
    arg_ids = {
        k: tf.placeholder(tf.int32, [None], 'arg_{}_id'.format(k.id))
        for k in policy[1].keys()}
    actions = (fn_id, arg_ids)
    self.actions = actions

    log_probs = compute_policy_log_probs(available_actions, policy, actions)

    policy_loss = -tf.reduce_mean(advs * log_probs)
    value_loss = tf.reduce_mean(tf.square(returns - value) / 2)
    entropy = tf.reduce_mean(compute_policy_entropy(policy))

    loss = (policy_loss
            + value_loss * self.value_loss_weight
            - entropy * self.entropy_weight)

    tf.summary.scalar('loss/policy', policy_loss)
    tf.summary.scalar('loss/value', value_loss)
    tf.summary.scalar('loss/entropy', entropy)
    tf.summary.scalar('loss/total', loss)

    # TODO gradient clipping? (see baselines/a2c/a2c.py)

    # TODO support learning rate schedule
    opt = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate)
    self.train_op = opt.minimize(loss)

    self.samples = sample_actions(available_actions, policy)

  def get_obs_feed(self, obs):
    return {self.screen: obs['screen'],
            self.minimap: obs['minimap'],
            self.flat: obs['flat'],
            self.available_actions: obs['available_actions']}

  def get_actions_feed(self, actions):
    feed_dict = {self.actions[0]: actions[0]}
    feed_dict.update({v: actions[1][k] for k, v in self.actions[1].items()})
    return feed_dict

  def train(self, obs, actions, returns, advs, summary=False):
    """
    Args:
      obs: dict of preprocessed observation arrays, with num_batch elements
        in the first dimensions.
      actions: see `compute_total_log_probs`.
      returns: array of shape [num_batch].
      advs: array of shape [num_batch].
      summary: Whether to return a summary.

    Returns:
      summary: (agent_step, Summary) or None.
    """
    feed_dict = self.get_obs_feed(obs)
    feed_dict.update(self.get_actions_feed(actions))
    feed_dict.update({
        self.returns: returns,
        self.advs: advs})

    ops = [self.train_op]

    if summary:
      ops.append(self.train_summary_op)

    res = self.sess.run(ops, feed_dict=feed_dict)
    agent_step = self.train_step
    self.train_step += 1

    if summary:
      return (agent_step, res[-1])

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

  def init(self):
    self.sess.run(self.init_op)

  def save(self, path, step=None):
    os.makedirs(path, exist_ok=True)
    step = step or self.train_step
    print("Saving agent to %s, step %d" % (path, step))
    ckpt_path = os.path.join(path, 'model.ckpt')
    self.saver.save(self.sess, ckpt_path, global_step=step)

  def load(self, path):
    ckpt = tf.train.get_checkpoint_state(path)
    self.saver.restore(self.sess, ckpt.model_checkpoint_path)
    self.train_step = int(ckpt.model_checkpoint_path.split('-')[-1])
    print("Loaded agent at train_step %d" % self.train_step)


def mask_unavailable_actions(available_actions, fn_pi):
  fn_pi *= available_actions
  fn_pi /= tf.reduce_sum(fn_pi, axis=1, keep_dims=True)
  return fn_pi


def compute_policy_entropy(policy):
  # TODO compute the entropy only for the applicable arguments? (see compute_policy_log_probs)

  def compute_entropy(probs):
    dist = Categorical(probs=probs)
    return dist.entropy()

  fn_pi, arg_pis = policy
  entropy = compute_entropy(fn_pi)

  for arg_pi in arg_pis.values():
    entropy += compute_entropy(arg_pi)

  return entropy


def sample_actions(available_actions, policy):
  """Sample function ids and arguments from a predicted policy."""

  def sample(probs):
    dist = Categorical(probs=probs)
    return dist.sample()

  fn_pi, arg_pis = policy
  fn_pi = mask_unavailable_actions(available_actions, fn_pi)
  fn_samples = sample(fn_pi)

  arg_samples = dict()
  for arg_type, arg_pi in arg_pis.items():
    arg_samples[arg_type] = sample(arg_pi)

  return fn_samples, arg_samples


def compute_policy_log_probs(available_actions, policy, actions):
  """Compute action log probabilities given predicted policies and selected
  actions.

  Args:
    available_actions: one-hot (in last dimenson) tensor of shape
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
    probs = tf.maximum(probs, 1e-10)
     # Gather arbitrary id for unused arguments (log probs will be masked)
    labels = tf.maximum(labels, 0)
    return tf.log(tf.gather(probs, labels, axis=1))

  fn_id, arg_ids = actions
  fn_pi, arg_pis = policy
  fn_pi = mask_unavailable_actions(available_actions, fn_pi)
  fn_log_prob = compute_log_probs(fn_pi, fn_id)

  # TODO logging for each arg_type
  total = fn_log_prob
  for arg_type in ACTION_TYPES:
    arg_id = arg_ids[arg_type]
    arg_pi = arg_pis[arg_type]
    arg_log_prob = compute_log_probs(arg_pi, arg_id)
    # Mask argument log prob if the argument is not relevant
    arg_log_prob *= (arg_id != -1)
    total += arg_log_prob

  return total
