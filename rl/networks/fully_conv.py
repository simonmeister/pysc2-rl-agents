import tensorflow as tf
from tensorflow.contrib import layers

from pysc2.lib import actions
from pysc2.lib import features

from rl.algorithms.utils import sample


class FullyConv():
  """FullyConv network from https://arxiv.org/pdf/1708.04782.pdf.

  Both, NHWC and NCHW data formats are supported for the network
  computations. Inputs and outputs are always in NHWC.
  """

  def __init__(self, data_format='NHWC'):
    self.data_format = data_format

  def input_conv(in):
    conv1 = layers.conv2d(
        in, 16,
        kernel_size=5,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        data_format=self.data_format,
        scope="%s/conv1" % name,
        trainable=self.trainable)
    conv2 = layers.conv2d(
        conv1, 32,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        data_format=self.data_format,
        scope="%s/conv2" % name,
        trainable=self.trainable)
    return conv2

  def input_fc(in):
    # TODO find out correct number of channels
    return layers.fully_connected(in, 256, activation_fn=tf.tanh)

  def non_spatial_output(self, in, channels):
    logits = layers.fully_connected(in, channels, activation_fn=None)
    return tf.nn.softmax(logits)

  def spatial_output(self, in):
    logits = layers.conv2d(in, 1, kernel_size=1, stride=1, activation_fn=None)
    logits = layers.flatten(logits)
    return tf.nn.softmax(logits)

  def concat2d(self, lst):
    if self.data_format == 'NCHW':
      return tf.concat(lst, axis=1)
    return tf.concat(lst, axis=3)

  def broadcast_along_channels(self, flat):
    if self.data_format == 'NCHW':
      return tf.tile(tf.expand_dims(tf.expand_dims(flat, 2), 3),
                     [1, 1, size2d[0], size2d[1]])
    return tf.tile(tf.expand_dims(tf.expand_dims(flat, 1), 2),
                   [1, size2d[0], size2d[1], 1])

  def get_size2d(self, feature_map):
    if self.data_format == 'NCHW':
      return map2d[2:]
    return map2d[1:3]

  def to_nhwc(self, map2d):
    if self.data_format == 'NCHW':
      return tf.transpose(map2d, [0, 2, 3, 1])
    return map2d

  def from_nhwc(self, map2d):
    if self.data_format == 'NCHW':
      return tf.transpose(map2d, [0, 3, 1, 2])
    return map2d

  def build(self, screen_input, minimap_input, non_spatial_input):
    screen_out = input_conv(self.from_nhwc(screen_input))
    minimap_out = input_conv(self.from_nhwc(minimap_input))
    non_spatial_out = input_fc(non_spatial_input)

    size2d = self.get_size2d(screen_out)
    broadcast_out = self.broadcast_along_channels(non_spatial_out, size2d)

    state_out = self.concat2d([screen_out, minimap_out, broadcast_out])
    flat_out = layers.flatten(self.to_nhwc(state_out))
    fc = layers.fully_connected(flat_out, 256, activation_fn=tf.nn.relu)

    value = layers.fully_connected(fc, 1, activation_fn=None)

    # TODO for minigames, only model available actions?
    fn_out = self.non_spatial_output(fc, len(actions.FUNCTIONS))
    args_out = dict()
    for name, arg_type in actions.TYPES._asdict().items():
      # HACK: we should infer the point type automatically
      if name in ['minimap', 'screen', 'screen2']:
        arg_out = (True, self.to_nhwc(self.spatial_output(state_out)))
      else:
        arg_out = (False, self.non_spatial_output(fc, arg_type.sizes[0]))
      args_out[arg_type] = arg_out

    policy = (fn_out, args_out)

    return policy, value



# TODO move this somewhere else and re-organize the code
# A2C

def mask_invalid_actions(batch_valid_actions, fn_pi):
  fn_pi *= batch_valid_actions
  fn_pi /= tf.reduce_sum(fn_pi, axis=1, keep_dims=True)
  return fn_pi

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


def step(sampling_fn, obs_list, policy, value):
  valid_actions_list = [obs.observation['available_actions'] for obs in obs_list]
  batch_valid_actions = np.stack(valid_actions_list, axis=0)

  fn_samples, arg_samples = sample_independent(batch_valid_actions, policy)
  arg_samples_np, fn_samples_np, value_np = sess.run(
      [fn_samples, arg_samples, value], feed_dict=get_feed_dict(obs))

  actions_list = []
  for n in range(samples_np.shape[0]):
    a_0 = fn_samples_np[n]
    a_l = []
    for arg_type in actions.FUNCTIONS._func_list[a_0].args:
      a_l.append(arg_samples_np[arg_type])
    action = actions.FunctionCall(a_0, a_l)
    actions_list.append(action)

  return actions_list, values


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
