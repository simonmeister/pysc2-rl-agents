import tensorflow as tf
from tensorflow.contrib import layers

from pysc2.lib import actions
from pysc2.lib import features

from rl.pre_processing import is_spatial_action, NUM_FUNCTIONS


class FullyConv():
  """FullyConv network from https://arxiv.org/pdf/1708.04782.pdf.

  Both, NHWC and NCHW data formats are supported for the network
  computations. Inputs and outputs are always in NHWC.
  """

  def __init__(self, data_format='NHWC'):
    self.data_format = data_format

  def input_conv(self, x, name):
    conv1 = layers.conv2d(
        x, 16,
        kernel_size=5,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        data_format=self.data_format,
        scope="%s/conv1" % name)
    conv2 = layers.conv2d(
        conv1, 32,
        kernel_size=3,
        stride=1,
        padding='SAME',
        activation_fn=tf.nn.relu,
        data_format=self.data_format,
        scope="%s/conv2" % name)
    return conv2

  def input_fc(self, x):
    # TODO find out correct number of channels
    return layers.fully_connected(x, 256, activation_fn=tf.tanh)

  def non_spatial_output(self, x, channels):
    logits = layers.fully_connected(x, channels, activation_fn=None)
    return tf.nn.softmax(logits)

  def spatial_output(self, x):
    logits = layers.conv2d(x, 1, kernel_size=1, stride=1, activation_fn=None)
    logits = layers.flatten(logits)
    return tf.nn.softmax(logits)

  def concat2d(self, lst):
    if self.data_format == 'NCHW':
      return tf.concat(lst, axis=1)
    return tf.concat(lst, axis=3)

  def broadcast_along_channels(self, flat, size2d):
    if self.data_format == 'NCHW':
      return tf.tile(tf.expand_dims(tf.expand_dims(flat, 2), 3),
                     tf.stack([1, 1, size2d[0], size2d[1]]))
    return tf.tile(tf.expand_dims(tf.expand_dims(flat, 1), 2),
                   tf.stack([1, size2d[0], size2d[1], 1]))

  def get_size2d(self, map2d):
    if self.data_format == 'NCHW':
      size = tf.shape(map2d)[2:]
    else:
      size = tf.shape(map2d)[1:3]
    return tf.unstack(size)

  def to_nhwc(self, map2d):
    if self.data_format == 'NCHW':
      return tf.transpose(map2d, [0, 2, 3, 1])
    return map2d

  def from_nhwc(self, map2d):
    if self.data_format == 'NCHW':
      return tf.transpose(map2d, [0, 3, 1, 2])
    return map2d

  def build(self, screen_input, minimap_input, non_spatial_input):
    screen_out = self.input_conv(self.from_nhwc(screen_input), 'screen')
    minimap_out = self.input_conv(self.from_nhwc(minimap_input), 'minimap')
    non_spatial_out = self.input_fc(non_spatial_input)

    size2d = self.get_size2d(screen_out)
    broadcast_out = self.broadcast_along_channels(non_spatial_out, size2d)

    state_out = self.concat2d([screen_out, minimap_out, broadcast_out])
    flat_out = layers.flatten(self.to_nhwc(state_out))
    fc = layers.fully_connected(flat_out, 256, activation_fn=tf.nn.relu)

    value = layers.fully_connected(fc, 1, activation_fn=None)
    value = tf.reshape(value, [-1])

    # TODO for minigames, only model available actions?
    fn_out = self.non_spatial_output(fc, NUM_FUNCTIONS)
    args_out = dict()
    for arg_type in actions.TYPES:
      if is_spatial_action[arg_type]:
        arg_out = self.to_nhwc(self.spatial_output(state_out))
      else:
        arg_out = self.non_spatial_output(fc, arg_type.sizes[0])
      args_out[arg_type] = arg_out

    policy = (fn_out, args_out)

    return policy, value
